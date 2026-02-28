"""
- [x] support streaming for planner & reporter
"""

import asyncio

from agents import trace
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utu.tools.memory_toolkit import VectorMemoryToolkit

from ...config import AgentConfig, ConfigLoader
from ...utils import AgentsUtils, get_logger
from ...agents.common import QueueCompleteSentinel
from ...agents.simple_agent import SimpleAgent
from ...tools.text2sql import mysql_excuted_sql_tool
from ...tools.text2sql import sqlite_excuted_sql_tool
from ...tools.text2sql.unified_schemalink_valuelink import unified_schemalink_with_valuelink
from ...utils import FileUtils, PrintUtils

from ...agents.orchestra import (
    AnalysisResult,
    CreatePlanResult,
    OrchestraTaskRecorder,
    PlannerAgent,
    ReporterAgent,
    SimpleWorkerAgent,
    WorkerResult,
)

logger = get_logger(__name__)


class Text2sqlAgent:
    """Text2SQL Agent for multi-database SQL generation and execution"""

    # Template for task recorder updates
    _TASK_TEMPLATE = r"""Original Problem:
{problem}

Plan:
{plan}

Previous Trajectory:
{trajectory}
""".strip()

    def __init__(self):
        self._agent_cache = {}  # Cache for different types of agents
        self.prompts = FileUtils.load_prompts("ragref/text2sql/gensql.yaml")  # Load prompts once during initialization
        
    @property
    def name(self) -> str:
        return "text2sqlagent"
    
    async def text2sql(self, task_recorder: OrchestraTaskRecorder):
        PrintUtils.print_info(f"\n>> Text2sqlAgent (Multi-Database Support)...")
        # Step1: schema link + value link (multi-database support)
        task = task_recorder.plan.todo[-1]
        question = task.task

        # Use unified schema link function (pass task_recorder for streaming intermediate results)
        schema_result = await unified_schemalink_with_valuelink(question, task_recorder)

        if not schema_result["db_groups"]:
            error_msg = """
Text2SQL Configuration Error

No relevant database tables found. Please ensure:
1. A knowledge base containing relational databases is selected in the frontend [📚 Knowledge Base] selector
2. VECTOR_STORE_PATH is set in your .env file to specify the vector store path
3. Database table schema information has been indexed in the knowledge base
"""
            PrintUtils.print_info(error_msg)
            raise ValueError(error_msg)

        self._display_schema_link_summary(schema_result, task_recorder)

        # Step2: sql_generate and execute (for each database)
        results = []

        for db_id, db_info in schema_result["db_groups"].items():
            db_type = db_info["db_type"]
            tables_str = "\n\n".join(db_info["tables"])
            table_names = db_info["table_names"]

            PrintUtils.print_info(f"\n>> Generating SQL for database {db_type}")
            PrintUtils.print_info(f"   db_id: {db_id[:80]}...")
            PrintUtils.print_info(f"   Tables: {', '.join(table_names)}")

            # CRITICAL: Reconfigure database connection before SQL execution
            # This ensures the correct database is used when multiple databases are processed
            if db_type == "excel":
                # Excel tables are in rag_demo.sqlite
                from ...tools.text2sql.sqlite_tools import set_sqlite_config
                rag_demo_path = os.getenv("RELATIONAL_DB_PATH", "rag_data/relational_database/rag_demo.sqlite")
                if not rag_demo_path.startswith('/'):
                    rag_demo_path = os.path.abspath(rag_demo_path)
                set_sqlite_config(file_path=rag_demo_path)
                PrintUtils.print_info(f"   [Pre-execution] Set SQLite config to: {rag_demo_path}")
            elif db_type == "sqlite":
                # Regular SQLite database - extract path from db_id
                from ...tools.text2sql.sqlite_tools import set_sqlite_config, parse_sqlite_connection_string
                conn_info = parse_sqlite_connection_string(db_id)
                file_path = conn_info.get("file_path")
                if file_path:
                    set_sqlite_config(file_path=file_path)
                    PrintUtils.print_info(f"   [Pre-execution] Set SQLite config to: {file_path}")
            elif db_type == "mysql":
                # MySQL database - extract connection info from db_id
                from ...tools.text2sql.mysql_tools import set_mysql_config, parse_mysql_connection_string
                from ...rag.api.database import get_db, KBSourceConfig

                first_table = table_names[0]
                source_identifier = f"{db_id}:{first_table}"

                db_session = next(get_db())
                source_config = db_session.query(KBSourceConfig).filter(
                    KBSourceConfig.source_identifier == source_identifier
                ).first()

                if source_config and source_config.config:
                    config = source_config.config
                    conn_info = parse_mysql_connection_string(db_id)

                    host = config.get("host") or conn_info.get("host")
                    user = config.get("username") or conn_info.get("user")
                    password = config.get("password") or ""
                    port = config.get("port") or conn_info.get("port", 3306)
                    database = config.get("database") or db_id.split('/')[-1]

                    set_mysql_config(host=host, user=user, password=password, port=port, database=database)
                    PrintUtils.print_info(f"   [Pre-execution] Set MySQL config to: {host}:{port}/{database}")

            database_dialect = "MySql" if db_type == "mysql" else "Sqlite"  # Excel tables use SQLite backend

            few_shot_examples = getattr(task_recorder, '_few_shot_examples', [])
            sql_prompt = self.get_prompt(database_dialect, tables_str, question, db_id, table_names, few_shot_examples)

            self._display_schema_details(db_type, db_id, table_names, sql_prompt,task_recorder)

            # Map excel to sqlite for agent selection (excel uses sqlite backend)
            agent_db_type = "sqlite" if db_type == "excel" else db_type
            result = await self.sql_gen_exec(sql_prompt, task_recorder, agent_db_type, db_type)

            results.append({
                "db_id": db_id,
                "db_type": db_type,
                "table_names": table_names,
                "result": result
            })

        final_result = self._merge_results(results, question)

        self.update_task_recorder(task_recorder, question, final_result)

    def _get_or_create_agent(self, db_type: str):
        """Get or create SQL agent based on database type
        
        Use cache to avoid duplicate creation

        Agent configuration is loaded from: configs/agents/ragref/text2sql/sql_executor.yaml
        """
        if db_type not in self._agent_cache:
            config = ConfigLoader.load_agent_config("ragref/text2sql/sql_executor")

            tool = sqlite_excuted_sql_tool if db_type == "sqlite" else mysql_excuted_sql_tool

            self._agent_cache[db_type] = SimpleAgent(
                config=config,
                name=f"sql generate and execute agent ({db_type})",
                tools=[tool],
            )
            PrintUtils.print_info(f"   Created SQL agent for {db_type}")
        return self._agent_cache[db_type]

    async def sql_gen_exec(self, sql_prompt, task_recorder, agent_db_type, display_db_type):
        """Generate and execute SQL query

        Args:
            sql_prompt: SQL generation prompt
            task_recorder: Task recorder for streaming
            agent_db_type: Database type for agent selection (sqlite/mysql)
            display_db_type: Database type for display (excel/sqlite/mysql)
        """
        try:
            agent = self._get_or_create_agent(agent_db_type)

            result = agent.run_streamed(sql_prompt)

            # Stream agent events and extract generated SQL
            generated_sql = None
            async for event in result.stream_events():
                task_recorder._event_queue.put_nowait(event)

                # Extract SQL from agent messages
                if hasattr(event, 'data') and hasattr(event.data, 'content'):
                    content = event.data.content
                    if '```sql' in content and generated_sql is None:
                        import re
                        sql_match = re.search(r'```sql\n(.*?)```', content, re.DOTALL)
                        if sql_match:
                            generated_sql = sql_match.group(1).strip()

            if generated_sql:
                self._display_generated_sql(generated_sql, display_db_type, task_recorder)

            final_output = str(result.final_output)
            self._display_sql_execution_result(final_output, display_db_type, task_recorder)

            return final_output
        except Exception as e:
            print(e)
            return str(e)

    def get_prompt(self, database_dialect, table_info, query, db_id=None, table_names=None, few_shot_examples=None):
        sqlite_syntax_rules = """    - **Identifier Quotes**: ALWAYS use double quotes (") for table names and column names
    - **Example**: SELECT "column_name" FROM "table_name" WHERE "id" = 1
    - **Special characters**: Table/column names with Chinese characters, spaces, or special symbols MUST be quoted
    - **Correct**: SELECT "职位代码", "招聘人数" FROM "excel_1_2025年吉安市事业单位公开_test2_2025年吉安市事业单位公开招聘条件"
    - **Wrong**: SELECT 职位代码 FROM excel_1_2025年吉安市事业单位公开_test2_2025年吉安市事业单位公开招聘条件"""

        mysql_syntax_rules = """    - **Identifier Quotes**: ALWAYS use backticks (`) for table names and column names
    - **Example**: SELECT `column_name` FROM `table_name` WHERE `id` = 1
    - **Special characters**: Table/column names with Chinese characters, spaces, or special symbols MUST be quoted
    - **Correct**: SELECT `职位代码`, `招聘人数` FROM `招聘信息表`
    - **Wrong**: SELECT 职位代码 FROM 招聘信息表"""

        syntax_rules = sqlite_syntax_rules if database_dialect == "Sqlite" else mysql_syntax_rules

        if few_shot_examples:
            examples_text = "\n\n## Reference Examples from Memory:\n"
            for i, example in enumerate(few_shot_examples, 1):
                examples_text += f"\n### Example {i}:\n{example}\n"
            few_shot_block = examples_text + "\n"
        else:
            few_shot_block = ""

        generate_sql_prompt = (
            self.prompts["sql_generate_prompt"]
            .replace("{database_dialect}", database_dialect)
            .replace("{syntax_rules}", syntax_rules)
            .replace("{table_info}", table_info)
            .replace("{few_shot_examples}", few_shot_block)
            .replace("{query}", query)
        )

        return generate_sql_prompt

    def _merge_results(self, results, question):
        """Merge query results from multiple databases

        Args:
            results: [{"db_id": ..., "db_type": ..., "table_names": [...], "result": ...}, ...]
            question: User question

        Returns:
            Merged result string
        """
        if not results:
            return "No relevant databases or tables found."

        if len(results) == 1:
            # Single database, return result directly
            return results[0]["result"]

        # Multiple databases, need to merge results
        merged_output = f"## Multi-Database Query Results (Queried {len(results)} databases)\n\n"
        merged_output += f"**User Question**: {question}\n\n"

        for i, res in enumerate(results, 1):
            db_type = res["db_type"]
            db_id = res["db_id"]
            table_names = res["table_names"]
            result = res["result"]

            merged_output += f"### Database {i}: {db_type}\n"
            merged_output += f"- **Database ID**: {db_id[:100]}{'...' if len(db_id) > 100 else ''}\n"
            merged_output += f"- **Queried Tables**: {', '.join(table_names)}\n\n"
            merged_output += f"**Query Results**:\n{result}\n\n"
            merged_output += "---\n\n"

        return merged_output

    def update_task_recorder(self, task_recorder: OrchestraTaskRecorder, question, sqlex_result):
        result_streaming = WorkerResult()
        result_streaming.task = question
        result_streaming.output = sqlex_result

        task_recorder.add_worker_result(result_streaming)
        str_plan = task_recorder.get_plan_str()
        str_traj = task_recorder.get_trajectory_str()
        new_task = self._TASK_TEMPLATE.format(
            problem=task_recorder.task,
            plan=str_plan,
            trajectory=str_traj,
        )
        task_recorder.task = new_task

    def _display_schema_link_summary(self, schema_result: dict, task_recorder: OrchestraTaskRecorder):
        """Display schema link summary: which tables were recalled from which databases"""
        summary_lines = ["## 📊 Schema Link - Table Recall Summary\n"]

        total_tables = sum(len(db_info["table_names"]) for db_info in schema_result["db_groups"].values())
        summary_lines.append(f"**Total Recalled**: {total_tables} tables from {len(schema_result['db_groups'])} database(s)\n")

        for idx, (db_id, db_info) in enumerate(schema_result["db_groups"].items(), 1):
            db_type = db_info["db_type"]
            table_names = db_info["table_names"]

            db_display = db_id if len(db_id) <= 80 else f"{db_id[:77]}..."

            summary_lines.append(f"\n### Database {idx}: {db_type}")
            summary_lines.append(f"- **Source**: `{db_display}`")
            summary_lines.append(f"- **Tables** ({len(table_names)}): {', '.join([f'`{t}`' for t in table_names])}")

        summary = "\n".join(summary_lines)

        PrintUtils.print_and_stream_tool(
            summary,
            event_queue=task_recorder._event_queue,
            tool_name="schema_link_recall"
        )

    def _display_schema_details(self, db_type: str, db_id: str, table_names: list, sql_prompt: str, task_recorder: OrchestraTaskRecorder):
        """Display detailed schema information without truncation"""
        header_lines = [
            f"## 🗂️  Schema Details - {db_type.upper()} Database\n",
            f"**Database**: `{db_id}`",
            f"**Tables**: {', '.join([f'`{t}`' for t in table_names])}\n",
            "### Table Schemas for gensql:\n"
        ]

        detail_output = "\n".join(header_lines) + "\n"+ str(sql_prompt)

        PrintUtils.print_and_stream_tool(
            detail_output,
            event_queue=task_recorder._event_queue,
            tool_name="schema_details"
        )

    def _display_generated_sql(self, sql: str, db_type: str, task_recorder: OrchestraTaskRecorder):
        """Display the generated SQL statement before execution"""
        output = f"""## 🔍 Generated SQL ({db_type})

```sql
{sql}
```

*Executing SQL query...*
"""

        PrintUtils.print_and_stream_tool(
            output,
            event_queue=task_recorder._event_queue,
            tool_name="sql_generation"
        )

    def _display_sql_execution_result(self, final_output: str, db_type: str, task_recorder: OrchestraTaskRecorder):
        """Display the SQL execution result and Agent's final output"""
        output = f"""## ✅ SQL Execution Result ({db_type})

{final_output}
"""

        PrintUtils.print_and_stream_tool(
            output,
            event_queue=task_recorder._event_queue,
            tool_name="sql_execution_result"
        )
    

class OrchestraReactSqlAgent:
    def __init__(self, config: AgentConfig | str):
        """
        Initialize the text2sql orchestra agent
        """
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config
        self.max_turn_num = config.max_turns  # Load from config instead of hardcoding

        self.planner_agent = PlannerAgent(config)
        self.reporter_agent = ReporterAgent(config)
        self.worker_agents = self._setup_workers()

        self._memory_toolkit: "VectorMemoryToolkit | None" = None

    def set_memory_toolkit(self, memory_toolkit: "VectorMemoryToolkit") -> None:
        """Set the memory toolkit for this agent.

        Args:
            memory_toolkit: VectorMemoryToolkit instance for memory operations.
        """
        self._memory_toolkit = memory_toolkit

    @property
    def memory_toolkit(self) -> "VectorMemoryToolkit | None":
        """Get the memory toolkit if set."""
        return self._memory_toolkit

    def set_planner(self, planner: PlannerAgent):
        self.planner_agent = planner

    def _setup_workers(self) -> dict[str, SimpleWorkerAgent]:
        workers = {}
        workers["Text2sqlAgent"] = Text2sqlAgent()
        return workers

    async def run(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        task_recorder = self.run_streamed(input, trace_id)
        async for _ in task_recorder.stream_events():
            pass
        return task_recorder

    def run_streamed(self, input: str, trace_id: str = None, use_memory: bool = True) -> OrchestraTaskRecorder:
        # TODO: error_tracing
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        task_recorder = OrchestraTaskRecorder(task=input, trace_id=trace_id)
        # Kick off the actual agent loop in the background and return the streamed result object.
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, use_memory=use_memory))
        return task_recorder

    async def _start_streaming(self, task_recorder: OrchestraTaskRecorder, use_memory: bool = True):
        # NOTE The use_memory parameter is always reset by the environment variable
        # env_memory_setting = os.environ.get("memoryEnabled", "false").lower() == "true"
        # use_memory = env_memory_setting
        logger.info(f"[OrchestraReactSqlAgent] use_memory from env: {use_memory}")
        logger.info(f"[OrchestraReactSqlAgent] self._memory_toolkit: {self._memory_toolkit}")

        original_question = task_recorder.task
        if use_memory and self._memory_toolkit:
            logger.info(f"[Text2SQL] use_memory: {use_memory}")

            memory_contexts = await self._memory_toolkit.retrieve_all_context(
                query=original_question,
                include_skills=True,  # Text2SQL agent supports skills
            )

            working_context = memory_contexts["working_context"]
            episodic_context = memory_contexts["episodic_context"]
            semantic_context = memory_contexts["semantic_context"]
            skills_context = memory_contexts["skills_context"]
            memory_context = memory_contexts["memory_context"]

            if memory_context:
                logger.info(f"Retrieved memory context: {len(memory_context)} chars")
                enhanced_task = f"# 相关历史上下文\n{memory_context}\n\n---\n# 当前问题\n{original_question}"
                task_recorder.task = enhanced_task
                logger.info("Injected memory context into Text2SQL task")

            # Query episodic memory for few-shot SQL examples
            try:
                episodic_results = await self._memory_toolkit.search_memories(
                    query=original_question,
                    memory_type="episodic",
                    top_k=5,
                    include_outdated=True,
                )
                logger.info(f"[Text2SQL] Episodic memory search returned {len(episodic_results)} results")
                few_shot_examples = []
                for mem in episodic_results:
                    content = mem.memory.content if hasattr(mem, 'memory') else str(mem)
                    content_lower = content.lower()
                    logger.info(f"[Text2SQL] content_lower {content_lower}")
                    if "select" in content_lower and "howtofind" in content_lower:
                        few_shot_examples.append(content)
                task_recorder._few_shot_examples = few_shot_examples
                logger.info(f"[Text2SQL] Found {len(few_shot_examples)} few-shot examples from episodic memory")
            except Exception as e:
                logger.warning(f"[Text2SQL] Failed to retrieve few-shot examples: {e}")
                task_recorder._few_shot_examples = []

        with trace(workflow_name="orchestrareactsql_agent", trace_id=task_recorder.trace_id):
            try:
                for _ in range(self.max_turn_num):
                    await self.plan(task_recorder)
                    task = task_recorder.plan.todo[-1]
                    if task.completed:
                        break
                    worker_agent = self.worker_agents[task.agent_name]
                    await worker_agent.text2sql(task_recorder)
                await self.report(task_recorder)
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                task_recorder._is_complete = True
            except Exception as e:
                task_recorder._is_complete = True
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                raise e
        
        final_output = str(task_recorder.final_output or "")
        logger.debug(f"Final output: {final_output}")


        if use_memory and self._memory_toolkit and final_output:
            try:
                await self._memory_toolkit.store_working_memory(final_output, role="assistant")

                # Restore the original question by removing the injected context
                clean_question = original_question
                if "\n# 当前问题\n" in task_recorder.task:
                    clean_question = task_recorder.task.split("\n# 当前问题\n")[-1]

                await self._memory_toolkit.save_conversation_to_episodic(
                    question=clean_question,
                    answer=final_output,
                    importance_score=0.6,  # SQL output is usually important
                )
                logger.debug("Saved conversation to episodic memory")
            except Exception as e:
                logger.warning(f"Memory storage error: {e}")

    async def plan(self, task_recorder: OrchestraTaskRecorder) -> CreatePlanResult:
        plan = await self.planner_agent.create_plan(task_recorder)
        assert all(t.agent_name in self.worker_agents  for t in plan.todo), (
            f"agent_name in plan.todo must be in worker_agents, get {plan.todo}"
        )
        logger.info(f"plan: {plan}")
        if not task_recorder.plan:
            task_recorder.plan = plan
            task_recorder.trajectories.append(plan.trajectory)
        else:
            for todo in plan.todo:
                task_recorder.plan.todo.append(todo)
            task_recorder.plan.analysis = plan.analysis
            task_recorder.trajectories.append(plan.trajectory)

        return plan

    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        analysis_result = await self.reporter_agent.report(task_recorder)
        task_recorder.add_reporter_result(analysis_result)
        task_recorder.set_final_output(analysis_result.output)
        return analysis_result

