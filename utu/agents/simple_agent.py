import os 
import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from utu.tools.memory_toolkit import VectorMemoryToolkit

from agents import (
    Agent,
    AgentOutputSchemaBase,
    Model,
    ModelSettings,
    RunConfig,
    RunHooks,
    Runner,
    StopAtTools,
    TContext,
    Tool,
    TResponseInputItem,
    trace,
)
from agents.mcp import MCPServer

from ..config import AgentConfig, ConfigLoader, ToolkitConfig
from ..context import BaseContextManager, build_context_manager
from ..db import DBService, TrajectoryModel
from ..env import BaseEnv, get_env
from ..hooks import get_run_hooks
from ..tools import TOOLKIT_MAP, AsyncBaseToolkit
from ..tools.utils import get_mcp_server
from ..utils import AgentsUtils, get_logger, load_class_from_file
from .common import QueueCompleteSentinel, TaskRecorder

logger = get_logger(__name__)


class SimpleAgent:
    """A simple agent with env, tools, mcps, and context manager, wrapped on openai-agents."""

    def __init__(
        self,
        *,
        config: AgentConfig | str | None = None,  # use config to pass agent configs
        name: str | None = None,
        instructions: str | Callable | None = None,
        model: str | Model | None = None,
        model_settings: ModelSettings | None = None,
        tools: list[Tool] = None,  # config tools
        toolkits: list[str] | None = None,  # load tools from toolkit configs
        output_type: type[Any] | AgentOutputSchemaBase | None = None,
        tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = "run_llm_again",
    ):
        assert not (tools and toolkits), "You can't pass both tools and toolkits."
        self.config = self._get_config(config)
        if name:
            self.config.agent.name = name
        if instructions:
            self.config.agent.instructions = instructions
        self.model = self._get_model(self.config, model)
        self.model_settings = self._get_model_settings(self.config, model_settings)
        self.tools: list[Tool] = tools or []
        self.toolkits: list[str] = toolkits or []
        self.output_type: type[Any] | AgentOutputSchemaBase | None = output_type
        self.tool_use_behavior: Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools = tool_use_behavior
        self.context_manager: BaseContextManager = None
        self.env: BaseEnv = None
        self.current_agent: Agent[TContext] = None  # move to task recorder?
        self.input_items: list[TResponseInputItem] = []
        self.run_hooks: RunHooks = get_run_hooks(self.config)

        self._mcp_servers: list[MCPServer] = []
        self._toolkits: dict[str, AsyncBaseToolkit] = {}
        self._mcps_exit_stack = AsyncExitStack()
        self._initialized = False

        # Add memory toolkit support
        self._memory_toolkit: "VectorMemoryToolkit | None" = None

    def _get_config(self, config: AgentConfig | str | None) -> AgentConfig:
        if isinstance(config, AgentConfig):
            return config
        return ConfigLoader.load_agent_config(config or "simple/base")

    def _get_model(self, config: AgentConfig, model: str | Model | None = None) -> Model:
        if isinstance(model, Model):
            return model
        model_provider_config = config.model.model_provider.model_dump()
        if isinstance(model, str):
            model_provider_config["model"] = model
        return AgentsUtils.get_agents_model(**model_provider_config)

    def _get_model_settings(self, config: AgentConfig, model_settings: ModelSettings | None = None) -> ModelSettings:
        if isinstance(model_settings, ModelSettings):
            return model_settings
        return config.model.model_settings

    # ==================== Memory Support ====================

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

    async def store_to_memory(
        self,
        question: str,
        answer: str,
        importance: float = 0.5,
        extra_metadata: dict | None = None,
    ) -> str | None:
        """Store a conversation to episodic memory if memory toolkit is available.

        Args:
            question: User question.
            answer: Agent answer.
            importance: Importance score (0.0-1.0).
            extra_metadata: Additional metadata to store.

        Returns:
            Memory ID if stored, None otherwise.
        """
        if self._memory_toolkit:
            return await self._memory_toolkit.save_conversation_to_episodic(
                question=question,
                answer=answer,
                importance_score=importance,
                extra_metadata=extra_metadata,
            )
        return None

    async def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from memory.

        Args:
            query: Query string for semantic search.
            top_k: Number of results to retrieve.

        Returns:
            Formatted context string, empty if no memory toolkit.
        """
        if self._memory_toolkit:
            context = await self._memory_toolkit.retrieve_relevant_context(query, top_k=top_k)
            return self._memory_toolkit.format_context_for_prompt(context)
        return ""

    async def get_memory_enhanced_prompt(self, question: str, top_k: int = 5) -> str:
        """Get a memory-enhanced version of the question.

        Args:
            question: Original user question.
            top_k: Number of context items to retrieve.

        Returns:
            Enhanced prompt with memory context, or original question if no context.
        """
        context = await self.retrieve_context(question, top_k=top_k)
        if context:
            return f"请参考以下历史上下文信息:\n{context}\n\n当前问题: {question}"
        return question

    # ==================== Memory Support End ====================

    async def __aenter__(self) -> "SimpleAgent":
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def build(self, trace_id: str = None):
        """Build the agent"""
        if self._initialized:
            logger.info("Agent already initialized! Skipping build.")
            return
        self.env = await get_env(self.config, trace_id or AgentsUtils.gen_trace_id())  # Pass trace_id
        await self.env.build()
        self.current_agent = Agent(
            name=self.config.agent.name,
            instructions=self.config.agent.instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=await self.get_tools(),
            output_type=self.output_type,
            tool_use_behavior=self.tool_use_behavior,
            mcp_servers=self._mcp_servers,
        )
        self.context_manager = build_context_manager(self.config)
        self._initialized = True

    async def cleanup(self):
        """Cleanup"""
        logger.info("Cleaning up MCP servers...")
        await self._mcps_exit_stack.aclose()
        self._mcp_servers = []
        logger.info("Cleaning up tools...")
        self._toolkits = {}
        logger.info("Cleaning up env...")
        await self.env.cleanup()
        self._initialized = False

    async def get_tools(self) -> list[Tool]:
        if self.tools:
            return self.tools

        if self.toolkits:
            await self._load_toolkits_config()
            return self.tools

        tools_list: list[Tool] = []
        tools_list += await self.env.get_tools()  # add env tools
        # TODO: handle duplicate tool names
        for _, toolkit_config in self.config.toolkits.items():
            toolkit = await self._load_toolkit(toolkit_config)
            if toolkit_config.mode in ["customized", "builtin"]:
                tools_list.extend(toolkit.get_tools_in_agents())
        tool_names = [tool.name for tool in tools_list]
        logger.info(f"Loaded {len(tool_names)} tools: {tool_names}")
        self.tools = tools_list
        return self.tools

    async def _load_toolkits_config(self):
        assert isinstance(self.toolkits, list) and all(isinstance(tool, str) for tool in self.toolkits)
        parsed_tools = []
        for tool_name in self.toolkits:
            config = ConfigLoader.load_toolkit_config(tool_name)
            toolkit = await self._load_toolkit(config)
            if config.mode in ["customized", "builtin"]:
                parsed_tools.extend(toolkit.get_tools_in_agents())
        self.tools = parsed_tools

    async def _load_toolkit(self, toolkit_config: ToolkitConfig) -> AsyncBaseToolkit | MCPServer:
        if toolkit_config.mode == "builtin":
            return await self._load_builtin_toolkit(toolkit_config)
        elif toolkit_config.mode == "customized":
            return await self._load_customized_toolkit(toolkit_config)
        elif toolkit_config.mode == "mcp":
            return await self._load_mcp_server(toolkit_config)
        else:
            raise ValueError(f"Unknown toolkit mode: {toolkit_config.mode}")

    async def _load_builtin_toolkit(self, toolkit_config: ToolkitConfig) -> AsyncBaseToolkit:
        logger.info(f"Loading builtin toolkit `{toolkit_config.name}` with config {toolkit_config}")
        toolkit = TOOLKIT_MAP[toolkit_config.name](toolkit_config)
        self._toolkits[toolkit_config.name] = toolkit
        return toolkit

    async def _load_customized_toolkit(self, toolkit_config: ToolkitConfig) -> AsyncBaseToolkit:
        logger.info(f"Loading customized toolkit `{toolkit_config.name}` with config {toolkit_config}")
        assert toolkit_config.customized_filepath is not None and toolkit_config.customized_classname is not None
        toolkit_class = load_class_from_file(toolkit_config.customized_filepath, toolkit_config.customized_classname)
        toolkit = toolkit_class(toolkit_config)
        self._toolkits[toolkit_config.name] = toolkit
        return toolkit

    async def _load_mcp_server(self, toolkit_config: ToolkitConfig) -> MCPServer:
        logger.info(f"Loading MCP server `{toolkit_config.name}` with params {toolkit_config.config}")
        mcp_server = get_mcp_server(toolkit_config)
        server = await self._mcps_exit_stack.enter_async_context(mcp_server)
        self._mcp_servers.append(server)
        return server

    def _get_run_config(self) -> RunConfig:
        run_config = RunConfig(
            model=self.current_agent.model,
            model_settings=self.config.model.model_settings,
            workflow_name=self.config.agent.name,
        )
        return run_config

    def _get_context(self) -> dict:
        return {
            "context_manager": self.context_manager,
            "env": self.env,
        }

    def _prepare_run_kwargs(self, input: str | list[TResponseInputItem]) -> dict:
        return {
            "starting_agent": self.current_agent,
            "input": input,
            "context": self._get_context(),
            "max_turns": self.config.max_turns,
            "hooks": self.run_hooks,
            "run_config": self._get_run_config(),
        }

    # wrap `Runner` apis in @openai-agents
    async def run(
        self, input: str | list[TResponseInputItem], trace_id: str = None, save: bool = False
    ) -> TaskRecorder:
        """Entrypoint for running the agent

        Args:
            trace_id: str to identify the run
            save: whether to update massage history (use `input_items`)
        """
        recorder = self.run_streamed(input, trace_id)
        async for _ in recorder.stream_events():
            pass
        return recorder

    def run_streamed(
        self, input: str | list[TResponseInputItem], trace_id: str = None, save: bool = False, log_to_db: bool = True, use_memory: bool = True
    ) -> TaskRecorder:
        """Entrypoint for running the agent streamly

        Args:
            trace_id: str to identify the run
        """
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        if isinstance(input, list):
            assert isinstance(input[-1], dict) and "content" in input[-1], "invalid input format!"
            task = input[-1]["content"]
        else:
            assert isinstance(input, str), "input should be str or list of TResponseInputItem!"
            task = input
        recorder = TaskRecorder(task=task, input=input, trace_id=trace_id)
        recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder, save, log_to_db, use_memory))
        return recorder

    def _check_memory_config(self, use_memory: bool) -> bool:
        """Check environment variable and override use_memory parameter.

        Args:
            use_memory: Original use_memory parameter value.

        Returns:
            Final use_memory value based on environment configuration.
        """
        env_memory_setting = os.environ.get("memoryEnabled", "false").lower() == "true"
        logger.info(f"[SimpleAgent] Updated use_memory from env: {env_memory_setting}")
        return env_memory_setting

    async def _retrieve_memory_context(self, original_question: str, use_memory: bool) -> tuple[str, list]:
        """Retrieve all memory contexts (working, episodic, semantic).

        Args:
            original_question: The user's original question.
            use_memory: Whether memory is enabled.

        Returns:
            Tuple of (memory_context: str, tool_calls_collected: list)
        """
        memory_context = ""
        tool_calls_collected = []

        logger.info(f"[SimpleAgent] self._memory_toolkit: {self._memory_toolkit}")

        # Only retrieve memory if enabled and toolkit exists
        if use_memory and self._memory_toolkit:
            try:
                # Lazy import to avoid circular dependency
                from utu.tools.memory_toolkit import ToolCall

                # 使用通用方法检索所有类型的记忆
                memory_contexts = await self._memory_toolkit.retrieve_all_context(
                    query=original_question,
                    include_skills=False,  # simple_agent 中暂时不包含 skills
                )
                
                working_context = memory_contexts["working_context"]
                episodic_context = memory_contexts["episodic_context"]
                semantic_context = memory_contexts["semantic_context"]
                skills_context = memory_contexts["skills_context"]
                memory_context = memory_contexts["memory_context"]

                # 旧代码已注释：Skills 检索逻辑
                # 现在通过 retrieve_all_context 处理
                # # 5. 检索相关 Skills（程序性记忆）
                # # try:
                # #     skill_results = await self._memory_toolkit.search_skills(
                # #         query=original_question,
                # #         top_k=3,
                # #         min_success_rate=0.3,
                # #     )
                # #     if skill_results:
                # #         skills_context = self._memory_toolkit.format_skills_for_prompt(skill_results)
                # #         logger.info(f"Retrieved {len(skill_results)} relevant skills")
                # # except Exception as e:
                # #     logger.debug(f"Skill search failed: {e}")

                # 旧代码已删除：合并上下文逻辑
                # 现在 memory_context 已经由 retrieve_all_context 返回

                if memory_context:
                    logger.info(f"Retrieved memory context: {len(memory_context)} chars")

            except Exception as e:
                logger.warning(f"Memory retrieval error: {e}")

        return memory_context, tool_calls_collected

    def _enhance_input_with_memory(self, input: str | list[TResponseInputItem], memory_context: str) -> list[TResponseInputItem]:
        """Inject memory context into the input prompt.

        Args:
            input: Original input (string or list of message dicts).
            memory_context: Memory context to inject.

        Returns:
            Enhanced input as list of message dicts.
        """
        if isinstance(input, str):
            if memory_context:
                enhanced_input = f"# 相关历史上下文\n{memory_context}\n\n---\n# 当前问题\n{input}"
            else:
                enhanced_input = input
            return self.input_items + [{"content": enhanced_input, "role": "user"}]
        else:
            if memory_context and isinstance(input[-1], dict) and "content" in input[-1]:
                original_content = input[-1]["content"]
                input[-1]["content"] = f"# 相关历史上下文\n{memory_context}\n\n---\n# 当前问题\n{original_content}"
            return input

    def _collect_tool_call_from_event(self, event, tool_calls_collected: list, use_memory: bool) -> None:
        """Extract tool calls from stream events.

        Args:
            event: Stream event to process.
            tool_calls_collected: List to append tool calls to.
            use_memory: Whether memory is enabled.
        """
        if use_memory and self._memory_toolkit:
            if hasattr(event, "type") and event.type == "run_item_stream_event":
                item = getattr(event, "item", None)
                if item and hasattr(item, "type") and item.type == "tool_call_item":
                    from utu.tools.memory_toolkit import ToolCall
                    tool_name = getattr(item, "name", None) or getattr(item, "tool", "unknown")
                    tool_args = getattr(item, "arguments", {}) or getattr(item, "args", {})
                    tool_result = getattr(item, "output", None)
                    tool_calls_collected.append(
                        ToolCall(tool=tool_name, args=tool_args, result=tool_result, success=True)
                    )

    async def _store_execution_to_memory(self, original_question: str, final_output: str, use_memory: bool) -> None:
        """Store conversation results to memory.

        Args:
            original_question: The user's original question.
            final_output: The agent's final output.
            use_memory: Whether memory is enabled.
        """
        if use_memory and self._memory_toolkit and final_output:
            try:
                final_output_str = str(final_output)
                # Store to working memory
                await self._memory_toolkit.store_working_memory(final_output_str, role="assistant")
                # Store to episodic memory (persistent)
                await self._memory_toolkit.save_conversation_to_episodic(
                    question=original_question,
                    answer=final_output_str,
                    importance_score=0.5,
                )
                logger.debug("Saved conversation to episodic memory")
            except Exception as e:
                logger.warning(f"Memory storage error: {e}")

    def _execute_agent(self, run_kwargs: dict, trace_id: str):
        """Execute agent with or without trace context.

        Args:
            run_kwargs: Keyword arguments for Runner.run_streamed.
            trace_id: Trace ID for the execution.

        Returns:
            run_streamed_result from Runner.
        """
        if AgentsUtils.get_current_trace():
            return Runner.run_streamed(**run_kwargs)
        else:
            with trace(workflow_name="simple_agent", trace_id=trace_id):
                return Runner.run_streamed(**run_kwargs)

    async def _start_streaming(self, recorder: TaskRecorder, save: bool = False, log_to_db: bool = True, use_memory: bool = True):
        """Main streaming execution method with memory support.

        Args:
            recorder: TaskRecorder to manage event streaming.
            save: Whether to update message history.
            log_to_db: Whether to log trajectory to database.
            use_memory: Whether to use memory (overridden by environment variable).
        """
        # Override use_memory from environment configuration
        use_memory = self._check_memory_config(use_memory)

        if not self._initialized:
            await self.build(recorder.trace_id)

        try:
            input = recorder.input
            original_question = recorder.task

            # Retrieve memory context
            memory_context, tool_calls_collected = await self._retrieve_memory_context(
                original_question, use_memory
            )

            # Enhance input with memory context
            input = self._enhance_input_with_memory(input, memory_context)

            # Execute agent
            run_kwargs = self._prepare_run_kwargs(input)
            run_streamed_result = self._execute_agent(run_kwargs, recorder.trace_id)

            # Stream events and collect tool calls
            async for event in run_streamed_result.stream_events():
                recorder._event_queue.put_nowait(event)
                self._collect_tool_call_from_event(event, tool_calls_collected, use_memory)

            # Save final output and trajectory
            recorder.add_run_result(run_streamed_result)
            logger.info(f"Run Result Final Output: {run_streamed_result.final_output}")

            # Store execution to memory
            await self._store_execution_to_memory(
                original_question,
                run_streamed_result.final_output,
                use_memory
            )

            # Save state if requested
            if save:
                self.input_items = run_streamed_result.to_input_list()
                self.current_agent = run_streamed_result.last_agent

            # Log to database
            if log_to_db:
                DBService.add(TrajectoryModel.from_task_recorder(recorder))

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True
            raise e
        finally:
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True

    # util apis
    async def chat(self, input: str) -> TaskRecorder:
        # TODO: set "session-level" tracing for multi-turn chat
        recorder = await self.run(input, save=True)
        run_result = recorder.get_run_result()
        AgentsUtils.print_new_items(run_result.new_items)
        return run_result

    async def chat_streamed(self, input: str) -> TaskRecorder:
        recorder = self.run_streamed(input, save=True)
        await AgentsUtils.print_stream_events(recorder.stream_events())
        return recorder

    def set_instructions(self, instructions: str):
        logger.warning("WARNING: reset instructions is dangerous!")
        self.current_agent.instructions = instructions

    def clear_input_items(self):
        # reset chat history
        self.input_items = []