"""
统一的Schema Link和Value Link模块
支持多数据库场景（MySQL和SQLite混合）
"""
import os
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from utu.utils import PrintUtils, FileUtils
from utu.rag.knowledge_retrieval.chroma_retrical_text2sql import CourseSearcher
from utu.agents import SimpleAgent

# 导入MySQL和SQLite的工具函数
from .mysql_tools import (
    set_mysql_config,
    create_connection as mysql_create_connection,
    from_db_get_column as mysql_get_column,
    get_column_value as mysql_get_column_value,
    get_creat_table_details as mysql_get_table_ddl,
    parse_mysql_connection_string
)
from .sqlite_tools import (
    set_sqlite_config,
    create_connection as sqlite_create_connection,
    from_db_get_column as sqlite_get_column,
    get_column_value as sqlite_get_column_value,
    get_creat_table_details as sqlite_get_table_ddl,
    parse_sqlite_connection_string
)


def extract_db_id_from_source(source: str, db_type: str = None) -> str:
    """
    从source字段中提取db_id

    Args:
        source: source字段值
        db_type: 数据库类型 (mysql/sqlite/excel)

    Returns:
        db_id: 数据库标识

    示例:
        - MySQL: mysql://root@host:port/database:table_name -> mysql://root@host:port/database
        - SQLite: sqlite:////path/to/database.db:table_name -> sqlite:////path/to/database.db
        - Excel: filename.xlsx -> sqlite:///rag_data/relational_database/rag_demo.sqlite
    """
    # Excel表存储在rag_demo.sqlite中
    if db_type == "excel":
        rag_demo_path = os.getenv("RELATIONAL_DB_PATH", "rag_data/relational_database/rag_demo.sqlite")
        # 转换为绝对路径
        if not rag_demo_path.startswith('/'):
            rag_demo_path = os.path.abspath(rag_demo_path)
        return f"sqlite:///{rag_demo_path}"

    # 去掉最后的表名部分（:table_name）
    if ':' in source:
        parts = source.rsplit(':', 1)
        return parts[0]
    return source


def format_ddl_method(sql):
    """格式化DDL语句"""
    formatted = sql.replace(', ', ',\n    ')
    formatted = formatted.replace('("', '(\n    "')
    formatted = formatted.replace(')', '\n)')
    return formatted


def get_valuelink_tablestr(table_sql, col_dict):
    """
    在DDL中添加列值示例

    Args:
        table_sql: 建表语句
        col_dict: {列名: [示例值列表]}

    Returns:
        带值示例的DDL语句
    """
    if "\n" not in table_sql:
        table_sql = format_ddl_method(table_sql)

    lines = table_sql.split('\n')
    new_lines = [lines[0]]

    for line in lines[1:-1]:
        column_name = line.strip().split()[0].strip().strip('\"').strip('"').strip("'").strip("`")
        example = col_dict.get(column_name)

        if example is not None:
            if not line.endswith(","):
                line += f", -- example {example}"
            else:
                line += f" -- example {example}"
        new_lines.append(line)

    new_lines.append(lines[-1])
    return '\n'.join(new_lines)


async def get_retrieval_tables_chroma(question: str, searcher: CourseSearcher, k: int = 5, filter_conditions: List[Dict] = None) -> List[Dict]:
    """召回top-k相关文档"""
    top_tables = await searcher.search(
        query=question,
        top_k=k,
        filter_conditions=filter_conditions,
    )
    return top_tables


async def select_databases_with_llm(question: str, db_groups: Dict, event_queue=None) -> List[str]:
    """
    Use LLM to select which databases are relevant to answer the question

    Args:
        question: User's question
        db_groups: Dictionary of database groups with metadata
        event_queue: Event queue for streaming intermediate results (optional)

    Returns:
        List of selected db_ids
    """
    PrintUtils.print_info(f"   >> Calling LLM to select relevant databases...")


    # Build database information for LLM
    databases_info_lines = []
    db_id_list = list(db_groups.keys())

    for idx, (db_id, group_info) in enumerate(db_groups.items(), 1):
        db_type = group_info["db_type"]
        table_names = group_info["table_names"]
        tables_info = group_info["tables_info"]

        db_display = db_id if len(db_id) <= 100 else f"{db_id[:97]}..."

        databases_info_lines.append(f"### Database {idx} (ID: {db_id})")
        databases_info_lines.append(f"- **Type**: {db_type}")
        databases_info_lines.append(f"- **Source**: `{db_display}`")
        databases_info_lines.append(f"- **Tables** ({len(table_names)}):\n")

        # Format table schema info for each table (with columns)
        for table_info in tables_info:
            table_name = table_info["metadata"].get("table_name", "unknown")
            content = table_info.get("content", "")
            databases_info_lines.append(f"  **{table_name}**: " + content)

        databases_info_lines.append("")  # Empty line for separation

    databases_info = "\n".join(databases_info_lines)

    # Load prompt template
    prompts = FileUtils.load_prompts("ragref/text2sql/selectdb.yaml")
    select_prompt = prompts["select_database_prompt"].replace("{question}", question).replace("{databases_info}", databases_info)

    # Stream the prompt to frontend for transparency
    if event_queue:
        # Truncate prompt for display if too long
        max_prompt_display_length = 5000
        prompt_display = select_prompt[:max_prompt_display_length]
        if len(select_prompt) > max_prompt_display_length:
            prompt_display += f"\n\n... (提示词总长度: {len(select_prompt)} 字符，已截断显示前 {max_prompt_display_length} 字符)"

        prompt_msg = f"## 📝 LLM Prompt\n\n```\n{prompt_display}\n```"
        PrintUtils.print_and_stream_tool(prompt_msg, event_queue=event_queue, tool_name="llm_prompt")

    # Create LLM agent for selection
    agent = SimpleAgent(
        name="database_selector",
        instructions="You are a database selection expert. Analyze the question and available databases, then select which database(s) are most relevant."
    )

    try:
        # Run LLM selection
        result = await agent.run(select_prompt)
        response_text = str(result.final_output)

        PrintUtils.print_info(f"   >> LLM selection response:\n{response_text}")


        # Extract JSON from response
        # Look for JSON block
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_match = re.search(r'\{[^{}]*"selected_db_ids"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                PrintUtils.print_info(f"   警告: 无法从LLM响应中提取JSON，使用所有数据库")
                return db_id_list

        # Parse JSON
        selection = json.loads(json_str)
        selected_db_ids = selection.get("selected_db_ids", [])
        reasoning = selection.get("reasoning", "No reasoning provided")

        PrintUtils.print_info(f"   >> LLM reasoning: {reasoning}")
        PrintUtils.print_info(f"   >> Selected {len(selected_db_ids)} database(s)")

        # Stream LLM reasoning and selection result to frontend
        if event_queue:
            # Build selected databases summary
            selected_summary_lines = [
                f"## ✅ Database Selection Result\n",
                f"**推理过程**: {reasoning}\n",
                f"\n**选择结果**: 从 {len(db_groups)} 个数据库中选择了 **{len(selected_db_ids)}** 个\n"
            ]

            # List selected databases
            for idx, db_id in enumerate(selected_db_ids, 1):
                if db_id in db_groups:
                    db_type = db_groups[db_id]["db_type"]
                    table_count = len(db_groups[db_id]["table_names"])
                    selected_summary_lines.append(f"**{idx}.** {db_type} - {table_count} 个表")
                    selected_summary_lines.append(f"   `{db_id}`\n")

            selection_result_msg = "\n".join(selected_summary_lines)
            PrintUtils.print_and_stream_tool(selection_result_msg, event_queue=event_queue, tool_name="llm_selection_result")

        # Validate selection
        if not selected_db_ids:
            PrintUtils.print_info(f"   警告: LLM未选择任何数据库，使用第一个数据库")
            return [db_id_list[0]]

        # Filter out invalid db_ids
        valid_selected = [db_id for db_id in selected_db_ids if db_id in db_groups]

        if not valid_selected:
            PrintUtils.print_info(f"   警告: LLM选择的数据库ID无效，使用第一个数据库")
            return [db_id_list[0]]

        return valid_selected

    except Exception as e:
        PrintUtils.print_info(f"   错误: LLM选择失败 - {e}，使用所有数据库")
        return db_id_list


async def value_link_for_table(db_type: str, db_id: str, table_name: str, question: str, searcher: CourseSearcher) -> Tuple[str, Dict[str, List]]:
    """
    对单个表进行value link

    Args:
        db_type: 数据库类型 (mysql/sqlite/excel)
        db_id: 数据库ID (connection_string)
        table_name: 表名
        question: 用户问题（用于检索列值）
        searcher: 向量检索器

    Returns:
        (带value link的DDL, {列名: [示例值]})
    """
    PrintUtils.print_info(f"   >> value_link {db_type}:{table_name}...")

    # Select appropriate tool functions based on database type
    # Excel tables are stored in SQLite, so treat as SQLite
    if db_type == "mysql":
        # MySQL functions expect database name string, not connection object
        # Extract database name from db_id (e.g., mysql://host:port/database)
        database_name = db_id.split('/')[-1]

        # MySQL functions signature: func(database, table_name, ...)
        # They create their own connections internally
        get_column_func = lambda db, tbl: mysql_get_column(db, tbl)
        get_value_func = lambda db, tbl, col, top_k=2: mysql_get_column_value(db, tbl, col, top_k)
        get_ddl_func = lambda db, tbl: mysql_get_table_ddl(db, tbl)

        db_param = database_name  # For MySQL, pass database name string
    else:  # sqlite or excel (excel uses sqlite backend)
        # SQLite functions expect connection object
        conn = sqlite_create_connection()

        # SQLite functions signature: func(conn, table_name, ...)
        get_column_func = lambda c, tbl: sqlite_get_column(c, tbl)
        get_value_func = lambda c, tbl, col, top_k=1: sqlite_get_column_value(c, tbl, col, top_k)
        get_ddl_func = lambda c, tbl: sqlite_get_table_ddl(c, tbl)

        db_param = conn  # For SQLite, pass connection object

    # 获取表的所有列信息
    table_columninfos = get_column_func(db_param, table_name)
    column_example_dict = {}

    for column_info in table_columninfos:
        column_name = column_info["COLUMN_NAME"]
        column_type = column_info["DATA_TYPE"]

        # 只对文本列进行value link（通过向量检索）
        if "char" in column_type.lower() or "text" in column_type.lower():
            try:
                # 从向量库召回相关列值
                filter_conditions = [
                    {"type": "column_value"},
                    {"table_name": table_name},
                    {"column_name": column_name}
                ]
                top_values_info = await get_retrieval_tables_chroma(question, searcher, 3, filter_conditions)
                top_values = [item["content"] for item in top_values_info]
                column_example_dict[column_name] = top_values
            except:
                # 如果检索失败，直接从数据库取值
                cells = get_value_func(db_param, table_name, column_name, top_k=3)
                column_example_dict[column_name] = cells
        else:
            # 非文本列，随机取一个值
            cells = get_value_func(db_param, table_name, column_name, top_k=1)
            column_example_dict[column_name] = cells

    # 获取建表语句并添加value link
    table_sql = get_ddl_func(db_param, table_name)
    final_table_sql = get_valuelink_tablestr(table_sql, column_example_dict)

    return final_table_sql, column_example_dict


async def unified_schemalink_with_valuelink(question: str, task_recorder=None) -> Dict[str, Any]:
    """
    统一的schema link + value link函数
    支持多数据库场景，按db_id分组处理

    Args:
        question: 用户问题
        task_recorder: 任务记录器，用于流式输出中间结果到前端（可选）

    Returns:
        {
            "db_groups": {
                "db_id1": {
                    "db_type": "mysql/sqlite",
                    "db_id": "connection_string",
                    "tables": ["table1_ddl", "table2_ddl", ...],
                    "table_names": ["table1", "table2", ...]
                },
                ...
            },
            "all_tables_str": "所有表的DDL字符串（用于向后兼容）"
        }
    """
    PrintUtils.print_info(">> unified schema link...")

    # 运行时读取环境变量
    kb_collection_name = os.environ.get('kb_collection_name')
    vector_save_path = os.environ.get('VECTOR_STORE_PATH')

    # 检查必需的环境变量
    if not kb_collection_name or not vector_save_path:
        PrintUtils.print_info(f"   错误: 缺少必需的环境变量")
        PrintUtils.print_info(f"   kb_collection_name: {kb_collection_name}")
        PrintUtils.print_info(f"   VECTOR_STORE_PATH: {vector_save_path}")
        PrintUtils.print_info(f"   返回空结果，将使用fallback逻辑")
        return {"db_groups": {}, "all_tables_str": ""}

    searcher = CourseSearcher(kb_collection_name, vector_save_path)

    # Step 1: 召回所有候选表（不限制db_type）
    filter_conditions = [{"type": "table_schema"}]
    top_tables_info = await get_retrieval_tables_chroma(question, searcher, k=10, filter_conditions=filter_conditions)

    PrintUtils.print_info(f"   召回 {len(top_tables_info)} 个候选表")

    # Stream recall result to frontend
    event_queue = task_recorder._event_queue if task_recorder else None

    # Step 2: 按db_id分组
    db_groups = defaultdict(lambda: {
        "db_type": None,
        "db_id": None,
        "tables_info": [],  # 原始召回信息
        "table_names": [],
        "config_set": False
    })

    for doc in top_tables_info:
        metadata = doc["metadata"]
        source = metadata.get("source", "")
        table_name = metadata.get("table_name")
        db_type = metadata.get("db_type")

        # Extract db_id based on db_type
        db_id = extract_db_id_from_source(source, db_type)

        # Group by db_id
        db_groups[db_id]["db_type"] = db_type
        db_groups[db_id]["db_id"] = db_id
        db_groups[db_id]["tables_info"].append(doc)
        db_groups[db_id]["table_names"].append(table_name)

    PrintUtils.print_info(f"   按db_id分组后：共 {len(db_groups)} 个数据库")

    # Stream grouping result to frontend
    if event_queue:
        # Build database summary
        db_summary_lines = [f"## 📂 Database Grouping\n", f"按数据库分组后：共 **{len(db_groups)}** 个数据库\n"]
        for idx, (db_id, group_info) in enumerate(db_groups.items(), 1):
            db_type = group_info["db_type"]
            table_count = len(group_info["table_names"])
            db_display = db_id if len(db_id) <= 60 else f"{db_id[:57]}..."
            db_summary_lines.append(f"**Database {idx}** ({db_type}): {table_count} 个表")
            db_summary_lines.append(f"  - Source: `{db_display}`\n")

        grouping_msg = "\n".join(db_summary_lines)
        PrintUtils.print_and_stream_tool(grouping_msg, event_queue=event_queue, tool_name="database_grouping")

    # Step 2.5: If multiple databases, use LLM to select relevant ones
    if len(db_groups) > 1:
        selected_db_ids = await select_databases_with_llm(question, db_groups, event_queue)
        PrintUtils.print_info(f"   LLM选择后：保留 {len(selected_db_ids)} 个数据库")

        # Filter db_groups to only include selected databases
        db_groups = {db_id: db_groups[db_id] for db_id in selected_db_ids if db_id in db_groups}
    else:
        PrintUtils.print_info(f"   只有1个数据库，跳过LLM选择")

        # Stream skip message to frontend
        if event_queue:
            skip_msg = f"## ℹ️  Single Database\n\n只有1个数据库，跳过LLM选择步骤"
            PrintUtils.print_and_stream_tool(skip_msg, event_queue=event_queue, tool_name="skip_llm_selection")

    # Step 3: 对每个db_id分别进行schema recall + value link
    result = {"db_groups": {}, "all_tables_str": ""}
    all_tables_ddl = []

    for db_idx, (db_id, group_info) in enumerate(db_groups.items(), 1):
        db_type = group_info["db_type"]
        table_names = group_info["table_names"]

        PrintUtils.print_info(f"\n   >> 处理数据库: {db_type}:{db_id}")
        PrintUtils.print_info(f"      包含表: {', '.join(table_names)}")

        # Stream database processing to frontend
        if event_queue:
            db_display = db_id if len(db_id) <= 80 else f"{db_id[:77]}..."
            processing_msg = f"## 🔄 Processing Database {db_idx}/{len(db_groups)}\n\n**类型**: {db_type}\n**数据库**: `{db_display}`\n**表数量**: {len(table_names)}\n**表列表**: {', '.join([f'`{t}`' for t in table_names])}"
            PrintUtils.print_and_stream_tool(processing_msg, event_queue=event_queue, tool_name="database_processing")

        # Step 3.1: Configure database connection based on type
        if not group_info["config_set"]:
            first_doc = group_info["tables_info"][0]
            metadata = first_doc["metadata"]
            source = metadata.get("source", "")
            first_table_name = metadata.get("table_name")

            try:
                # Excel tables are stored in rag_demo.sqlite - use direct file path
                if db_type == "excel":
                    rag_demo_path = os.getenv("RELATIONAL_DB_PATH", "rag_data/relational_database/rag_demo.sqlite")
                    # Convert to absolute path
                    if not rag_demo_path.startswith('/'):
                        rag_demo_path = os.path.abspath(rag_demo_path)

                    set_sqlite_config(file_path=rag_demo_path)
                    PrintUtils.print_info(f"      Excel tables SQLite config set: {rag_demo_path}")
                    group_info["config_set"] = True
                else:
                    # MySQL/SQLite - query from kb_source_configs
                    from utu.rag.api.database import get_db, KBSourceConfig
                    db_session = next(get_db())

                    # Find config by source_identifier
                    source_identifier = f"{db_id}:{first_table_name}"
                    PrintUtils.print_info(f"      Looking for config: {source_identifier}")

                    source_config = db_session.query(KBSourceConfig).filter(
                        KBSourceConfig.source_identifier == source_identifier
                    ).first()

                    if source_config and source_config.config:
                        config = source_config.config

                        if db_type == "mysql":
                            conn_info = parse_mysql_connection_string(source)
                            host = config.get("host") or conn_info.get("host")
                            user = config.get("username") or conn_info.get("user")
                            password = config.get("password") or ""
                            port = config.get("port") or conn_info.get("port", 3306)
                            database = config.get("database") or db_id.split('/')[-1]

                            if not host or not user:
                                raise ValueError(f"Incomplete MySQL config: host={host}, user={user}")

                            set_mysql_config(
                                host=host,
                                user=user,
                                password=password,
                                port=port,
                                database=database
                            )
                            PrintUtils.print_info(f"      MySQL config set: {host}:{port}/{database}")

                        else:  # sqlite
                            conn_info = parse_sqlite_connection_string(source)
                            file_path = config.get("file_path") or conn_info.get("file_path")

                            if not file_path:
                                raise ValueError(f"Incomplete SQLite config: file_path={file_path}")

                            set_sqlite_config(file_path=file_path)
                            PrintUtils.print_info(f"      SQLite config set: {file_path}")

                        group_info["config_set"] = True
                    else:
                        raise ValueError(f"Config not found for table {first_table_name}")

            except Exception as e:
                PrintUtils.print_info(f"      Error: Failed to configure database: {e}")
                continue

        # Step 3.2: 对该db下的所有表进行value link
        tables_ddl = []
        for table_idx, table_name in enumerate(table_names, 1):
            try:
                table_ddl, _ = await value_link_for_table(
                    db_type=db_type,
                    db_id=db_id,
                    table_name=table_name,
                    question=question,
                    searcher=searcher
                )
                tables_ddl.append(table_ddl)
                all_tables_ddl.append(table_ddl)
            except Exception as e:
                PrintUtils.print_info(f"      警告: 表 {table_name} value link失败: {e}")

        # 保存该数据库的结果
        result["db_groups"][db_id] = {
            "db_type": db_type,
            "db_id": db_id,
            "tables": tables_ddl,
            "table_names": table_names
        }

    # 生成所有表的DDL字符串（用于向后兼容）
    result["all_tables_str"] = "\n\n".join(all_tables_ddl)

    PrintUtils.print_info(f"\n   Schema Link完成，共处理 {len(all_tables_ddl)} 个表")

    # Clear embedding cache to free memory after processing
    searcher.clear_embedding_cache()

    return result
