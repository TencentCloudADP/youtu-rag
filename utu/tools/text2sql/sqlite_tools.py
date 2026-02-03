# sqlite_tools.py
import os
import sqlite3
import pandas as pd
from urllib.parse import urlparse
from agents import function_tool
from typing import List, Dict, Any, Union

# 运行时SQLite连接配置（由schema link动态设置）
_dynamic_sqlite_config = {
    "file_path": None
}

def set_sqlite_config(file_path: str):
    """设置运行时SQLite连接配置"""
    global _dynamic_sqlite_config
    _dynamic_sqlite_config = {
        "file_path": file_path
    }

def parse_sqlite_connection_string(conn_str: str) -> dict:
    """
    解析SQLite连接字符串
    格式: sqlite:////path/to/database.db
    """
    try:
        # 移除 sqlite:///前缀
        if conn_str.startswith("sqlite:///"):
            file_path = conn_str.replace("sqlite:///", "")
            # 处理可能的表名后缀 (:table_name)
            file_path = file_path.split(':')[0]
            return {"file_path": file_path}
        return {}
    except Exception as e:
        print(f"解析连接字符串失败: {e}")
        return {}

# ========== 基础连接 ==========
def create_database(database_name: str) -> bool:
    """
    SQLite 没有 CREATE DATABASE，只要 connect 一个不存在的文件就会新建。
    这里统一在函数里 touch 文件，表示“数据库已准备好”。
    """
    try:
        # 如果文件已存在，sqlite3.connect 不会覆盖，符合“IF NOT EXISTS”语义
        conn = sqlite3.connect(f"{database_name}")
        conn.close()
        print(f"数据库 {database_name} 准备成功！")
        return True
    except Exception as e:
        print(f"创建数据库失败: {e}")
        return False


def create_connection(database: str = None) -> sqlite3.Connection | None:
    """建立 SQLite 连接（返回 Connection 对象）
    使用动态配置或传入的database参数
    """
    try:
        # 使用运行时配置
        config = _dynamic_sqlite_config.copy()
        db_path = database if database else config.get("file_path")

        # 验证配置完整性
        if not db_path:
            raise ValueError("SQLite连接配置不完整，请确保已选择知识库或传入database参数")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 支持列名访问
        conn.execute("PRAGMA foreign_keys = ON")
        # print("SQLite 数据库连接成功")
        return conn
    except Exception as e:
        print(f"连接 SQLite 数据库失败: {e}")
        return None


# def drop_database(database_name: str) -> bool:
#     """删除 SQLite 数据库文件"""
#     try:
#         db_file = f"{database_name}"
#         if os.path.exists(db_file):
#             os.remove(db_file)
#         print(f"数据库 {database_name} 删除成功")
#         return True
#     except Exception as e:
#         print(f"删除数据库失败: {e}")
#         return False


# ========== 表级操作 ==========
def check_table_exist(conn: sqlite3.Connection, table_name: str) -> bool:
    """判断表是否存在"""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cur.fetchone() is not None


# def drop_table(conn: sqlite3.Connection, table_name: str = "users") -> str:
#     """删除表"""
#     try:
#         conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
#         conn.commit()
#         msg = f"表 {table_name} 删除成功"
#         print(msg)
#         return msg
#     except Exception as e:
#         msg = f"删除表失败: {e}"
#         print(msg)
#         return msg


# ========== 数据查询 ==========
def fetch_data(conn: sqlite3.Connection, table_name: str, column: str) -> List[Any]:
    """
    查询某列的取值及其出现次数，按次数降序取前 100 个
    返回: 字段值列表（与原 MySQL 版保持一致）
    """
    try:
        sql = f"""
            SELECT "{column}" AS val, COUNT(*) AS cnt
            FROM "{table_name}"
            GROUP BY "{column}"
            ORDER BY cnt DESC
            LIMIT 100
        """
        rows = conn.execute(sql).fetchall()
        return [row["val"] for row in rows]
    except Exception as e:
        print(f"查询数据失败: {e}")
        return []


def ex_sql(conn: sqlite3.Connection, sql: str) -> str:
    """
    执行任意查询语句，返回 markdown 表格
    """
    try:
        cur = conn.execute(sql)
        data = cur.fetchall()
        # cols = [desc[0] for desc in cur.description]
        # df = pd.DataFrame(data, columns=cols)
        # df.to_markdown(index=False)

        #得到markdown结果
        column_names = [desc[0] for desc in cur.description]

        # 构建Markdown表格
        markdown_lines = []

        # 添加表头
        header = "| " + " | ".join(column_names) + " |"
        markdown_lines.append(header)

        # 添加分隔线
        separator = "| " + " | ".join(["---"] * len(column_names)) + " |"
        markdown_lines.append(separator)

        # 添加数据行
        for record in data:
            # 处理每个字段，确保转换为字符串并处理None值
            row = "| " + " | ".join([str(field) if field is not None else "" for field in record]) + " |"
            markdown_lines.append(row)

        # 合并所有行
        markdown_table = "\n".join(markdown_lines)


        return markdown_table
    except Exception as e:
        print(f"查询数据失败: {e}")
        return str(e)


# ========== 元信息 ==========
def get_tables_with_details(conn: sqlite3.Connection) -> Dict[str, Dict[str, str]]:
    """
    获取所有表的建表语句（DDL）
    返回: { table_name: { "create_sql": ddl} }
    SQLite 没有表级注释，comment 统一为空字符串
    """
    try:
        tables = {}
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cur.fetchall():
            tbl = row["name"]
            ddl = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone()[0]
            tables[tbl] = {"create_sql": ddl}
        return tables
    except Exception as e:
        print(f"获取表详情失败: {e}")
        return {}
    
def get_creat_table_details(conn: Union[sqlite3.Connection, str], table_name) -> Dict[str, Dict[str, str]]:
    """
    获取所有表的建表语句（DDL）
    返回: { table_name: { "create_sql": ddl} }
    SQLite 没有表级注释，comment 统一为空字符串
    """
    try:
        if isinstance(conn, str):
            conn = create_connection(conn)
        ddl = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()[0]
        return ddl
    except Exception as e:
        print(f"获取表详情失败: {e}")
        return {}
    
def get_sql_table_details(conn: sqlite3.Connection, table_name: str) -> Dict[str, Dict[str, str]]:
    """
    获取table_name表的建表语句（DDL）
    返回: { table_name: { "create_sql": ddl, "comment": "" } }
    SQLite 没有表级注释，comment 统一为空字符串
    """
    try:
        tables = {}
        ddl = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()[0]
        tables[table_name] = {"create_sql": ddl, "comment": ""}
        return tables
    except Exception as e:
        print(f"获取表详情失败: {e}")
        return {}


def from_db_get_column(conn: Union[sqlite3.Connection, str], table_name: str) -> List[Dict[str, str]]:
    """
    获取指定表的所有列名及其数据类型
    返回: [ {"COLUMN_NAME": col, "DATA_TYPE": type}, ... ]
    参数:
        conn: sqlite3.Connection对象或数据库文件路径字符串
        table_name: 表名
    """
    try:
        # PRAGMA table_info(table_name)
        if isinstance(conn, str):
            conn = create_connection(conn)
        rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        return [{"COLUMN_NAME": r["name"], "DATA_TYPE": r["type"]} for r in rows]
    except Exception as e:
        print(f"获取列信息失败: {e}")
        return []


def get_column_top_value(conn: sqlite3.Connection, table_name: str, column: str, top_k: int = 100) -> List[str]:
    """查询某列出现频率最高的 top_k 个值（字符串化后返回）"""
    try:
        sql = f"""
            SELECT "{column}" AS val, COUNT(*) AS cnt
            FROM "{table_name}"
            GROUP BY "{column}"
            ORDER BY cnt DESC
            LIMIT ?
        """
        rows = conn.execute(sql, (top_k,)).fetchall()
        return [str(row["val"]) for row in rows]
    except Exception as e:
        print(f"查询数据失败: {e}")
        return []


def get_column_value(conn: sqlite3.Connection, table_name: str, column: str, top_k: int = 1) -> List[Any]:
    """简单取指定列的前 top_k 条值"""
    try:
        sql = f'SELECT "{column}" FROM "{table_name}" LIMIT ?'
        rows = conn.execute(sql, (top_k,)).fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"查询数据失败: {e}")
        return []
    
@function_tool
def sqlite_excuted_sql_tool(sql):
    """
    sql执行工具，执行sql，并以md形式返回
    使用运行时配置的数据库连接（由schema link动态设置）
    """
    # Debug: Print current configuration
    print(f"[DEBUG] SQLite config - file_path: {_dynamic_sqlite_config.get('file_path')}")

    connection = create_connection()
    if not connection:
        return "错误：无法连接到数据库，请确保已正确配置数据库连接"

    # Debug: Print actual connection database
    cursor = connection.execute("PRAGMA database_list")
    db_list = cursor.fetchall()
    print(f"[DEBUG] Connected to database: {db_list}")

    ex_result = ex_sql(connection, sql)
    return ex_result


# ========== 自测 ==========
if __name__ == "__main__":
    db = "/Users/_jie-wang_/Desktop/agent/data/text2sql/人工标注/标准表格问题/sqlite/wj_agenticrag_text2sql_human_excel_stard_1202.db"
    # create_database(db)
    conn = create_connection(db)

    # 得到所有的表
    # print(get_tables_with_details(conn))
    print(get_sql_table_details(conn, "按样电站"))

    # print("表存在?", check_table_exist(conn, "洗衣机品牌"))
    # print("列信息:", from_db_get_column(conn, "洗衣机品牌"))
    # print("top3 值:", get_column_top_value(conn, "洗衣机品牌", "名称", 5))
    # print(fetch_data(conn, "洗衣机品牌", "名称"))

    conn.close()