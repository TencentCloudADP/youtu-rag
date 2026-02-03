import os
import pandas as pd
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from agents import function_tool
import mysql.connector
from mysql.connector import Error
load_dotenv()

# 运行时MySQL连接配置（由schema link动态设置）
_dynamic_mysql_config = {
    "host": None,
    "user": None,
    "password": None,
    "port": 3306,
    "database": None
}

def set_mysql_config(host: str, user: str, password: str, port: int, database: str):
    """设置运行时MySQL连接配置"""
    global _dynamic_mysql_config
    _dynamic_mysql_config = {
        "host": host,
        "user": user,
        "password": password,
        "port": port,
        "database": database
    }

def parse_mysql_connection_string(conn_str: str) -> dict:
    """
    解析MySQL连接字符串
    格式: mysql://user@host:port/database
    """
    try:
        # 使用urlparse解析
        parsed = urlparse(conn_str)
        return {
            "host": parsed.hostname,
            "user": parsed.username,
            "port": parsed.port or 3306,
            "database": parsed.path.lstrip('/').split(':')[0] if parsed.path else None
        }
    except Exception as e:
        print(f"解析连接字符串失败: {e}")
        return {}


def create_database(database_name):
    """创建新数据库"""
    try:
        connection = mysql.connector.connect(
            host=HOST,  
            user=USER,  
            password=PASSWORD,  
            port=PORT
        )
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"数据库 {database_name} 准备成功！")
        connection.close()
        return True
    except Error as e:
        print(f"创建数据库失败: {e}")
        return False

def create_connection(database=None):
    """创建数据库连接，使用动态配置"""
    try:
        # 使用运行时配置
        config = _dynamic_mysql_config.copy()
        if database:
            config["database"] = database

        # 验证配置完整性
        if not config["host"] or not config["user"]:
            raise ValueError("MySQL连接配置不完整，请确保已选择知识库")

        connection = mysql.connector.connect(
            host=config["host"],
            user=config["user"],
            password=config["password"] or "",
            port=config["port"],
            database=config.get("database") or database
        )
        # print("MySQL数据库连接成功")
        return connection
    except Error as e:
        print(f"连接MySQL数据库失败: {e}")
        return None

# def drop_database(database_name):
#     """删除数据库"""
#     try:
#         connection = mysql.connector.connect(
#             host=HOST,  
#             user=USER,  
#             password=PASSWORD,  
#             port=PORT
#         )
#         cursor = connection.cursor()
#         cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
#         print(f"数据库 {database_name} 删除成功")
#         connection.close()
#         return True
#     except Error as e:
#         print(f"删除数据库失败: {e}")
#         return False
    
def fetch_data(table_name, column, connection):
    """查询数据某列的数据"""
    try:
        cursor = connection.cursor(dictionary=True)  # 返回字典形式的结果
        sql = f"SELECT {column}, COUNT(*) AS count FROM {table_name} GROUP BY {column} ORDER BY count DESC LIMIT 100;"
        cursor.execute(sql)
        records = cursor.fetchall()
        result = []
        for row in records:
            result.append(row[column])
        return result
    except Error as e:
        print(f"查询数据失败: {e}")
        return None
    
def ex_sql(connection, sql):
    """查询数据某列的数据"""
    try:
        cursor = connection.cursor() 
        cursor.execute(sql)
        records = cursor.fetchall()
        #得到markdown结果
        column_names = [desc[0] for desc in cursor.description]

        # 构建Markdown表格
        markdown_lines = []

        # 添加表头
        header = "| " + " | ".join(column_names) + " |"
        markdown_lines.append(header)

        # 添加分隔线
        separator = "| " + " | ".join(["---"] * len(column_names)) + " |"
        markdown_lines.append(separator)

        # 添加数据行
        for record in records:
            # 处理每个字段，确保转换为字符串并处理None值
            row = "| " + " | ".join([str(field) if field is not None else "" for field in record]) + " |"
            markdown_lines.append(row)

        # 合并所有行
        markdown_table = "\n".join(markdown_lines)

        #得到markdown结果, 会用科学计数法，还是用上面的吧
        # column_names = [desc[0] for desc in cursor.description]
        # df = pd.DataFrame(records, columns=column_names)
        # markdown_table = df.to_markdown(index=False)
        return markdown_table
    except Error as e:
        print(f"查询数据失败: {e}")
        return str(e)
    
# def drop_table(connection, table_name="users"):
#     """删除表"""
#     try:
#         cursor = connection.cursor()
#         drop_query = f"DROP TABLE IF EXISTS {table_name}"
#         cursor.execute(drop_query)
#         connection.commit()
#         print(f"表 {table_name} 删除成功")
#         return f"表 {table_name} 删除成功"
#     except Error as e:
#         print(f"删除表失败: {e}")
#         return f"删除表失败: {e}"

def check_table_exist(connection, table_name):
    try:
        cursor = connection.cursor()
        sql = f"SHOW TABLES LIKE '{table_name}';"
        cursor.execute(sql)
        records = cursor.fetchall()
        if records:
            return True
        else:
            return False
    except:
        print(f"查询表是否存在失败: {e}")
        return True
    
def get_tables_with_details(database):
    """
    获取表名和建表语句，包含更多表信息
    使用动态配置连接数据库
    """
    connection = None
    try:
        connection = create_connection(database)
        if not connection:
            return {}

        cursor = connection.cursor()
        
        # 获取数据库所有表信息
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_COMMENT 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = %s
        """, (database,))
        
        tables_info = cursor.fetchall()
        result = {}
        
        for table_name, table_comment in tables_info:
            # 获取建表语句
            cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
            create_sql = cursor.fetchone()[1].replace(" ENGINE=InnoDB DEFAULT CHARSET=utf8", "")
            
            result[table_name] = {
                'create_sql': create_sql,
                'comment': table_comment
            }
        
        return result
        
    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        return {}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

def get_creat_table_details(database, table_name):
    """
    获取表名和建表语句，包含更多表信息
    使用动态配置连接数据库
    """
    connection = None
    try:
        connection = create_connection(database)
        if not connection:
            return {}

        cursor = connection.cursor()
        cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
        create_sql = cursor.fetchone()[1].replace(" ENGINE=InnoDB DEFAULT CHARSET=utf8", "")
        return create_sql

    except mysql.connector.Error as e:
        print(f"数据库错误: {e}")
        return {}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def from_db_get_column(database, table_name):
    """
    从数据库中得到指定表的所有列名以及对应的数据类型
    """
    connection = create_connection(database)
    if not connection:
        raise ValueError(f"无法连接到MySQL数据库: {database}")

    cursor = connection.cursor(dictionary=True)  # 返回字典形式的结果
    sql = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{database}' AND TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION;"
    cursor.execute(sql)
    records = cursor.fetchall()
    result = []
    for row in records:
        result.append(row)
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals() and connection.is_connected():
        connection.close()
    return result

def get_column_top_value(database, table_name, column, top_k=100):
    """查询数据某列的数据的频率top值"""
    try:
        connection = create_connection(database)
        if not connection:
            raise ValueError(f"无法连接到MySQL数据库: {database}")
        cursor = connection.cursor(dictionary=True)  # 返回字典形式的结果
        sql = f"SELECT {column}, COUNT(*) AS count FROM {table_name} GROUP BY {column} ORDER BY count DESC LIMIT {top_k};"
        cursor.execute(sql)
        records = cursor.fetchall()
        result = []
        for row in records:
            result.append(str(row[column]))
        return result
    except Error as e:
        print(f"查询数据失败: {e}")
        return None
    
def get_column_value(database, table_name, column, top_k=1):
    """查询数据某列的数据的频率top值"""
    try:
        connection = create_connection(database)
        if not connection:
            raise ValueError(f"无法连接到MySQL数据库: {database}")
        cursor = connection.cursor(dictionary=True)  # 返回字典形式的结果
        sql = f"SELECT {column} FROM {table_name} LIMIT {top_k};"
        cursor.execute(sql)
        records = cursor.fetchall()
        result = []
        for row in records:
            result.append(row[column])
        return result
    except Error as e:
        print(f"查询数据失败: {e}")
        return None
    
@function_tool
def mysql_excuted_sql_tool(sql):
    """
    sql执行工具，执行sql，并以md形式返回
    使用运行时配置的数据库连接（由schema link动态设置）
    """
    connection = create_connection()
    if not connection:
        return "错误：无法连接到数据库，请确保已正确配置数据库连接"
    ex_result = ex_sql(connection, sql)
    return ex_result

if __name__ == "__main__":
    db_id = 'wj_agenticrag_text2sql_multi_mini_2512'
    table_name = '综艺节目'
    sql = get_creat_table_details(db_id, table_name)
    print(sql)

