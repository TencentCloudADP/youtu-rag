"""MySQL database monitoring."""

import logging
from datetime import datetime
from typing import Any

from .relational_db_monitor import RelationalDBMonitor

logger = logging.getLogger(__name__)


class MySQLMonitor(RelationalDBMonitor):
    """Monitor MySQL database health and metrics."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
    ):
        """Initialize MySQL monitor.

        Args:
            host: MySQL server host
            port: MySQL server port
            user: MySQL username
            password: MySQL password
            database: Database name
            charset: Character set (default: utf8mb4)
        """
        super().__init__(db_type="mysql")
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.connection = None

        # Try to import pymysql
        try:
            import pymysql

            self.pymysql = pymysql
            self.available = True
        except ImportError:
            logger.error("pymysql not installed. Install with: pip install pymysql")
            self.available = False
            self.pymysql = None

    def connect(self) -> bool:
        """Connect to MySQL database.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.available:
            logger.error("pymysql is not available")
            return False

        try:
            self.connection = self.pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing MySQL connection: {str(e)}")
            finally:
                self.connection = None

    def _get_table_list(self) -> list[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables

    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def _check_database_integrity(self) -> tuple[bool, str]:
        """Check database integrity.

        For MySQL, we check if we can query system tables.

        Returns:
            Tuple of (is_ok, message)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            if result and result[0] == 1:
                return True, "ok"
            return False, "Unexpected query result"
        except Exception as e:
            return False, str(e)

    def _get_database_size(self) -> int:
        """Get database size in bytes.

        Returns:
            Database size in bytes
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT SUM(data_length + index_length) as size
                FROM information_schema.TABLES
                WHERE table_schema = %s
                """,
                (self.database,),
            )
            result = cursor.fetchone()
            cursor.close()
            return int(result[0]) if result[0] else 0
        except Exception as e:
            logger.warning(f"Could not get database size: {str(e)}")
            return 0

    def get_metrics(self) -> dict[str, Any]:
        """Get detailed MySQL metrics, including
        - MySQL version
        - Current database
        - Character set
        - Collation
        - Max connections
        - Current connections
        - Uptime
        - Table details
        - Index details

        Returns:
            Metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "type": "mysql",
            "connection_info": {
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "user": self.user,
            },
        }

        try:
            if not self.connect():
                metrics["error"] = "Cannot connect to database"
                return metrics

            cursor = self.connection.cursor()

            cursor.execute("SELECT VERSION()")
            metrics["version"] = cursor.fetchone()[0]

            cursor.execute("SELECT DATABASE()")
            metrics["current_database"] = cursor.fetchone()[0]

            cursor.execute("SHOW VARIABLES LIKE 'character_set_database'")
            result = cursor.fetchone()
            metrics["character_set"] = result[1] if result else None

            cursor.execute("SHOW VARIABLES LIKE 'collation_database'")
            result = cursor.fetchone()
            metrics["collation"] = result[1] if result else None

            cursor.execute("SHOW VARIABLES LIKE 'max_connections'")
            result = cursor.fetchone()
            metrics["max_connections"] = int(result[1]) if result else None

            cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
            result = cursor.fetchone()
            metrics["current_connections"] = int(result[1]) if result else None

            cursor.execute("SHOW STATUS LIKE 'Uptime'")
            result = cursor.fetchone()
            metrics["uptime_seconds"] = int(result[1]) if result else None

            cursor.execute(
                """
                SELECT table_name, engine, table_rows,
                       data_length, index_length, auto_increment
                FROM information_schema.TABLES
                WHERE table_schema = %s
                ORDER BY table_name
                """,
                (self.database,),
            )

            tables = []
            for row in cursor.fetchall():
                table_info = {
                    "name": row[0],
                    "engine": row[1],
                    "rows": row[2],
                    "data_length": row[3],
                    "index_length": row[4],
                    "auto_increment": row[5],
                }
                tables.append(table_info)

            metrics["tables"] = tables

            cursor.execute(
                """
                SELECT table_name, index_name, non_unique, column_name
                FROM information_schema.STATISTICS
                WHERE table_schema = %s
                ORDER BY table_name, index_name, seq_in_index
                """,
                (self.database,),
            )

            indexes = []
            for row in cursor.fetchall():
                index_info = {
                    "table": row[0],
                    "name": row[1],
                    "unique": not bool(row[2]),
                    "column": row[3],
                }
                indexes.append(index_info)

            metrics["indexes"] = indexes

            cursor.close()

        except Exception as e:
            logger.error(f"Error collecting MySQL metrics: {str(e)}")
            metrics["error"] = str(e)

        finally:
            self.close()

        return metrics
