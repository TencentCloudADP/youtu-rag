"""SQLite database monitoring."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .relational_db_monitor import RelationalDBMonitor

logger = logging.getLogger(__name__)


class SQLiteMonitor(RelationalDBMonitor):
    """Monitor SQLite database health and metrics."""

    def __init__(self, db_path: str):
        """Initialize SQLite monitor.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(db_type="sqlite")
        self.db_path = Path(db_path)
        self.connection = None

    def connect(self) -> bool:
        """Connect to SQLite database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def _get_table_list(self) -> list[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
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
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def _check_database_integrity(self) -> tuple[bool, str]:
        """Check database integrity.

        Returns:
            Tuple of (is_ok, message)
        """
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        cursor.close()
        return (result == "ok", result)

    def _get_database_size(self) -> int:
        """Get database size in bytes.

        Returns:
            Database size in bytes
        """
        if not self.db_path.exists():
            return 0
        return self.db_path.stat().st_size

    def check_health(self) -> dict[str, Any]:
        """Check SQLite database health.

        Returns:
            Health status dictionary
        """
        health = {
            "type": "sqlite",
            "is_healthy": False,
            "timestamp": datetime.now().isoformat(),
            "db_path": str(self.db_path),
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            if not self.db_path.exists():
                health["errors"].append(f"Database file not found: {self.db_path}")
                return health

            # Use base class check_health which will call our implementation methods
            base_health = super().check_health()

            # Merge base health with SQLite-specific info
            health.update(base_health)
            health["db_path"] = str(self.db_path)

            # Add SQLite-specific metrics
            if self.connect():
                try:
                    cursor = self.connection.cursor()

                    cursor.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    cursor.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]

                    health["metrics"]["page_size"] = page_size
                    health["metrics"]["page_count"] = page_count

                    cursor.execute("PRAGMA cache_size")
                    cache_size = cursor.fetchone()[0]
                    health["metrics"]["cache_size"] = cache_size

                    cursor.close()
                finally:
                    self.close()

        except Exception as e:
            logger.error(f"SQLite health check failed: {str(e)}")
            health["errors"].append(str(e))
            self.close()

        return health

    def get_metrics(self) -> dict[str, Any]:
        """Get detailed SQLite metrics.

        Returns:
            Metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "type": "sqlite",
        }

        try:
            if not self.connect():
                metrics["error"] = "Cannot connect to database"
                return metrics

            cursor = self.connection.cursor()

            cursor.execute("PRAGMA database_list")
            db_info = cursor.fetchall()
            metrics["databases"] = [{"name": name, "file": file} for _, name, file in db_info]

            cursor.execute("PRAGMA journal_mode")
            metrics["journal_mode"] = cursor.fetchone()[0]

            cursor.execute("PRAGMA synchronous")
            metrics["synchronous"] = cursor.fetchone()[0]

            cursor.execute("PRAGMA auto_vacuum")
            metrics["auto_vacuum"] = cursor.fetchone()[0]

            cursor.execute("PRAGMA user_version")
            metrics["user_version"] = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            tables = cursor.fetchall()
            metrics["table_schemas"] = {name: sql for name, sql in tables}

            cursor.execute(
                """
                SELECT name, tbl_name, sql FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            indexes = cursor.fetchall()
            metrics["indexes"] = [{"name": name, "table": tbl, "sql": sql} for name, tbl, sql in indexes]

        except Exception as e:
            logger.error(f"Error collecting SQLite metrics: {str(e)}")
            metrics["error"] = str(e)

        finally:
            self.close()

        return metrics

    def vacuum_database(self) -> dict[str, Any]:
        """Run VACUUM to optimize database.

        Returns:
            Result dictionary
        """
        result = {"success": False, "message": "", "timestamp": datetime.now().isoformat()}

        try:
            if not self.connect():
                result["message"] = "Cannot connect to database"
                return result

            size_before = self.db_path.stat().st_size

            cursor = self.connection.cursor()
            cursor.execute("VACUUM")
            self.connection.commit()

            size_after = self.db_path.stat().st_size
            space_saved = size_before - size_after

            result["success"] = True
            result["message"] = "VACUUM completed successfully"
            result["size_before_mb"] = round(size_before / (1024 * 1024), 2)
            result["size_after_mb"] = round(size_after / (1024 * 1024), 2)
            result["space_saved_mb"] = round(space_saved / (1024 * 1024), 2)

            logger.info(f"Database VACUUM completed, saved {space_saved / (1024 * 1024):.2f} MB")

        except Exception as e:
            logger.error(f"VACUUM failed: {str(e)}")
            result["message"] = f"VACUUM failed: {str(e)}"

        finally:
            self.close()

        return result
