"""Base class for relational database monitoring."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class RelationalDBMonitor(ABC):
    """Abstract base class for monitoring relational databases.

    Should implement:
    - connect()
    - close()
    - _get_table_list()
    - _get_table_row_count()
    - _check_database_integrity()
    - _get_database_size()
    - get_metrics()

    Supports both local databases (SQLite) and network databases (MySQL, PostgreSQL).
    """

    def __init__(self, db_type: str):
        """Initialize relational database monitor.

        Args:
            db_type: Type of database (sqlite, mysql, postgresql)
        """
        self.db_type = db_type
        self.connection = None

    @abstractmethod
    def connect(self) -> bool:
        """Connect to database.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def _get_table_list(self) -> list[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        pass

    @abstractmethod
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in the table
        """
        pass

    @abstractmethod
    def _check_database_integrity(self) -> tuple[bool, str]:
        """Check database integrity.

        Returns:
            Tuple of (is_ok, message)
        """
        pass

    @abstractmethod
    def _get_database_size(self) -> int:
        """Get database size in bytes.

        Returns:
            Database size in bytes, or 0 if unable to determine
        """
        pass

    def check_health(self) -> dict[str, Any]:
        """Check database health.

        Returns:
            Health status dictionary
        """
        health = {
            "type": self.db_type,
            "is_healthy": False,
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            if not self.connect():
                health["errors"].append("Cannot connect to database")
                return health

            integrity_ok, integrity_msg = self._check_database_integrity()
            if not integrity_ok:
                health["errors"].append(f"Integrity check failed: {integrity_msg}")
                return health

            try:
                db_size = self._get_database_size()
                if db_size > 0:
                    health["metrics"]["size_bytes"] = db_size
                    health["metrics"]["size_mb"] = round(db_size / (1024 * 1024), 2)

                    # Warn if database is large
                    if db_size > 1024 * 1024 * 1024:  # > 1GB
                        health["warnings"].append(f"Large database: {db_size / (1024**3):.2f} GB")
            except Exception as e:
                logger.warning(f"Could not get database size: {str(e)}")

            tables = self._get_table_list()
            table_count = len(tables)
            health["metrics"]["table_count"] = table_count

            if table_count == 0:
                health["warnings"].append("No tables found in database")

            total_rows = 0
            table_stats = {}
            for table_name in tables:
                try:
                    row_count = self._get_table_row_count(table_name)
                    total_rows += row_count
                    table_stats[table_name] = row_count
                except Exception as e:
                    logger.warning(f"Could not count rows in table {table_name}: {str(e)}")

            health["metrics"]["total_rows"] = total_rows
            health["metrics"]["table_stats"] = table_stats

            health["is_healthy"] = True
            logger.info(f"{self.db_type.upper()} health check passed")

        except Exception as e:
            logger.error(f"{self.db_type.upper()} health check failed: {str(e)}")
            health["errors"].append(str(e))

        finally:
            self.close()

        return health

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get detailed database metrics.

        Returns:
            Metrics dictionary
        """
        pass
