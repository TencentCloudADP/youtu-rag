"""Unified monitoring for multiple storage backends."""

import logging
from datetime import datetime
from typing import Any

from .chroma_monitor import ChromaMonitor
from .minio_monitor import MinIOMonitor
from .mysql_monitor import MySQLMonitor
from .sqlite_monitor import SQLiteMonitor

logger = logging.getLogger(__name__)


class UnifiedStorageMonitor:
    """Monitor multiple storage backends in a unified way."""

    def __init__(self, config: dict[str, Any]):
        """Initialize unified storage monitor.

        Args:
            config: Configuration dictionary with storage settings
        
        Example:
            ```python
                config = {
                    "sqlite": {"db_path": "/path/to/db.sqlite"},
                    "mysql": {
                        "host": "localhost",
                        "port": 3306,
                        "user": "user",
                        "password": "password",
                        "database": "dbname",
                        "charset": "utf8mb4"
                    },
                    "postgresql": {
                        "host": "localhost",
                        "port": 5432,
                        "user": "user",
                        "password": "password",
                        "database": "dbname",
                        "sslmode": "prefer"
                    },
                    "minio": {  # Legacy single bucket
                        "endpoint": "localhost:9000",
                        "access_key": "...",
                        "secret_key": "...",
                        "bucket_name": "...",
                        "secure": False
                    },
                    "minio_user": {  # User bucket
                        "endpoint": "localhost:9000",
                        "access_key": "...",
                        "secret_key": "...",
                        "bucket_name": "...",
                        "secure": False
                    },
                    "minio_sys": {  # System bucket
                        "endpoint": "localhost:9000",
                        "access_key": "...",
                        "secret_key": "...",
                        "bucket_name": "...",
                        "secure": False
                    },
                    "vector_store": <VectorStore instance>,
                    "enabled_monitors": ["sqlite", "mysql", "postgresql", "minio", "minio_user", "minio_sys", "vector_store"]
                }
            ```
        """
        self.config = config
        self.monitors = {}
        self.enabled_monitors = config.get("enabled_monitors", [])

        self._init_monitors()

    def _init_monitors(self) -> None:
        """Initialize configured monitors."""
        if "sqlite" in self.enabled_monitors and "sqlite" in self.config:
            try:
                sqlite_config = self.config["sqlite"]
                self.monitors["sqlite"] = SQLiteMonitor(db_path=sqlite_config["db_path"])
                logger.info("SQLite monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SQLite monitor: {str(e)}")

        if "mysql" in self.enabled_monitors and "mysql" in self.config:
            try:
                mysql_config = self.config["mysql"]
                self.monitors["mysql"] = MySQLMonitor(
                    host=mysql_config["host"],
                    port=mysql_config.get("port", 3306),
                    user=mysql_config["user"],
                    password=mysql_config["password"],
                    database=mysql_config["database"],
                    charset=mysql_config.get("charset", "utf8mb4"),
                )
                logger.info("MySQL monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MySQL monitor: {str(e)}")

        # legacy single bucket
        if "minio" in self.enabled_monitors and "minio" in self.config:
            try:
                minio_config = self.config["minio"]
                self.monitors["minio"] = MinIOMonitor(
                    endpoint=minio_config["endpoint"],
                    access_key=minio_config["access_key"],
                    secret_key=minio_config["secret_key"],
                    bucket_name=minio_config["bucket_name"],
                    secure=minio_config.get("secure", True),
                )
                logger.info("MinIO monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO monitor: {str(e)}")

        if "minio_user" in self.enabled_monitors and "minio_user" in self.config:
            try:
                minio_config = self.config["minio_user"]
                self.monitors["minio_user"] = MinIOMonitor(
                    endpoint=minio_config["endpoint"],
                    access_key=minio_config["access_key"],
                    secret_key=minio_config["secret_key"],
                    bucket_name=minio_config["bucket_name"],
                    secure=minio_config.get("secure", True),
                )
                logger.info(f"MinIO user bucket monitor initialized: {minio_config['bucket_name']}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO user bucket monitor: {str(e)}")

        if "minio_sys" in self.enabled_monitors and "minio_sys" in self.config:
            try:
                minio_config = self.config["minio_sys"]
                self.monitors["minio_sys"] = MinIOMonitor(
                    endpoint=minio_config["endpoint"],
                    access_key=minio_config["access_key"],
                    secret_key=minio_config["secret_key"],
                    bucket_name=minio_config["bucket_name"],
                    secure=minio_config.get("secure", True),
                )
                logger.info(f"MinIO system bucket monitor initialized: {minio_config['bucket_name']}")
            except Exception as e:
                logger.error(f"Failed to initialize MinIO system bucket monitor: {str(e)}")

        if "chroma" in self.enabled_monitors and "chroma" in self.config:
            try:
                chroma_config = self.config["chroma"]
                self.monitors["chroma"] = ChromaMonitor(
                    persist_directory=chroma_config["persist_directory"]
                )
                logger.info(f"ChromaDB monitor initialized: {chroma_config['persist_directory']}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB monitor: {str(e)}")

        # legacy - for backward compatibility
        if "vector_store" in self.enabled_monitors and "vector_store" in self.config:
            try:
                from .health_checker import HealthChecker

                vector_store = self.config["vector_store"]
                self.monitors["vector_store"] = HealthChecker(vector_store=vector_store)
                logger.info("Vector store monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vector store monitor: {str(e)}")

    def check_all_health(self) -> dict[str, Any]:
        """Check health of all configured storage backends.

        Returns:
            Dictionary with health status for each backend
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "backends": {},
        }

        for name, monitor in self.monitors.items():
            try:
                if name == "vector_store":
                    import asyncio

                    health = asyncio.run(monitor.check_health())
                    backend_health = {
                        "type": "vector_store",
                        "is_healthy": health.is_healthy,
                        "backend": health.backend,
                        "collection_name": health.collection_name,
                        "metrics": {
                            "total_chunks": health.total_chunks,
                            "collection_name": health.collection_name,
                        },
                        "errors": health.errors,
                        "warnings": health.warnings,
                        "timestamp": health.last_check_time,
                    }
                else:
                    backend_health = monitor.check_health()

                health_report["backends"][name] = backend_health

                if not backend_health.get("is_healthy", False):
                    health_report["overall_healthy"] = False

            except Exception as e:
                logger.error(f"Error checking health for {name}: {str(e)}")
                health_report["backends"][name] = {
                    "type": name,
                    "is_healthy": False,
                    "errors": [str(e)],
                    "timestamp": datetime.now().isoformat(),
                }
                health_report["overall_healthy"] = False

        return health_report

    def get_all_metrics(self) -> dict[str, Any]:
        """Get metrics from all configured storage backends.

        Returns:
            Dictionary with metrics for each backend
        """
        metrics_report = {
            "timestamp": datetime.now().isoformat(),
            "backends": {},
        }

        for name, monitor in self.monitors.items():
            try:
                if name == "vector_store":
                    # Vector store doesn't have get_metrics, skip or use alternative
                    metrics_report["backends"][name] = {
                        "type": "vector_store",
                        "note": "Use health check for vector store metrics",
                    }
                else:
                    metrics = monitor.get_metrics()
                    metrics_report["backends"][name] = metrics

            except Exception as e:
                logger.error(f"Error collecting metrics for {name}: {str(e)}")
                metrics_report["backends"][name] = {
                    "type": name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        return metrics_report

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all storage backends.

        Returns:
            Summary dictionary
        """
        health = self.check_all_health()

        summary = {
            "timestamp": health["timestamp"],
            "overall_healthy": health["overall_healthy"],
            "total_backends": len(self.monitors),
            "healthy_backends": sum(
                1 for b in health["backends"].values() if b.get("is_healthy", False)
            ),
            "unhealthy_backends": sum(
                1 for b in health["backends"].values() if not b.get("is_healthy", False)
            ),
            "backend_status": {},
        }

        for name, backend_health in health["backends"].items():
            status = "healthy" if backend_health.get("is_healthy", False) else "unhealthy"
            summary["backend_status"][name] = {
                "status": status,
                "type": backend_health.get("type", name),
                "error_count": len(backend_health.get("errors", [])),
                "warning_count": len(backend_health.get("warnings", [])),
            }

        return summary

    def get_detailed_report(self) -> dict[str, Any]:
        """Get a detailed report including health and metrics.

        Returns:
            Detailed report dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "health": self.check_all_health(),
            "metrics": self.get_all_metrics(),
        }
