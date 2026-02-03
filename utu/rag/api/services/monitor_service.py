"""Storage Monitoring Service.

This module provides a unified interface for storage monitoring to the routing layer.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any

from utu.rag.monitoring import UnifiedStorageMonitor
from utu.utils.path import FileUtils, DIR_ROOT
from utu.utils.log import get_logger

logger = get_logger("utu.rag.api.monitor_service")


class MonitorService:
    """Service interface for storage monitoring (singleton)."""

    _instance = None
    _monitor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_monitor()
        return cls._instance

    def _initialize_monitor(self) -> None:
        """Initialize the storage monitor."""
        config_path = DIR_ROOT / "configs" / "rag" / "default.yaml"
        monitor_config = self._load_config(config_path)
        self._monitor = UnifiedStorageMonitor(config=monitor_config)
        logger.info("Storage monitor initialized")

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load monitoring configuration.

        Args:
            config_path: Path to the configuration file.

        Returns:
            A dictionary containing the monitoring configuration.
        """
        # Default configuration
        monitor_config = {
            "enabled_monitors": [],
        }

        # Load YAML configuration (if exists)
        yaml_config = {}
        if config_path.exists():
            try:
                yaml_config = FileUtils.load_yaml(config_path)
                logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load YAML config: {e}. Using environment variables and defaults.")
        else:
            logger.info(f"YAML config not found at {config_path}. Using environment variables and defaults.")

        def get_config(env_key: str, yaml_path: list, default: Any = None) -> Any:
            """Get configuration value with priority: ENV > YAML > Default."""
            env_value = os.getenv(env_key)
            if env_value is not None:
                return env_value

            if yaml_path and yaml_config:
                yaml_value = yaml_config
                for key in yaml_path:
                    if isinstance(yaml_value, dict) and key in yaml_value:
                        yaml_value = yaml_value[key]
                    else:
                        yaml_value = None
                        break
                if yaml_value is not None:
                    return yaml_value

            return default

        # Determine which monitors are enabled.
        enabled_monitors_yaml = yaml_config.get("enabled_monitors", [])

        # Configure vector store (ChromaDB - monitor all collections)
        vector_enabled = get_config("ENABLE_VECTOR_MONITOR", [], None)
        if vector_enabled is not None:
            vector_enabled = str(vector_enabled).lower() == "true"
        else:
            vector_enabled = "chroma" in enabled_monitors_yaml or "vector_store" in enabled_monitors_yaml

        if vector_enabled:
            vector_store_path = get_config(
                "VECTOR_STORE_PATH",
                ["vector_store", "persist_directory"],
                "./rag_data/vector_store"
            )

            monitor_config["chroma"] = {
                "persist_directory": vector_store_path
            }
            monitor_config["enabled_monitors"].append("chroma")
            logger.info(f"ChromaDB monitor enabled at {vector_store_path}")

        # Configure relational database (SQLite)
        db_url = get_config(
            "UTU_DB_URL",
            ["relational_database", "sqlite", "db_url"],
            None
        )

        if db_url:
            # Parse database URL (format: sqlite:///./path/to/db.sqlite)
            if db_url.startswith("sqlite:///"):
                sqlite_db_path = db_url.replace("sqlite:///", "")
            else:
                sqlite_db_path = str(db_url)
                logger.warning(f"UTU_DB_URL doesn't start with 'sqlite:///', treating as direct path: {db_url}")

            monitor_config["sqlite"] = {"db_path": sqlite_db_path}
            monitor_config["enabled_monitors"].append("sqlite")
            logger.info(f"SQLite monitor enabled: {sqlite_db_path}")

        # Configure MinIO object storage monitoring
        minio_enabled = get_config("ENABLE_MINIO_MONITOR", [], None)
        if minio_enabled is not None:
            minio_enabled = str(minio_enabled).lower() == "true"
        else:
            minio_enabled = "object_storage" in enabled_monitors_yaml

        if minio_enabled:
            minio_endpoint = get_config("MINIO_ENDPOINT", ["object_storage", "minio", "endpoint"], "localhost:9000")
            minio_access_key = get_config("MINIO_ACCESS_KEY", [], "minioadmin")
            minio_secret_key = get_config("MINIO_SECRET_KEY", [], "minioadmin")
            minio_secure = str(get_config("MINIO_SECURE", ["object_storage", "minio", "secure"], "false")).lower() == "true"
            protocol = "https" if minio_secure else "http"

            # Monitor user file bucket (MINIO_BUCKET)
            user_bucket = get_config("MINIO_BUCKET", ["object_storage", "minio", "bucket_name"], "ufile")
            monitor_config["minio_user"] = {
                "endpoint": minio_endpoint,
                "access_key": minio_access_key,
                "secret_key": minio_secret_key,
                "bucket_name": user_bucket,
                "secure": minio_secure,
            }
            monitor_config["enabled_monitors"].append("minio_user")
            logger.info(f"MinIO user bucket monitor enabled ({protocol}://{minio_endpoint}/{user_bucket})")

            # Monitor system file bucket (MINIO_BUCKET_SYS)
            sys_bucket = get_config("MINIO_BUCKET_SYS", ["object_storage", "minio", "bucket_name_sys"], "sysfile")
            monitor_config["minio_sys"] = {
                "endpoint": minio_endpoint,
                "access_key": minio_access_key,
                "secret_key": minio_secret_key,
                "bucket_name": sys_bucket,
                "secure": minio_secure,
            }
            monitor_config["enabled_monitors"].append("minio_sys")
            logger.info(f"MinIO system bucket monitor enabled ({protocol}://{minio_endpoint}/{sys_bucket})")

        return monitor_config

    def get_storage_health(self) -> Dict[str, Any]:
        """Get health status of all storage backends (JSON format)

        Returns:
            Dictionary containing health status of all storage backends.
        """
        try:
            return self._monitor.check_all_health()
        except Exception as e:
            logger.error(f"Error getting storage health: {str(e)}")
            return {
                "overall_healthy": False,
                "backends": {},
                "timestamp": "",
                "error": str(e)
            }

    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics (JSON format).

        Returns:
            Dictionary containing detailed metrics of all storage backends.
        """
        try:
            return self._monitor.get_all_metrics()
        except Exception as e:
            logger.error(f"Error getting storage metrics: {str(e)}")
            return {"error": str(e)}

    def get_storage_dashboard_html(self) -> str:
        """Get storage dashboard (HTML format).

        Returns:
            Formatted HTML string.
        """
        try:
            health = self.get_storage_health()
            return self._format_health_html(health)
        except Exception as e:
            logger.error(f"Error formatting dashboard HTML: {str(e)}")
            return f'<div style="color: red; padding: 20px;">Error: {str(e)}</div>'

    def _format_health_html(self, health: Dict[str, Any]) -> str:
        """Format health status as HTML.

        Args:
            health: Health status dictionary.

        Returns:
            Formatted HTML string.
        """
        # The main categories -- use placeholders if missing
        categorized_backends = {
            "Vector Store": [],
            "Relational Database": [],
            "Object Storage": [],
        }

        # Track which main categories have actual backends
        has_vector_store = False
        has_relational_db = False
        has_object_storage = False

        # Classify backends by name
        for name, backend in health["backends"].items():
            if name in ["vector_store", "chroma"]:
                has_vector_store = True
                category = "Vector Store"
                display_name = "ChromaDB (All Collections)" if name == "chroma" else "Vector Store"
            elif name in ["sqlite"]:
                has_relational_db = True
                category = "Relational Database"
                display_name = "SQLite"
            elif name in ["minio", "minio_user", "minio_sys"]:
                has_object_storage = True
                category = "Object Storage"
                if name == "minio_user":
                    display_name = "MinIO User Bucket"
                elif name == "minio_sys":
                    display_name = "MinIO System Bucket"
                else:
                    display_name = "MinIO"
            else:
                category = "Other"
                display_name = name.upper()
                if category not in categorized_backends:
                    categorized_backends[category] = []

            categorized_backends[category].append((name, display_name, backend))

        # Add placeholders for missing main categories
        if not has_vector_store:
            categorized_backends["Vector Store"].append((
                "vector_store_placeholder",
                "Vector Store",
                {
                    "is_healthy": False,
                    "type": "Not Configured",
                    "errors": ["Vector store monitoring is not enabled"],
                }
            ))

        if not has_relational_db:
            categorized_backends["Relational Database"].append((
                "database_placeholder",
                "Database",
                {
                    "is_healthy": False,
                    "type": "Not Configured",
                    "errors": ["Database monitoring is not enabled"],
                }
            ))

        if not has_object_storage:
            categorized_backends["Object Storage"].append((
                "object_storage_placeholder",
                "Object Storage",
                {
                    "is_healthy": False,
                    "type": "Not Configured",
                    "warnings": ["Object storage monitoring is not enabled"],
                }
            ))

        # Calculate health statistics
        total_backends = len([b for b in health["backends"].values() if not b.get("type") == "Not Configured"])
        healthy_backends = len([b for b in health["backends"].values() if b.get("is_healthy", False)])
        health_percentage = int((healthy_backends / total_backends * 100)) if total_backends > 0 else 0

        # Build HTML
        html_parts = []

        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Storage Health Monitor</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    background: #f5f7fa;
                    padding: 20px;
                }
                .storage-overview {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .overview-header {
                    margin-bottom: 30px;
                    padding: 30px;
                    border-radius: 16px;
                    color: white;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
                }
                .overview-header.healthy {
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                }
                .overview-header.unhealthy {
                    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                }
                .header-content {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 30px;
                }
                .header-left { flex: 1; }
                .overview-header h1 {
                    margin: 0 0 15px 0;
                    font-size: 32px;
                    font-weight: 700;
                }
                .overview-meta {
                    font-size: 14px;
                    opacity: 0.95;
                    margin: 5px 0;
                }
                .status-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 12px;
                    background: rgba(255, 255, 255, 0.2);
                    padding: 12px 24px;
                    border-radius: 50px;
                    font-size: 18px;
                    font-weight: 600;
                    margin-top: 15px;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 15px;
                    margin-top: 20px;
                }
                .stat-card {
                    background: rgba(255, 255, 255, 0.15);
                    padding: 15px;
                    border-radius: 12px;
                    text-align: center;
                }
                .stat-value {
                    font-size: 32px;
                    font-weight: 700;
                }
                .stat-label {
                    font-size: 12px;
                    opacity: 0.9;
                    margin-top: 8px;
                    text-transform: uppercase;
                }
                .health-circle {
                    width: 160px;
                    height: 160px;
                    border-radius: 50%;
                    background: rgba(255, 255, 255, 0.2);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
                }
                .health-percentage {
                    font-size: 48px;
                    font-weight: 800;
                }
                .health-label {
                    font-size: 14px;
                    opacity: 0.9;
                    margin-top: 5px;
                }
                .storage-card {
                    margin-bottom: 25px;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                    background: white;
                }
                .storage-header {
                    padding: 20px 24px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    background: linear-gradient(135deg, #6b7280 0%, #9ca3af 100%);
                    color: white;
                }
                .storage-icon { font-size: 24px; }
                .storage-title {
                    font-size: 20px;
                    font-weight: 600;
                    margin: 0 8px 0 0;
                    flex: 1;
                }
                .status-indicator {
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                }
                .status-indicator.healthy { background-color: #22c55e; }
                .status-indicator.unhealthy { background-color: #ef4444; }
                .storage-body {
                    padding: 24px;
                }
                .info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 16px;
                    margin-bottom: 16px;
                }
                .info-item {
                    padding: 12px 16px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border-left: 3px solid #3b82f6;
                }
                .info-label {
                    font-size: 12px;
                    color: #64748b;
                    text-transform: uppercase;
                    margin-bottom: 4px;
                }
                .info-value {
                    font-size: 16px;
                    color: #1e293b;
                    font-weight: 600;
                }
                .alert {
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    display: flex;
                    gap: 10px;
                }
                .alert.error {
                    background: #fee2e2;
                    border-left: 4px solid #ef4444;
                    color: #991b1b;
                }
                .alert.warning {
                    background: #fef3c7;
                    border-left: 4px solid #f59e0b;
                    color: #92400e;
                }
            </style>
        </head>
        <body>
        """)

        html_parts.append('<div class="storage-overview">')

        # Header
        header_class = "healthy" if health["overall_healthy"] else "unhealthy"
        html_parts.append(f'<div class="overview-header {header_class}">')
        html_parts.append('<div class="header-content">')

        # Left side - title and info
        html_parts.append('<div class="header-left">')
        html_parts.append('<h1>📊 Storage Health Status</h1>')
        html_parts.append(f'<div class="overview-meta">🕐 Last Check: {health["timestamp"]}</div>')

        # Status badge
        status_icon = "✅" if health["overall_healthy"] else "⚠️"
        status_text = "All Systems Operational" if health["overall_healthy"] else "System Issues Detected"
        html_parts.append(f'<div class="status-badge"><span>{status_icon}</span><span>{status_text}</span></div>')

        # Stats grid
        unhealthy_backends = total_backends - healthy_backends
        html_parts.append('<div class="stats-grid">')
        html_parts.append(f'<div class="stat-card"><div class="stat-value">{total_backends}</div><div class="stat-label">Total</div></div>')
        html_parts.append(f'<div class="stat-card"><div class="stat-value">{healthy_backends}</div><div class="stat-label">✅ Healthy</div></div>')
        html_parts.append(f'<div class="stat-card"><div class="stat-value">{unhealthy_backends}</div><div class="stat-label">❌ Unhealthy</div></div>')
        html_parts.append('</div>')  # stats-grid
        html_parts.append('</div>')  # header-left

        # Right side - health circle
        html_parts.append('<div class="health-circle">')
        html_parts.append(f'<div class="health-percentage">{health_percentage}%</div>')
        html_parts.append('<div class="health-label">Health Score</div>')
        html_parts.append('</div>')

        html_parts.append('</div>')  # header-content
        html_parts.append('</div>')  # overview-header

        # Storage categories
        category_order = ["Vector Store", "Relational Database", "Object Storage", "Other"]

        for category in category_order:
            if category == "Other" and category not in categorized_backends:
                continue
            if category not in categorized_backends or not categorized_backends[category]:
                continue

            for name, display_name, backend in categorized_backends[category]:
                is_healthy = backend.get("is_healthy", False)
                health_class = "healthy" if is_healthy else "unhealthy"

                if name in ["vector_store", "chroma"]:
                    icon = "🔢"
                elif name in ["sqlite", "database_placeholder"]:
                    icon = "🗄️"
                elif name in ["minio", "minio_user", "minio_sys", "object_storage_placeholder"]:
                    icon = "🧳"
                else:
                    icon = "📦"

                html_parts.append(f'<div class="storage-card">')
                html_parts.append(f'<div class="storage-header">')
                html_parts.append(f'<div class="storage-icon">{icon}</div>')
                html_parts.append(f'<h2 class="storage-title">{display_name}</h2>')
                html_parts.append(f'<div class="status-indicator {health_class}"></div>')
                html_parts.append('</div>')

                html_parts.append('<div class="storage-body">')

                # Basic info
                html_parts.append('<div class="info-grid">')
                html_parts.append('<div class="info-item">')
                html_parts.append('<div class="info-label">Type</div>')
                html_parts.append(f'<div class="info-value">{backend.get("type", "unknown")}</div>')
                html_parts.append('</div>')
    
                # Backend-specific metrics
                if "metrics" in backend:
                    metrics = backend["metrics"]
                    for key, value in metrics.items():
                        html_parts.append('<div class="info-item">')
                        html_parts.append(f'<div class="info-label">{key.replace("_", " ").title()}</div>')
                        html_parts.append(f'<div class="info-value">{value}</div>')
                        html_parts.append('</div>')

                html_parts.append('</div>')  # info-grid

                # Errors and warnings
                if backend.get("errors") or backend.get("warnings"):
                    for error in backend.get("errors", []):
                        html_parts.append(f'<div class="alert error">🔴 {error}</div>')
                    for warning in backend.get("warnings", []):
                        html_parts.append(f'<div class="alert warning">⚠️ {warning}</div>')

                html_parts.append('</div>')  # storage-body
                html_parts.append('</div>')  # storage-card

        html_parts.append('</div>')  # storage-overview
        html_parts.append('</body></html>')

        return "".join(html_parts)
