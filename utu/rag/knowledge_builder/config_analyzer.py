"""Knowledge Builder Configuration Analyzer and Storage Initializer.

This tool analyzes the knowledge base configuration and storage state,
and generates a comprehensive storage plan.

Supports:
1. Collect kb_source_configs statistics (file_count, qa_file_count, db_connections)
2. Collect basic configuration
3. Collect storage state (configs/rag/monitoring/storage_monitoring.yaml)
4. Output storage plan
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from ...utils import get_logger

logger = get_logger(__name__)


# ==================== Pydantic Models ====================


class SourceStatistics(BaseModel):
    file_count: int = 0
    qa_file_count: int = 0
    db_connections: int = 0

    file_type_count: dict[str, int] = {}  # e.g. {"pdf": 2, "xlsx": 3, ...}

    files: list[dict[str, Any]] = []
    qa_files: list[dict[str, Any]] = []
    databases: list[dict[str, Any]] = []


class StorageState(BaseModel):
    vector_store_type: str = "chroma"  # chroma, faiss
    vector_store_exists: bool = False
    vector_collection_name: Optional[str] = None
    vector_persist_dir: str = "./rag_data/vector_store"

    relational_db_type: str = "sqlite"  # sqlite, mysql
    relational_db_path: str = "./rag_data/relational_database/rag_demo.sqlite"
    relational_db_exists: bool = False

    object_storage_enabled: bool = True
    object_storage_type: str = "minio"
    minio_endpoint: Optional[str] = None
    minio_bucket: Optional[str] = None


class FileProcessingPlan(BaseModel):
    file_type: str  # pdf, docx, txt, xlsx, etc.
    file_count: int
    processor: str  # PDFProcessor, WordProcessor, ExcelProcessor, etc.
    storage_targets: list[str]  # e.g. ["ChromaDB", "SQLite", ...]
    metadata_sync: bool = False  # Whether to sync MinIO metadata to ChromaDB
    metadata_fields: list[str] = []  # Fields to sync


class DatabaseProcessingPlan(BaseModel):
    db_type: str  # mysql, sqlite
    connection_string: str
    table_count: int
    tables: list[str]
    processor: str = "DatabaseProcessor"
    storage_targets: list[str]  # ["SQLite (mirror)", "ChromaDB (metadata)"]


class StoragePlan(BaseModel):
    kb_id: int
    kb_name: str

    source_stats: SourceStatistics

    storage_state: StorageState

    file_processing_plans: list[FileProcessingPlan] = []
    database_processing_plans: list[DatabaseProcessingPlan] = []

    kb_config: dict[str, Any] = {}  
    storage_monitoring_config: dict[str, Any] = {}  # Read from configs/rag/monitoring/storage_monitoring.yaml

    created_at: datetime = datetime.now(timezone.utc)


# ==================== Analyzer Tool ====================


class KnowledgeBuilderAnalyzer:
    def __init__(
        self,
        kb_config_path: str | None = None,
        storage_monitoring_config_path: str | None = None,
    ):
        """Initialize the analyzer.

        Args:
            kb_config_path: Knowledge base configuration path
            storage_monitoring_config_path: Storage monitoring configuration path
        """
        self.kb_config_path = kb_config_path or "configs/rag/default.yaml"
        self.storage_monitoring_config_path = kb_config_path or "configs/rag/default.yaml"

        logger.info(f"KnowledgeBuilderAnalyzer initialized")
        logger.info(f"  KB Config: {self.kb_config_path}")
        logger.info(f"  Storage Monitoring: {self.storage_monitoring_config_path}")

    def analyze(
        self,
        kb_id: int,
        kb_name: str,
        sources: list[dict[str, Any]],
        collection_name: str | None = None,
    ) -> StoragePlan:
        """Complete configuration analysis and storage initialization analysis.

        Args:
            kb_id: Knowledge base ID
            kb_name: Knowledge base name
            sources: Data source configuration list (from kb_source_configs)
            collection_name: Vector store collection name

        Returns:
            StoragePlan: Storage plan
        """
        logger.info("=" * 80)
        logger.info("开始配置分析和存储初始化分析")
        logger.info("=" * 80)

        logger.info("\n[Step 1/5] 收集数据源统计信息...")
        source_stats = self._collect_source_statistics(sources)
        self._log_source_statistics(source_stats)

        logger.info("\n[Step 2/5] 读取知识库基本配置...")
        kb_config = self._load_kb_config()

        logger.info("\n[Step 3/5] 读取存储监控配置...")
        storage_monitoring_config = self._load_storage_monitoring_config()

        logger.info("\n[Step 4/5] 检查存储态...")
        storage_state = self._check_storage_state(
            kb_id, collection_name, kb_config, storage_monitoring_config
        )
        self._log_storage_state(storage_state)

        logger.info("\n[Step 5/5] 生成存储计划...")
        file_processing_plans = self._generate_file_processing_plans(source_stats)
        database_processing_plans = self._generate_database_processing_plans(
            source_stats
        )

        storage_plan = StoragePlan(
            kb_id=kb_id,
            kb_name=kb_name,
            source_stats=source_stats,
            storage_state=storage_state,
            file_processing_plans=file_processing_plans,
            database_processing_plans=database_processing_plans,
            kb_config=kb_config,
            storage_monitoring_config=storage_monitoring_config,
        )

        self._log_storage_plan(storage_plan)

        logger.info("\n" + "=" * 80)
        logger.info("配置分析和存储初始化分析完成!")
        logger.info("=" * 80)

        return storage_plan

    def _collect_source_statistics(
        self, sources: list[dict[str, Any]]
    ) -> SourceStatistics:
        """Collect source statistics.

        Args:
            sources: Data source configuration list

        Returns:
            Source statistics
        """
        stats = SourceStatistics()

        file_type_count = defaultdict(int)

        for source in sources:
            source_type = source.get("source_type", "")
            source_identifier = source.get("source_identifier", "")
            config = source.get("config", {})

            if source_type == "minio_file":
                stats.file_count += 1

                file_type = config.get("file_type", "txt").lower()
                file_type_count[file_type] += 1

                stats.files.append(
                    {
                        "source_identifier": source_identifier,
                        "file_type": file_type,
                        "config": config,
                        "etag": source.get("source_etag"),
                        "status": source.get("status", "pending"),
                    }
                )

            elif source_type == "qa_file":
                stats.qa_file_count += 1

                stats.qa_files.append(
                    {
                        "source_identifier": source_identifier,
                        "config": config,
                        "etag": source.get("source_etag"),
                        "status": source.get("status", "pending"),
                    }
                )

            elif source_type == "database":
                stats.db_connections += 1

                stats.databases.append(
                    {
                        "source_identifier": source_identifier,
                        "db_type": config.get("db_type", "unknown"),
                        "table_name": config.get("table_name", ""),
                        "config": config,
                        "status": source.get("status", "pending"),
                    }
                )

        stats.file_type_count = dict(file_type_count)

        return stats

    def _load_kb_config(self) -> dict[str, Any]:
        """Load basic KB configuration.

        Returns:
            Configuration dictionary
        """
        try:
            config_path = Path(self.kb_config_path)
            if not config_path.exists():
                logger.warning(f"KB config file not found: {self.kb_config_path}")
                return {}

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info(f"  ✓ 成功读取配置: {self.kb_config_path}")
            logger.info(f"    - KB名称: {config.get('name', 'N/A')}")
            logger.info(f"    - 描述: {config.get('description', 'N/A')}")

            embedding_config = config.get("embedding", {})
            logger.info(
                f"    - Embedding: {embedding_config.get('model', 'N/A')} "
                f"({embedding_config.get('type', 'N/A')})"
            )

            chunking_config = config.get("chunking", {})
            logger.info(
                f"    - Chunking: {chunking_config.get('strategy', 'N/A')} "
                f"(size: {chunking_config.get('chunk_size', 'N/A')})"
            )

            return config

        except Exception as e:
            logger.error(f"Failed to load KB config: {e}")
            return {}

    def _load_storage_monitoring_config(self) -> dict[str, Any]:
        """Load storage monitoring configuration.

        Returns:
            Configuration dictionary
        """
        try:
            config_path = Path(self.storage_monitoring_config_path)
            if not config_path.exists():
                logger.warning(
                    f"Storage monitoring config not found: {self.storage_monitoring_config_path}"
                )
                return {}

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            logger.info(
                f"  ✓ 成功读取配置: {self.storage_monitoring_config_path}"
            )

            vector_config = config.get("vector_store", {})
            logger.info(f"    - Vector Store: {vector_config.get('backend', 'N/A')}")

            relational_config = config.get("relational_database", {})
            logger.info(
                f"    - Relational DB: {relational_config.get('backend', 'N/A')}"
            )

            object_config = config.get("object_storage", {})
            logger.info(
                f"    - Object Storage: {object_config.get('backend', 'N/A')}"
            )

            return config

        except Exception as e:
            logger.error(f"Failed to load storage monitoring config: {e}")
            return {}

    def _check_storage_state(
        self,
        kb_id: int,
        collection_name: str | None,
        kb_config: dict[str, Any],
        storage_monitoring_config: dict[str, Any],
    ) -> StorageState:
        """Check storage state.

        Args:
            kb_id: knowledge base ID
            collection_name: collection name
            kb_config: KB configuration
            storage_monitoring_config: storage monitoring configuration

        Returns:
            StorageState: storage state
        """
        vector_config = storage_monitoring_config.get("vector_store", {})
        relational_config = storage_monitoring_config.get("relational_database", {})
        object_config = storage_monitoring_config.get("object_storage", {})

        state = StorageState(
            vector_store_type=vector_config.get("backend", "chroma"),
            vector_persist_dir=vector_config.get(
                "persist_directory", "./rag_data/vector_store"
            ),
            relational_db_type=relational_config.get("backend", "sqlite"),
            relational_db_path=relational_config.get("sqlite", {}).get(
                "db_path", "./rag_data/relational_database/rag_demo.sqlite"
            ),
            object_storage_enabled=True,
            object_storage_type=object_config.get("backend", "minio"),
            minio_endpoint=object_config.get("minio", {}).get(
                "endpoint", os.getenv("MINIO_ENDPOINT", "localhost:9000")
            ),
            minio_bucket=object_config.get("minio", {}).get(
                "bucket_name", os.getenv("MINIO_BUCKET", "rag-documents")
            ),
        )

        state.vector_collection_name = collection_name or f"kb_{kb_id}"

        vector_dir = Path(state.vector_persist_dir)
        if vector_dir.exists():
            state.vector_store_exists = True
            logger.info(f"    ✓ Vector Store 目录存在: {vector_dir}")
        else:
            state.vector_store_exists = False
            logger.info(f"    ✗ Vector Store 目录不存在: {vector_dir}")

        db_path = Path(state.relational_db_path)
        if db_path.exists():
            state.relational_db_exists = True
            logger.info(f"    ✓ Relational DB 文件存在: {db_path}")
        else:
            state.relational_db_exists = False
            logger.info(f"    ✗ Relational DB 文件不存在: {db_path}")

        return state

    def _generate_file_processing_plans(
        self, source_stats: SourceStatistics
    ) -> list[FileProcessingPlan]:
        """Generate file processing plans.

        Args:
            source_stats: source statistics

        Returns:
            A list of processing plans
        """
        plans = []

        processor_map = {
            "pdf": ("PDFProcessor", ["ChromaDB"]),
            "docx": ("WordProcessor", ["ChromaDB"]),
            "doc": ("WordProcessor", ["ChromaDB"]),
            "txt": ("TextProcessor", ["ChromaDB"]),
            "md": ("TextProcessor", ["ChromaDB"]),
            "xlsx": ("ExcelProcessor", ["ChromaDB", "SQLite"]),
            "xls": ("ExcelProcessor", ["ChromaDB", "SQLite"]),
            "csv": ("ExcelProcessor", ["ChromaDB", "SQLite"]),
        }

        for file_type, count in source_stats.file_type_count.items():
            if file_type in processor_map:
                processor, storage_targets = processor_map[file_type]

                plan = FileProcessingPlan(
                    file_type=file_type,
                    file_count=count,
                    processor=processor,
                    storage_targets=storage_targets,
                    metadata_sync=True,  # 默认同步MinIO元数据
                    metadata_fields=["source", "file_type", "object_name", "etag"],
                )

                plans.append(plan)

        return plans

    def _generate_database_processing_plans(
        self, source_stats: SourceStatistics
    ) -> list[DatabaseProcessingPlan]:
        """Generate database processing plans.

        Args:
            source_stats: source statistics

        Returns:
            Processing plan list
        """
        plans = []

        db_groups = defaultdict(lambda: {"tables": [], "config": None})

        for db in source_stats.databases:
            conn_str = db.get("source_identifier", "")
            table_name = db.get("table_name", "")
            db_type = db.get("db_type", "unknown")
            config = db.get("config", {})

            db_groups[conn_str]["tables"].append(table_name)
            db_groups[conn_str]["config"] = config
            db_groups[conn_str]["db_type"] = db_type

        for conn_str, group_info in db_groups.items():
            plan = DatabaseProcessingPlan(
                db_type=group_info["db_type"],
                connection_string=conn_str,
                table_count=len(group_info["tables"]),
                tables=group_info["tables"],
                processor="DatabaseProcessor",
                storage_targets=["SQLite (mirror)", "ChromaDB (metadata)"],
            )

            plans.append(plan)

        return plans

    def _log_source_statistics(self, stats: SourceStatistics):
        logger.info(f"  数据源统计:")
        logger.info(f"    - 文件数量: {stats.file_count}")
        logger.info(f"    - QA文件数量: {stats.qa_file_count}")
        logger.info(f"    - 数据库连接数量: {stats.db_connections}")
        logger.info(f"\n  按文件类型统计:")
        for file_type, count in sorted(stats.file_type_count.items()):
            logger.info(f"    - {file_type.upper()}: {count} 个")

    def _log_storage_state(self, state: StorageState):
        logger.info(f"  存储态检查:")
        logger.info(
            f"    - Vector Store: {state.vector_store_type} "
            f"({'存在' if state.vector_store_exists else '不存在'})"
        )
        logger.info(f"      → Collection: {state.vector_collection_name}")
        logger.info(f"      → 路径: {state.vector_persist_dir}")

        logger.info(
            f"    - Relational DB: {state.relational_db_type} "
            f"({'存在' if state.relational_db_exists else '不存在'})"
        )
        logger.info(f"      → 路径: {state.relational_db_path}")

        logger.info(
            f"    - Object Storage: {state.object_storage_type} "
            f"({'启用' if state.object_storage_enabled else '禁用'})"
        )
        if state.object_storage_enabled:
            logger.info(f"      → Endpoint: {state.minio_endpoint}")
            logger.info(f"      → Bucket: {state.minio_bucket}")

    def _log_storage_plan(self, plan: StoragePlan):
        logger.info("\n" + "=" * 80)
        logger.info("存储计划 (Storage Plan)")
        logger.info("=" * 80)

        logger.info(f"\n知识库信息:")
        logger.info(f"  - ID: {plan.kb_id}")
        logger.info(f"  - 名称: {plan.kb_name}")

        logger.info(f"\n文件处理计划:")
        if plan.file_processing_plans:
            for file_plan in plan.file_processing_plans:
                logger.info(
                    f"  [{file_plan.file_type.upper()}] x{file_plan.file_count}:"
                )
                logger.info(f"    → 处理器: {file_plan.processor}")
                logger.info(
                    f"    → 存储目标: {', '.join(file_plan.storage_targets)}"
                )
                if file_plan.metadata_sync:
                    logger.info(
                        f"    → 元数据同步: MinIO → ChromaDB ({', '.join(file_plan.metadata_fields)})"
                    )
        else:
            logger.info("  (无)")

        logger.info(f"\n数据库处理计划:")
        if plan.database_processing_plans:
            for db_plan in plan.database_processing_plans:
                logger.info(f"  [{db_plan.db_type.upper()}] {db_plan.table_count} 表:")
                logger.info(f"    → 表: {', '.join(db_plan.tables)}")
                logger.info(f"    → 处理器: {db_plan.processor}")
                logger.info(
                    f"    → 存储目标: {', '.join(db_plan.storage_targets)}"
                )
        else:
            logger.info("  (无)")

        logger.info("\n" + "=" * 80)
