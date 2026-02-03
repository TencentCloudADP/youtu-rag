"""Knowledge Builder Agent - Research-style architecture for parallel processing.

This agent handles the complete knowledge base building pipeline:
1. Parse input sources (MinIO files, database connections, QA files)
2. Process sources in parallel (using asyncio)
3. Validate with QA pairs if provided
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ...agents.simple_agent import SimpleAgent
from ...config import AgentConfig
from ...utils import get_logger
from ..api.minio_client import MinIOClient
from ..storage.base_storage import BaseVectorStore
from .config_analyzer import KnowledgeBuilderAnalyzer, StoragePlan
from .processors import DatabaseProcessor, FileProcessorFactory, ProcessResult, QAProcessor

logger = get_logger(__name__)


# ==================== Pydantic Models ====================


class ToolsConfig(BaseModel):
    """Retrieval tool configuration.

    It includes configuration for semantic retrieval and Text2SQL.
    """

    # Configuration for semantic retrieval
    semantic_retrieval: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_type: str = "api"  # "local" or "api"
    embedding_base_url: str | None = None  # Base URL for embedding service
    reranker_model: str | None = "jina-reranker-v2"

    # Configuration for Text2SQL
    text2sql_enabled: bool = True
    sql_generator_model: str = "gpt-4"


class StorageConfig(BaseModel):
    """Storage configuration.
    
    It includes configuration for vector database (ChromaDB), relational database (SQLite), and object storage (MinIO).
    """

    # Configuration for vector database (a shared ChromaDB instance; KBs are distinguished by collections)
    vector_store_type: str = "chroma"
    vector_persist_dir: str = "rag_data/vector_store"

    # Configuration for relational database (a shared SQLite file; KVs are distinguished by kb_id)
    # Uses the same database as the API for unified management and monitoring.
    relational_db_path: str = "rag_data/relational_database/rag_demo.sqlite"

    # Configuration for object storage (MinIO)
    minio_enabled: bool = True
    minio_bucket: str = "knowledge-base"


class BuildProgress(BaseModel):
    """Build progress information."""

    kb_id: int
    total_sources: int
    completed: int
    failed: int
    skipped: int
    current_source: str | None = None
    progress_percent: float
    latest_result: ProcessResult | None = None  # Latest processing result, used for updating database status

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SourceConfig(BaseModel):
    """Configuration for a single source."""

    source_type: Literal["minio_file", "database", "qa_file"]
    source_identifier: str
    config: dict[str, Any] = {}
    source_etag: str | None = None  # file content hash for incremental build
    metadata_hash: str | None = None  # Metadata hash for detecting metadata-only changes
    derived_files_hash: str | None = None  # Hash of derived files (OCR/Chunk) for change detection
    status: str | None = None  # Status from database ("completed"/"failed"/"pending")


class BuildRequest(BaseModel):
    """Request for building a knowledge base."""

    knowledge_base_id: int
    kb_name: str
    collection_name: str
    sources: list[SourceConfig]

    tools_config: ToolsConfig = ToolsConfig()

    storage_config: StorageConfig = StorageConfig()

    force_rebuild: bool = False
    progress_callback: Callable[[BuildProgress], None] | None = None  # Progress callback

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BuildReport(BaseModel):
    """Report for building a knowledge base."""

    kb_id: int
    kb_name: str
    status: Literal["completed", "failed", "partial"]
    total_sources: int
    successful: int
    failed: int
    skipped: int = 0  # Number of skipped sources (unchanged)
    total_chunks: int
    total_tables: int
    duration_seconds: float
    errors: list[str] = []
    qa_validation: dict[str, Any] | None = None
    results: list["ProcessResult"] = []  # Detailed results for each source


class ProcessTask(BaseModel):
    """Task for processing a single source."""

    source_config: SourceConfig
    task_id: str


# ==================== Main Agent ====================


class KnowledgeBuilderAgent(SimpleAgent):

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        relational_db_path: str | None = None,
        config: AgentConfig | str | None = None,
        minio_client: MinIOClient | None = None,
    ):
        """Initialize KnowledgeBuilderAgent.
        
        Args:
            vector_store: A vector database instance
            relational_db_path: Path to the relational database, used to store structured tables
            config: Agent configuration (use either a custom config or the default "rag/knowledge_builder/knowledge_builder")
            minio_client: MinIO client, uesd for ETag checks and incremental builds
        """
        logger.info("🔧 KnowledgeBuilderAgent.__init__() called - START")

        # Initialize SimpleAgent with config
        super().__init__(config=config or "ragref/knowledge_builder/knowledge_builder")
        logger.info("🔧 SimpleAgent.__init__() completed")
        self.vector_store = vector_store
        self.relational_db_path = relational_db_path or "rag_data/relational_database/rag_demo.sqlite"
        self.minio_client = minio_client

        self.column_value_vectorization_strategy = "individual"
        self.column_value_top_n = 8
        self.batch_delay = 3.0
        self.batch_size = 50
        self.chunk_size = 500  # Default chunk size
        self.chunk_overlap = 50  # Default chunk overlap 

        import os
        default_embedding_url = os.getenv("UTU_EMBEDDING_URL")
        default_embedding_api_key = os.getenv("UTU_EMBEDDING_API_KEY")
        if default_embedding_url and not default_embedding_api_key:
            default_embedding_type = "local"
        else:
            default_embedding_type = "api"

        self.file_processor_factory = FileProcessorFactory(
            vector_store=self.vector_store,
            relational_db_path=self.relational_db_path,
            minio_client=self.minio_client,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            column_value_vectorization_strategy=self.column_value_vectorization_strategy,
            column_value_top_n=self.column_value_top_n,
            batch_delay=self.batch_delay,
            batch_size=self.batch_size,
            embedding_type=default_embedding_type,
            embedding_base_url=default_embedding_url
        )

        self.db_processor = DatabaseProcessor(
            vector_store=self.vector_store,
            relational_db_path=self.relational_db_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            column_value_vectorization_strategy=self.column_value_vectorization_strategy,
            column_value_top_n=self.column_value_top_n,
            batch_delay=self.batch_delay,
            embedding_type=default_embedding_type,
            embedding_base_url=default_embedding_url
        )

        self.qa_processor = None

        # The configuration analyzer, used in Stage 0
        self.config_analyzer = KnowledgeBuilderAnalyzer()
        # Storage plan (generated in Stage 0)
        self.storage_plan: StoragePlan | None = None

        logger.info(f"✅ KnowledgeBuilderAgent initialized with config: {self.config.agent.name}")
        logger.info("🔧 KnowledgeBuilderAgent.__init__() called - END")

    async def build(self, request: BuildRequest) -> BuildReport:
        """The main build process.

        It includes four stages:
        0. Configuration analysis and storage initialization;
        1. Grouping tasks by file type;
        2. Parallel processing of groups;
        3. Processing QA associations.

        Args:
            request: A build request

        Returns:
            BuildReport: The build report

        Note:
            DO NOT use @trace decorator! It's a data processing workflow.
            Otherwise, OpenAIInstrumentor will automatically trace it to Phoenix if LLM calls are added.
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting build for KB {request.kb_name} (ID: {request.knowledge_base_id})")

            self._load_column_value_config(request.kb_name)

            # ========== Stage 0: Configuration analysis and storage initialization ==========
            await self._analyze_and_init_config(request)

            # ========== Stage 1: Grouping tasks by file type ==========
            task_groups = await self._create_task_groups(request)

            # ========== Stage 2: Parallel processing of groups ==========
            all_results = []

            data_tasks = task_groups.get('text', []) + task_groups.get('table', [])

            if data_tasks:
                logger.info(f"Processing {len(data_tasks)} data source tasks...")
                results = await self._process_parallel(
                    data_tasks,
                    kb_id=request.knowledge_base_id,
                    progress_callback=request.progress_callback
                )
                all_results.extend(results)

            # ========== Stage 3: Processing QA associations ==========
            qa_result = None
            qa_result_dict = None
            if 'qa' in task_groups and task_groups['qa']:
                logger.info(f"Processing QA associations...")
                qa_result = await self._process_qa_associations(
                    task_groups['qa'][0],
                    request
                )
                if qa_result:
                    all_results.append(qa_result)
                    # Convert ProcessResult to dict for BuildReport
                    qa_result_dict = {
                        "status": qa_result.status,
                        "source": qa_result.source_identifier,
                        "chunks_created": qa_result.chunks_created,
                        "tables_created": qa_result.tables_created,
                        "error_message": qa_result.error_message
                    }

            report = self._generate_report(
                request, all_results, qa_result_dict, start_time, datetime.now()
            )

            logger.info(f"Build completed: {report.status}")

            return report

        except Exception as e:
            logger.error(f"Build failed: {str(e)}", exc_info=True)
            duration = (datetime.now() - start_time).total_seconds()

            return BuildReport(
                kb_id=request.knowledge_base_id,
                kb_name=request.kb_name,
                status="failed",
                total_sources=len(request.sources),
                successful=0,
                failed=len(request.sources),
                total_chunks=0,
                total_tables=0,
                duration_seconds=duration,
                errors=[str(e)],
            )

    def _set_fallback_defaults(self):
        """Set fallback default values for configuration parameters."""
        self.column_value_vectorization_strategy = "individual"
        self.column_value_top_n = 8
        self.batch_delay = 3.0
        self.batch_size = 50
        self.chunk_size = 500
        self.chunk_overlap = 50

    def _load_column_value_config(self, kb_name: str):
        """Load column value vectorization configuration.

        Load knowledge base specific configuration file (configs/rag/{kb_name}.yaml) in priority.
        Fallback to default.yaml if not found.

        Args:
            kb_name: knowledge base name
        """
        from pathlib import Path
        from omegaconf import OmegaConf

        current_dir = Path(__file__).parent.parent.parent.parent  # The root directory of the project
        rag_config_dir = current_dir / "configs" / "rag"

        config_name = kb_name
        config_path_to_try = rag_config_dir / f"{kb_name}.yaml"

        logger.info(f"🔍 Attempting to load config for KB: {kb_name}")
        logger.info(f"🔍 Checking path: {config_path_to_try}")

        # Fallback to default.yaml if KB-specific config not found
        if not config_path_to_try.exists():
            config_name = "default"
            config_path_to_try = rag_config_dir / "default.yaml"
            logger.info(f"📄 KB-specific config not found, using default.yaml")
            logger.info(f"📄 Loading from: {config_path_to_try}")
        else:
            logger.info(f"📄 Found KB-specific config: {config_path_to_try}")

        # Check if config file exists.
        # Enter the branch only if neither KB-specific nor default config exists.
        if not config_path_to_try.exists():
            logger.error(f"❌ Config file not found: {config_path_to_try}")
            self._set_fallback_defaults()
            logger.warning(
                f"⚠️ Using fallback defaults: strategy={self.column_value_vectorization_strategy}, "
                f"top_n={self.column_value_top_n}, batch_delay={self.batch_delay}s, batch_size={self.batch_size}, "
                f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
            )
            return

        try:
            logger.info(f"📥 Loading config: {config_name}.yaml from {rag_config_dir}/")
            # Load config using OmegaConf -- avoid parsing all env vars
            cfg = OmegaConf.load(config_path_to_try)

            # Only parse text2sql config
            text2sql_config = cfg.get("text2sql", {})
            logger.debug(f"🔍 text2sql_config: {text2sql_config}")

            # column_value_config does not contain environment variables
            if text2sql_config:
                column_value_config = text2sql_config.get("column_value_vectorization", {})
            else:
                column_value_config = {}

            logger.debug(f"🔍 column_value_config: {column_value_config}")

            self.column_value_vectorization_strategy = column_value_config.get("strategy", "individual")
            self.column_value_top_n = column_value_config.get("top_n", 8)

            knowledge_builder_config = cfg.get("knowledge_builder", {})
            self.batch_delay = knowledge_builder_config.get("batch_delay", 3.0)

            embedding_config = cfg.get("embedding", {})
            self.batch_size = embedding_config.get("batch_size", 50)

            chunking_config = cfg.get("chunking", {})
            self.chunk_size = chunking_config.get("chunk_size", 500)
            self.chunk_overlap = chunking_config.get("chunk_overlap", 50)

            logger.info(
                f"✅ Loaded column value vectorization config from '{config_name}.yaml': "
                f"strategy={self.column_value_vectorization_strategy}, top_n={self.column_value_top_n}"
            )
            logger.info(f"✅ Loaded batch_delay from '{config_name}.yaml': {self.batch_delay}s")
            logger.info(f"✅ Loaded batch_size from '{config_name}.yaml': {self.batch_size}")
            logger.info(f"✅ Loaded chunking config from '{config_name}.yaml': chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to load config from '{config_name}.yaml': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._set_fallback_defaults()
            logger.warning(
                f"⚠️ Using fallback defaults: strategy={self.column_value_vectorization_strategy}, "
                f"top_n={self.column_value_top_n}, batch_delay={self.batch_delay}s, batch_size={self.batch_size}, "
                f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
            )

    def _validate_config(self, request: BuildRequest):
        """Validate the configuration of a request.

        It validates the necessary fields (knowledge_base_id, kb_name, sources) and the tools configuration.

        Args:
            request: The build request to check

        Raises:
            ValueError: Raised if the configuration is invalid
        """
        if not request.knowledge_base_id:
            raise ValueError("knowledge_base_id is required")

        if not request.kb_name:
            raise ValueError("kb_name is required")

        if not request.sources:
            raise ValueError("At least one source is required")

        tools_config = request.tools_config

        if tools_config.semantic_retrieval and not tools_config.embedding_model:
            raise ValueError("embedding_model is required when semantic_retrieval is enabled")

        if tools_config.text2sql_enabled and not tools_config.sql_generator_model:
            raise ValueError("sql_generator_model is required when text2sql is enabled")

        logger.info("✓ Configuration validation passed")

    def _check_storage_exists(self, request: BuildRequest) -> tuple[bool, bool]:
        """Check whether the planned storage to build from the request already exists.

        This method checks whether the vector database and the knowledge base already exist, respectively.

        Args:
            request: A build request

        Returns:
            A tuple of booleans (vector_exists, kb_has_data), indicating whether the vector database exists and whether the knowledge base has data
        """
        from pathlib import Path

        # Check whether the vector database (ChromaDB) exists
        vector_exists = False
        try:
            collection_name = request.collection_name
            from ..storage import VectorStoreFactory

            chroma_dir = Path(request.storage_config.vector_persist_dir)
            if chroma_dir.exists():
                try:
                    _ = VectorStoreFactory.create_chroma(
                        collection_name=collection_name,
                        persist_directory=request.storage_config.vector_persist_dir
                    )
                    # If connection is successful, the ChromaDB collection exists
                    vector_exists = True
                    logger.info(f"✓ Found existing vector store collection: {collection_name}")
                except Exception as e:
                    logger.debug(f"Vector store collection not found or empty: {e}")
                    vector_exists = False
        except Exception as e:
            logger.debug(f"Error checking vector store: {e}")
            vector_exists = False

        # Check whether the relational database (SQLite) has data for this KB
        kb_has_data = False
        db_path = Path(request.storage_config.relational_db_path)
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                # Check QA association table exists and has data for the requested knowledge base ID
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='qa_associations'"
                )
                if cursor.fetchone():
                    cursor.execute(
                        "SELECT COUNT(*) FROM qa_associations WHERE kb_id = ?",
                        (request.knowledge_base_id,)
                    )
                    count = cursor.fetchone()[0]
                    if count > 0:
                        kb_has_data = True
                        logger.info(f"✓ Found {count} QA associations for KB {request.knowledge_base_id}")

                conn.close()
            except Exception as e:
                logger.debug(f"Error checking KB data in database: {e}")
                kb_has_data = False

        return vector_exists, kb_has_data

    async def _analyze_and_init_config(self, request: BuildRequest):
        """Analyze configuration and initialize storage.

        It is Stage 0 of the build process.
        
        Steps:
        0. Use KnowledgeBuilderAnalyzer to analyze configuration
        1. Validate configuration
        2. Check if storage exists
        3. Determine if incremental build or full build based on storage status
        4. Initialize vector store (KBs are distinguished by their collections)
        5. Initialize relational database (use a shared SQLite; KBs are distinguished by kb_id)
        6. Update processor storage configuration

        Args:
            request: The build request to analyze
        """
        logger.info("========== Stage 0: Configuration Analysis and Storage Initialization ==========")

        # 0. Use KnowledgeBuilderAnalyzer to analyze configuration
        logger.info("Step 0: Running Knowledge Builder Analyzer...")
        sources_dict = [
            {
                "source_type": s.source_type,
                "source_identifier": s.source_identifier,
                "config": s.config,
                "source_etag": s.source_etag,
                "status": s.status,
            }
            for s in request.sources
        ]

        self.storage_plan = self.config_analyzer.analyze(
            kb_id=request.knowledge_base_id,
            kb_name=request.kb_name,
            sources=sources_dict,
            collection_name=request.collection_name,
        )

        # 1. Validate configuration
        logger.info("\nStep 1: Validating configuration...")
        self._validate_config(request)

        # 2. Check if storage exists
        logger.info("Step 2: Checking storage existence...")
        vector_exists, kb_has_data = self._check_storage_exists(request)

        # 3. Determine if incremental build or full build
        is_incremental = (vector_exists or kb_has_data) and not request.force_rebuild

        if is_incremental:
            logger.info("✓ KB data exists in shared storage - INCREMENTAL BUILD mode")
            logger.info("  → Existing data will be preserved")
            logger.info("  → Only changed/new sources will be processed")
        else:
            if request.force_rebuild:
                logger.info("⚠ FORCE REBUILD mode - All KB data will be reprocessed")
            else:
                logger.info("✓ New knowledge base - FULL BUILD mode")

        # 3. Analyze tools configuration
        logger.info("Step 3: Analyzing tools configuration...")
        tools_config = request.tools_config
        logger.info(
            f"Tools Configuration:\n"
            f"  - Semantic Retrieval: {tools_config.semantic_retrieval} "
            f"(model: {tools_config.embedding_model})\n"
            f"  - Reranker: {tools_config.reranker_model}\n"
            f"  - Text2SQL: {tools_config.text2sql_enabled} "
            f"(model: {tools_config.sql_generator_model})"
        )

        # 4. Initialize vector store
        logger.info("Step 4: Initializing vector store...")
        if self.vector_store is None:
            # Use the collection_name from the request (from the database) instead of regenerating.
            # This ensures consistency with the collection_name in the database.
            collection_name = request.collection_name

            from ..config import VectorStoreConfig
            from ..storage import VectorStoreFactory

            vector_config = VectorStoreConfig(
                backend="chroma",
                collection_name=collection_name,
                persist_directory=request.storage_config.vector_persist_dir
            )
            self.vector_store = VectorStoreFactory.create(vector_config)

            if is_incremental:
                logger.info(f"✓ Connected to existing collection: {collection_name}")
            else:
                logger.info(f"✓ Created new collection: {collection_name}")

        # 5. Initialize relational database
        logger.info("Step 5: Initializing relational database...")
        from pathlib import Path
        db_path = Path(request.storage_config.relational_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.relational_db_path = str(db_path)

        if db_path.exists():
            logger.info(f"✓ Connected to shared database: {self.relational_db_path}")
            if kb_has_data:
                logger.info(f"  → KB {request.knowledge_base_id} has existing data")
        else:
            logger.info(f"✓ Created shared database: {self.relational_db_path}")

        # 6. Update processor configurations
        logger.info("Step 6: Updating processor configurations...")
        logger.info(
            f"  Column value vectorization config: "
            f"strategy={self.column_value_vectorization_strategy}, "
            f"top_n={self.column_value_top_n}, "
            f"batch_delay={self.batch_delay}s, "
            f"batch_size={self.batch_size}"
        )
        logger.info(
            f"  Chunking config: "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
        self.file_processor_factory = FileProcessorFactory(
            vector_store=self.vector_store,
            relational_db_path=self.relational_db_path,
            minio_client=self.minio_client,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            column_value_vectorization_strategy=self.column_value_vectorization_strategy,
            column_value_top_n=self.column_value_top_n,
            batch_delay=self.batch_delay,
            batch_size=self.batch_size,
            embedding_type=request.tools_config.embedding_type,
            embedding_base_url=request.tools_config.embedding_base_url
        )
        self.db_processor = DatabaseProcessor(
            vector_store=self.vector_store,
            relational_db_path=self.relational_db_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            column_value_vectorization_strategy=self.column_value_vectorization_strategy,
            column_value_top_n=self.column_value_top_n,
            batch_delay=self.batch_delay,
            embedding_type=request.tools_config.embedding_type,
            embedding_base_url=request.tools_config.embedding_base_url
        )

        logger.info("========== Stage 0 Completed ==========")
        logger.info(f"Build mode: {'INCREMENTAL' if is_incremental else 'FULL'}")
        logger.info(f"Vector store: {request.storage_config.vector_persist_dir}")
        logger.info(f"Relational DB: {self.relational_db_path}")
        logger.info("=" * 60)

    async def _create_task_groups(self, request: BuildRequest) -> dict[str, list[ProcessTask]]:
        """Group tasks by file type.

        It is Stage 1 of the build process. Group tasks by the following types and log the number of tasks for each type and each group:
        - text: pdf, docx, txt, md
        - table: excel, database
        - qa: qa_file

        Args:
            request: The build request.

        Returns:
            A dictionary of grouped tasks (dict[str, list[ProcessTask]]).
        """
        logger.info("========== Stage 1: Task Grouping ==========")

        groups = {
            'text': [],   # text files
            'table': [],  # tables and databases
            'qa': []      # QA files
        }

        from collections import defaultdict
        file_type_stats = defaultdict(int)

        for idx, source in enumerate(request.sources):
            task = ProcessTask(
                source_config=source,
                task_id=f"{request.knowledge_base_id}_{idx}"
            )

            if source.source_type == 'qa_file':
                groups['qa'].append(task)
                file_type_stats['qa_file'] += 1
            elif source.source_type == 'database':
                groups['table'].append(task)
                file_type_stats['database'] += 1
            elif source.source_type == 'minio_file':
                file_type = source.config.get('file_type', 'txt').lower()
                file_type_stats[file_type] += 1

                if file_type in ['pdf', 'docx', 'doc', 'txt', 'md']:
                    groups['text'].append(task)
                elif file_type in ['xlsx', 'xls', 'csv', 'excel']:
                    groups['table'].append(task)
                else:  # Default to text
                    logger.warning(f"Unknown file type: {file_type}, treating as text")
                    groups['text'].append(task)

        # Log file type statistics
        logger.info("\n按文件类型统计:")

        text_types = ['pdf', 'docx', 'doc', 'txt', 'md']
        text_count = sum(file_type_stats[ft] for ft in text_types if ft in file_type_stats)
        if text_count > 0:
            logger.info(f"  📄 文本文件: {text_count} 个")
            for ft in text_types:
                if file_type_stats[ft] > 0:
                    logger.info(f"    - {ft.upper()}: {file_type_stats[ft]} 个")

        table_types = ['xlsx', 'xls', 'csv', 'excel']
        table_count = sum(file_type_stats[ft] for ft in table_types if ft in file_type_stats)
        if table_count > 0:
            logger.info(f"  📊 表格文件: {table_count} 个")
            for ft in table_types:
                if file_type_stats[ft] > 0:
                    logger.info(f"    - {ft.upper()}: {file_type_stats[ft]} 个")

        if file_type_stats['database'] > 0:
            logger.info(f"  🗄️  数据库: {file_type_stats['database']} 个")

        if file_type_stats['qa_file'] > 0:
            logger.info(f"  ❓ QA文件: {file_type_stats['qa_file']} 个")

        # Log task group statistics
        logger.info("\n按任务组统计:")
        for group_name, tasks in groups.items():
            if tasks:
                logger.info(f"  Group '{group_name}': {len(tasks)} tasks")

        total_tasks = sum(len(tasks) for tasks in groups.values())
        logger.info(f"\n✓ Stage 1 completed: {total_tasks} tasks grouped")

        return groups

    async def _process_parallel(
        self,
        tasks: list[ProcessTask],
        kb_id: int | None = None,
        progress_callback: Callable[[BuildProgress], None] | None = None
    ) -> list[ProcessResult]:
        """Parallel processing of tasks.

        It is Stage 2 of the build process, which is the major advantage of this project.
        It supports real-time progress tracking by calling progress_callback after each
        task is completed.

        Args:
            tasks: List of tasks to process
            kb_id: Knowledge base ID used for progress callback
            progress_callback: Progress callback function

        Returns:
            List of processing results (list[ProcessResult]).
        """
        if not tasks:
            return []

        logger.info(f"Starting parallel processing of {len(tasks)} tasks")

        async_tasks = [asyncio.create_task(self._process_single(task)) for task in tasks]

        # Real-time progress metrics
        results = []
        completed_count = 0
        failed_count = 0
        skipped_count = 0

        for coro in asyncio.as_completed(async_tasks):
            try:
                result = await coro
                results.append(result)

                if result.status == "completed":
                    completed_count += 1
                elif result.status == "failed":
                    failed_count += 1
                elif result.status == "skipped":
                    skipped_count += 1

                total_processed = completed_count + failed_count + skipped_count
                progress_percent = (total_processed / len(tasks)) * 100

                logger.info(
                    f"Progress: {total_processed}/{len(tasks)} ({progress_percent:.1f}%) - "
                    f"✅ {completed_count} ❌ {failed_count} ⚡ {skipped_count}"
                )

                # Progress callback, including the latest processing result
                if progress_callback and kb_id is not None:
                    try:
                        progress = BuildProgress(
                            kb_id=kb_id,
                            total_sources=len(tasks),
                            completed=completed_count,
                            failed=failed_count,
                            skipped=skipped_count,
                            current_source=result.source_identifier,
                            progress_percent=progress_percent,
                            latest_result=result  # The lastest processing result
                        )
                        progress_callback(progress)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

            except Exception as e:
                # Record failure
                logger.error(f"Task failed: {str(e)}")
                failed_count += 1
                results.append(
                    ProcessResult(
                        source_identifier="unknown",
                        source_type="unknown",
                        status="failed",
                        chunks_created=0,
                        tables_created=[],
                        error_message=str(e),
                    )
                )

        return results

    async def _process_single(self, task: ProcessTask) -> ProcessResult:
        """Process a single task, using factory-based routing.

        Supports incremental build: skip if the file hasn't changed and was successfully processed previously.

        Args:
            task: A single task

        Returns:
            The processing result (ProcessResult)
        """
        source = task.source_config

        try:
            # Check whether the file has changed only when it was successfully processed (marked completed) previously
            if (
                self.minio_client
                and source.source_type in ["minio_file", "qa_file"]
                and source.status == "completed"
            ):
                try:
                    # Comparing file ETag and metadata hash
                    stat = self.minio_client.get_file_stat(source.source_identifier)
                    current_etag = stat.etag.strip('"') if stat.etag else None

                    current_metadata = self.minio_client.get_file_metadata(source.source_identifier) or {}
                    current_metadata_hash = None
                    if current_metadata:
                        # We need to recompute the hash as MinIO only stores the original fields.
                        import hashlib
                        import json
                        metadata_json = json.dumps(current_metadata, sort_keys=True, ensure_ascii=False)
                        current_metadata_hash = hashlib.md5(metadata_json.encode('utf-8')).hexdigest()

                    etag_unchanged = source.source_etag and current_etag and current_etag == source.source_etag
                    metadata_unchanged = source.metadata_hash and current_metadata_hash and current_metadata_hash == source.metadata_hash

                    # Check if derived files have changed (for OCR/Chunk-processed files)
                    # It is considered as unchanged if no derived files are found or the hash of the derived file ETags is unchanged.
                    derived_files_unchanged = True
                    current_derived_etags = []

                    # Check chunk-processed files first
                    if current_metadata.get("chunk_processed") == "chunk_success":
                        import os
                        from pathlib import Path
                        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
                        chunked_filename = f"{Path(source.source_identifier).stem}_chunklevel.md"

                        try:
                            chunk_stat = self.minio_client.get_file_stat(chunked_filename, bucket_name=sys_bucket)
                            if chunk_stat and hasattr(chunk_stat, 'etag'):
                                current_derived_etags.append(chunk_stat.etag.strip('"'))
                        except Exception as e:
                            logger.debug(f"Chunk file {chunked_filename} not found or error: {e}")

                    # Check OCR-processed files (can coexist with chunk files)
                    if current_metadata.get("ocr_processed") == "ocr_success":
                        import os
                        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")

                        # Calculate current derived files hash
                        _, ocr_derived_etags = self.minio_client.load_derived_markdown_files(
                            source_filename=source.source_identifier,
                            sys_bucket=sys_bucket
                        )

                        if ocr_derived_etags:
                            current_derived_etags.extend(ocr_derived_etags)

                    # Calculate combined hash if we have any derived files
                    if current_derived_etags:
                        current_derived_hash = self.minio_client.calculate_derived_files_hash(current_derived_etags)
                        derived_files_unchanged = (
                            source.derived_files_hash and
                            current_derived_hash == source.derived_files_hash
                        )

                        if not derived_files_unchanged:
                            logger.info(
                                f"🔄 Derived files changed for {source.source_identifier}: "
                                f"{source.derived_files_hash[:8] if source.derived_files_hash else 'None'}... -> {current_derived_hash[:8]}..."
                            )

                    # Skip only if all three (ETag, metadata, derived files) are unchanged
                    if etag_unchanged and metadata_unchanged and derived_files_unchanged:
                        logger.info(
                            f"⚡ Skipping unchanged file: {source.source_identifier} "
                            f"(ETag: {current_etag[:8]}..., Metadata: {current_metadata_hash[:8]}...)"
                        )
                        return ProcessResult(
                            source_identifier=source.source_identifier,
                            source_type=source.source_type,
                            status="skipped",
                            chunks_created=0,
                            tables_created=[],
                            metadata={"reason": "unchanged", "etag": current_etag, "metadata_hash": current_metadata_hash},
                        )
                    else:
                        # 记录变化原因
                        change_reasons = []
                        if not etag_unchanged:
                            change_reasons.append(
                                f"ETag changed (Old: {source.source_etag[:8] if source.source_etag else 'N/A'}..., "
                                f"New: {current_etag[:8] if current_etag else 'N/A'}...)"
                            )
                        if not metadata_unchanged:
                            change_reasons.append(
                                f"Metadata changed (Old: {source.metadata_hash[:8] if source.metadata_hash else 'N/A'}..., "
                                f"New: {current_metadata_hash[:8] if current_metadata_hash else 'N/A'}...)"
                            )
                        if not derived_files_unchanged:
                            change_reasons.append("Derived files changed")

                        logger.info(
                            f"📝 File/metadata changed, will reprocess: {source.source_identifier} "
                            f"({'; '.join(change_reasons)})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to check ETag/metadata for {source.source_identifier}: {e}, will process anyway")

            # 正常处理流程
            if source.source_type == "minio_file":
                # 使用工厂模式选择处理器
                file_type = source.config.get("file_type", "txt")
                processor = self.file_processor_factory.create(file_type)

                return await processor.process(
                    source_identifier=source.source_identifier, config=source.config
                )

            elif source.source_type == "database":
                # 数据库处理
                return await self.db_processor.process(
                    source_identifier=source.source_identifier, config=source.config
                )

            else:
                raise ValueError(f"Unsupported source type: {source.source_type}")

        except Exception as e:
            logger.error(f"Process single task failed: {str(e)}", exc_info=True)
            return ProcessResult(
                source_identifier=source.source_identifier,
                source_type=source.source_type,
                status="failed",
                chunks_created=0,
                tables_created=[],
                error_message=str(e),
            )

    async def _process_qa_associations(
        self, qa_task: ProcessTask, request: BuildRequest
    ) -> ProcessResult | None:
        """
        Stage 3: QA 关联处理

        功能：
        1. 初始化 QAProcessor
        2. 处理 QA 文件，建立关联关系

        Args:
            qa_task: QA 任务
            request: 构建请求

        Returns:
            ProcessResult | None: 处理结果
        """
        logger.info("========== Stage 3: QA Association Processing ==========")

        try:
            # 初始化 QA 处理器
            self.qa_processor = QAProcessor(
                kb_id=request.knowledge_base_id,
                minio_client=self.minio_client,
                vector_store=self.vector_store,
                relational_db_path=self.relational_db_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                batch_delay=self.batch_delay,
                batch_size=self.batch_size,
                embedding_type=request.tools_config.embedding_type,
                embedding_base_url=request.tools_config.embedding_base_url
            )

            # 处理 QA 文件
            result = await self.qa_processor.process(
                source_identifier=qa_task.source_config.source_identifier,
                config=qa_task.source_config.config
            )

            logger.info(f"Stage 3 completed: {result.metadata.get('qa_pairs', 0)} QA pairs processed")
            return result

        except Exception as e:
            logger.error(f"QA association processing failed: {str(e)}", exc_info=True)
            return ProcessResult(
                source_identifier=qa_task.source_config.source_identifier,
                source_type="qa_file",
                status="failed",
                chunks_created=0,
                error_message=str(e)
            )

    def _generate_report(
        self,
        request: BuildRequest,
        results: list[ProcessResult],
        qa_result: dict | None,
        start_time: datetime,
        end_time: datetime,
    ) -> BuildReport:
        """
        生成构建报告

        Args:
            request: 构建请求
            results: 处理结果列表
            qa_result: QA验证结果
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            BuildReport: 构建报告
        """
        successful = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")
        total_chunks = sum(r.chunks_created for r in results)
        total_tables = sum(len(r.tables_created) for r in results)

        errors = [r.error_message for r in results if r.error_message]

        duration = (end_time - start_time).total_seconds()

        # 确定整体状态
        if failed == 0 and successful + skipped > 0:
            status = "completed"
        elif successful == 0 and skipped == 0:
            status = "failed"
        else:
            status = "partial"

        # Log incremental build stats
        if skipped > 0:
            logger.info(f"⚡ Incremental build: Skipped {skipped} unchanged sources, processed {successful + failed}")

        return BuildReport(
            kb_id=request.knowledge_base_id,
            kb_name=request.kb_name,
            status=status,
            total_sources=len(results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_chunks=total_chunks,
            total_tables=total_tables,
            duration_seconds=duration,
            errors=errors,
            qa_validation=qa_result,
            results=results,  # 包含每个源的详细结果
        )
