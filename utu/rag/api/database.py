"""Database models for RAG Knowledge Management.

⚠️ SQL Injection Prevention:
- All database operations use SQLAlchemy ORM with parameter binding
- Never concatenate user input into raw SQL queries
- For dynamic queries, always validate inputs against whitelists
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, ForeignKey, create_engine, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

DATABASE_URL = settings.database_url

if "sqlite" in DATABASE_URL:
    # Extract file path from SQLite URL (format: sqlite:///path/to/db.sqlite)
    db_path_str = DATABASE_URL.replace("sqlite:///", "")
    db_path = Path(db_path_str)
    db_dir = db_path.parent

    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created database directory: {db_dir}")

connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args = {
        "check_same_thread": False,
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    json_serializer=lambda obj: __import__('json').dumps(obj, ensure_ascii=False),
    json_deserializer=lambda obj: __import__('json').loads(obj)
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class KnowledgeBase(Base):
    """Knowledge base model."""

    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(String(1000))
    collection_name = Column(String(255), unique=True, nullable=False)  # For vector store
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class KBSourceConfig(Base):
    """
    Knowledge base data source configuration table.

    This configuration table manages MinIO files, database connections, and QA files uniformly.
    """

    __tablename__ = "kb_source_configs"
    __table_args__ = (
        # The unique constraint that prevents adding the same source (files, tables, etc.) multiple times to the same knowledge base.
        # Notice that the same source can be used in different knowledge bases.
        UniqueConstraint('knowledge_base_id', 'source_type', 'source_identifier', name='uq_kb_source'),
    )

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)

    source_type = Column(String(50), nullable=False)  # 'minio_file', 'database', 'qa_file'
    
    # Source identifier (full path, connection string, table name, etc.), so that different files with the same name can be uniquely identified.
    # ⚠️ Important: For MinIO files, use the full object path (including folders).
    # For example: 'folder1/document.pdf' instead of just 'document.pdf'.
    source_identifier = Column(String(500), nullable=False)

    # Extra configuration in JSON format.
    # For MinIO files: {"object_name": "folder1/xxx.pdf", "file_type": "pdf", "display_name": "xxx.pdf"}
    # For database tables: {"connection_string": "xxx", "table_name": "xxx", "db_type": "mysql"}
    # For QA files: {"object_name": "qa.xlsx
    config = Column(JSON)

    # Hash of the source content (ETag), used to detect content changes.
    # For MinIO files: the object's ETag (usually an MD5 hash).
    # For database tables: the table checksum or row count.
    # For QA files: the file's ETag.
    # ⚠️ Important: The ETag changes as the content changes, which is used for incremental build optimization.
    source_etag = Column(String(100))

    # Metadata hash, used to detect metadata changes.
    # It stores the MD5 hash of file metadata (char_length, publish_date, key_timepoints, summary, custom tags, etc.)
    # ⚠️ Important: Even if the file content (ETag) does not change, metadata changes will also trigger a rebuild.
    metadata_hash = Column(String(100))

    # OCR-derived files hash, used to detect derived markdown files changes.
    # It stores the joint MD5 hash of the Etags of all the derived markdown files.
    # ⚠️ Important: When the OCR-derived files change (including manual correction), it triggers an intelligent rebuild.
    derived_files_hash = Column(String(100))

    status = Column(String(50), default='pending')  # 'pending', 'processing', 'completed', 'failed'

    chunks_created = Column(Integer, default=0)
    tables_created = Column(Text)  # Table names, separated by commas

    error_message = Column(Text)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class KBBuildConfig(Base):
    """Knowledge base build configuration table, storing tool configurations."""

    __tablename__ = "kb_build_configs"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)

    # Tool configuration in JSON format. Example:
    # {
    #   "embedding": {"enabled": true, "settings": {"model": "text-embedding-3-small"}},
    #   "chunk": {"enabled": true, "settings": {"chunk_size": 500, "overlap": 50}},
    #   "reranker": {"enabled": false, "settings": {}}
    # }
    tools_config = Column(JSON)

    # Build options in JSON format. Example:
    # {"force_rebuild": false, "parallel_processing": true, "max_workers": 4}
    build_options = Column(JSON)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class KBBuildLog(Base):
    """Knowledge base build log table."""

    __tablename__ = "kb_build_logs"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)

    status = Column(String(50), nullable=False)  # 'pending', 'running', 'completed', 'failed'

    total_sources = Column(Integer, default=0)
    processed_sources = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    total_tables = Column(Integer, default=0)

    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)

    # Detailed result in JSON format. Example:
    # {"successful": 5, "failed": 1, "errors": ["xxx"], "qa_validation": {...}}
    result_detail = Column(JSON)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class KBExcelTable(Base):
    """Excel table mapping table, storing SQLite tables created from Excel files."""

    __tablename__ = "kb_excel_tables"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("kb_source_configs.id"), nullable=True, index=True)

    source_file = Column(String(500), nullable=False, comment="源Excel文件路径")
    sheet_name = Column(String(200), nullable=False, comment="Sheet名称")

    table_name = Column(String(200), nullable=False, unique=True, comment="SQLite表名（唯一）")
    row_count = Column(Integer, default=0, comment="表行数")
    column_count = Column(Integer, default=0, comment="表列数")

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), comment="更新时间")


Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get a database session (for non-FastAPI contexts)."""
    return SessionLocal()
