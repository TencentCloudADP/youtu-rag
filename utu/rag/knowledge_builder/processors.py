"""Document processors for different file types and data sources.

This module provides:
1. BaseProcessor - Abstract base class for all processors
2. FileProcessorFactory - Factory to create appropriate processor for file types
3. Concrete processors for PDF, DOCX, TXT, Excel, Database
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..base import Document
from ..document_loaders import BaseDocumentLoader
from ..storage.base_storage import BaseVectorStore
from .chunker import RecursiveTextSplitter, HierarchicalMarkdownSplitter
from ..embeddings.factory import EmbedderFactory
from ..config import ChunkingConfig

logger = logging.getLogger(__name__)


class ProcessResult(BaseModel):
    source_identifier: str  # File name or database identifier
    source_type: str  # 'pdf', 'docx', 'txt', 'excel', 'database'
    status: Literal["completed", "failed", "skipped"]  # Added "skipped" for incremental build
    chunks_created: int
    tables_created: list[str] = []
    error_message: str | None = None
    metadata: dict[str, Any] = {}


# ==================== Base Processor ====================


class BaseProcessor(ABC):
    """Base class for all processors.
    
    Should implement:
    - process(self, source_identifier: str, config: dict[str, Any]) -> ProcessResult
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        relational_db_path: str | None = None,
        minio_client = None,  # MinIOClient from api.minio_client
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        column_value_vectorization_strategy: str = "individual",
        column_value_top_n: int = 10,
        batch_delay: float = 3.0,
        batch_size: int = 50,
        embedding_type: str = "api",  # "local" or "api"
        embedding_base_url: str | None = None,  # Base URL for embedding service
    ):
        """Initialize the processor.

        Args:
            vector_store: Vector store instance
            relational_db_path: Path to relational database (for structured data)
            minio_client: MinIO client (for file downloads)
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            column_value_vectorization_strategy: Column value vectorization strategy ("individual" or "concatenate")
            column_value_top_n: Top N frequent values per column for vectorization
            batch_delay: Delay between batches (seconds)
            batch_size: embedding API batch size
            embedding_type: Embedding type ("local" or "api")
            embedding_base_url: Base URL for embedding service
        """
        self.vector_store = vector_store
        self.relational_db_path = relational_db_path or "rag_data/relational_database/rag_demo.sqlite"
        self.minio_client = minio_client
        self.column_value_vectorization_strategy = column_value_vectorization_strategy
        self.column_value_top_n = column_value_top_n
        self.batch_delay = batch_delay
        self.batch_size = batch_size

        chunking_config = ChunkingConfig(
            strategy="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunker = RecursiveTextSplitter(config=chunking_config)
        logger.info(f"🔧 Chunker initialized with: chunk_size={self.chunker.config.chunk_size}, chunk_overlap={self.chunker.config.chunk_overlap}")

        if embedding_type == "local":
            logger.info("🔧 Using LOCAL embedding service (ServiceEmbedder)")
            self.embedding_generator = EmbedderFactory.create(
                backend="service",
                service_url=embedding_base_url if embedding_base_url else None,
                batch_delay=batch_delay,
                batch_size=batch_size
            )
        else:
            logger.info("🔧 Using API-based embedding service (OpenAIEmbedder)")
            self.embedding_generator = EmbedderFactory.create(
                backend="openai",
                base_url=embedding_base_url,
                batch_delay=batch_delay,
                batch_size=batch_size
            )
  

    async def _get_file_metadata(self, source_identifier: str) -> dict[str, Any]:
        """Read metadata from MinIO (by file name).

        Args:
            source_identifier: File name or object name

        Returns:
            Metadata dictionary, empty dict if read fails or no MinIO client
        """
        if not self.minio_client:
            logger.debug(f"No MinIO client available for {source_identifier}")
            return {}

        try:
            metadata = self.minio_client.get_file_metadata(source_identifier)
            if metadata:
                logger.info(f"✓ Retrieved metadata for {source_identifier}: {len(metadata)} fields")
                return metadata
            else:
                logger.debug(f"No metadata found for {source_identifier}")
                return {}
        except Exception as e:
            logger.warning(f"Failed to get metadata for {source_identifier}: {e}")
            return {}

    async def _get_file_etag(self, source_identifier: str) -> str | None:
        """Get ETag from MinIO (used for version identification).

        Args:
            source_identifier: File name or object name

        Returns:
            ETag string, None if failed
        """
        if not self.minio_client:
            return None

        try:
            stat = self.minio_client.get_file_stat(source_identifier)
            if stat and hasattr(stat, 'etag'):
                logger.debug(f"✓ Retrieved ETAG for {source_identifier}: {stat.etag}")
                return stat.etag
            return None
        except Exception as e:
            logger.warning(f"Failed to get ETAG for {source_identifier}: {e}")
            return None

    def _get_metadata_hash(self, metadata: dict[str, Any]) -> str:
        """Compute metadata hash for version identification.

        The metadata hash is stored in the kb_source_configs table (not in MinIO).
        It is used to detect metadata changes and automatically triggers rebuild when changed.

        Args:
            metadata: Metadata dictionary read from MinIO (only original fields)

        Returns:
            MD5 hash string
        """
        metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=False)

        return hashlib.md5(metadata_json.encode('utf-8')).hexdigest()

    @abstractmethod
    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        """Process a data source.

        Args:
            source_identifier: Source identifier (filename, database connection, etc.)
            config: Source configuration

        Returns:
            Process result
        """
        pass

    async def _load_document(
        self,
        file_data: bytes,
        filename: str,
        file_type: str,
        minio_metadata: dict[str, Any] | None = None,
        etag: str | None = None
    ) -> Document:
        """Load a document and merge MinIO metadata.

        Priority (from high to low):
        1. Chunk-derived files (if chunk_processed="chunk_success")
        2. OCR-derived files (if ocr_processed="ocr_success")
        3. Original file

        Args:
            file_data: Bytes data
            filename: File name
            file_type: File type
            minio_metadata: MinIO metadata (optional)
            etag: File ETag (optional)

        Returns:
            Document object (with merged metadata)
        """
        from pathlib import Path

        chunk_processed = minio_metadata.get("chunk_processed") == "chunk_success" if minio_metadata else False
        ocr_processed = minio_metadata.get("ocr_processed") == "ocr_success" if minio_metadata else False
        content = None
        use_hierarchical_splitter = False

        # Step 1: Collect all derived file ETags (independent of Step 2)
        derived_etags = []
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")

        if chunk_processed and self.minio_client:
            chunked_filename = f"{Path(filename).stem}_chunklevel.md"
            try:
                chunk_stat = self.minio_client.get_file_stat(chunked_filename, bucket_name=sys_bucket)
                if chunk_stat and hasattr(chunk_stat, 'etag'):
                    derived_etags.append(chunk_stat.etag.strip('"'))
                    logger.debug(f"Collected chunk ETag for {chunked_filename}")
            except Exception as e:
                logger.debug(f"Chunk file {chunked_filename} not found: {e}")

        if ocr_processed and self.minio_client:
            _, ocr_derived_etags = self.minio_client.load_derived_markdown_files(
                source_filename=filename,
                sys_bucket=sys_bucket
            )
            if ocr_derived_etags:
                derived_etags.extend(ocr_derived_etags)
                logger.debug(f"Collected {len(ocr_derived_etags)} OCR ETags for {filename}")

        # Step 2: Load content (chunk-derived -> OCR-derived -> original)
        if chunk_processed and self.minio_client:
            chunked_filename = f"{Path(filename).stem}_chunklevel.md"
            logger.info(f"Attempting to load chunk-processed file: {chunked_filename}")

            try:
                file_stream = self.minio_client.download_file(
                    object_name=chunked_filename,
                    bucket_name=sys_bucket
                )
                if file_stream:
                    content = file_stream.read().decode('utf-8')
                    logger.info(f"✓ Loaded chunk-processed file: {chunked_filename} ({len(content)} chars)")
                    # 🔥 Need special spliter for chunked files
                    use_hierarchical_splitter = True
            except Exception as e:
                logger.warning(f"Failed to load chunk file {chunked_filename}: {e}")
                content = None

        if content is None and ocr_processed and self.minio_client:
            logger.info(f"Attempting to load OCR derived markdown files for {filename}")

            ocr_content, _ = self.minio_client.load_derived_markdown_files(
                source_filename=filename,
                sys_bucket=sys_bucket
            )

            if ocr_content:
                content = ocr_content
                logger.info(f"✓ Loaded OCR derived markdown for {filename} ({len(content)} chars)")
            else:
                logger.warning(f"⚠ OCR derived markdown not found or corrupted for {filename}, falling back to original loader")

        if content is None:
            # Note: Extract extension from filename, instead of using the file_type.
            # Otherwise, image files may be incorrectly treated as text files.
            actual_extension = Path(filename).suffix.lstrip('.').lower()
            if actual_extension:
                loader = BaseDocumentLoader.from_extension(actual_extension)
            else:  # Fallback to file_type if filename has no extension
                loader = BaseDocumentLoader.from_extension(file_type)

            content = loader.load(file_data, filename)

            # Special handling: image files with OCR flag but missing derived files
            if ocr_processed and actual_extension in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
                logger.warning(f"⚠ Image file {filename} marked as ocr_processed but no derived files found")
                content = (
                    f"[OCR派生文件缺失] 此图片已进行OCR处理，但派生markdown文件未找到。"
                    f"请重新处理或联系管理员。\n原始文件: {filename}"
                )

        metadata = {
            "source": filename,
            "file_type": file_type
        }

        if etag:
            metadata["etag"] = etag

        if minio_metadata:
            if "char_length" in minio_metadata:
                metadata["char_length"] = minio_metadata["char_length"]
            if "publish_date" in minio_metadata:
                metadata["publish_date"] = minio_metadata["publish_date"]
            if "key_timepoints" in minio_metadata:
                metadata["key_timepoints"] = minio_metadata["key_timepoints"]
            if "summary" in minio_metadata:
                metadata["summary"] = minio_metadata["summary"]

            standard_fields = {"char_length", "publish_date", "key_timepoints", "summary"}
            custom_fields = {k: v for k, v in minio_metadata.items() if k not in standard_fields}
            if custom_fields:
                metadata.update(custom_fields)
                logger.debug(f"Added {len(custom_fields)} custom metadata fields for {filename}")

        if derived_etags:
            metadata["_derived_files_etags"] = derived_etags

        if use_hierarchical_splitter:
            metadata["_use_hierarchical_splitter"] = True
            logger.info(f"📊 Will use HierarchicalMarkdownSplitter for {filename}")

        return Document(
            id=filename,
            content=content,
            metadata=metadata,
        )

    async def _chunk_and_store(
        self, document: Document, metadata: dict[str, Any] | None = None
    ) -> int:
        """Chunk the given document and store it in the vector store.

        ⚠️ Important: If the document already exists, old chunks will be deleted before adding new ones.
        This ensures no duplicate vector data when files are updated.

        🔥 Smart chunking strategy:
        - If the document comes from chunklevel.md (metadata["_use_hierarchical_splitter"]=True),
          use HierarchicalMarkdownSplitter to split by heading levels
        - Otherwise use default RecursiveTextSplitter

        Args:
            document: Document object
            metadata: Additional metadata

        Returns:
            Number of chunks created
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        try:
            deleted_count = await self.vector_store.delete_by_document_id(document.id)
            if deleted_count > 0:
                logger.info(f"🗑️  Deleted {deleted_count} old chunks for document: {document.id}")
        except Exception as e:
            logger.warning(f"Failed to delete old chunks for {document.id}: {e}")
            # Continue processing even if deletion fails

        use_hierarchical = document.metadata.get("_use_hierarchical_splitter", False)
        if use_hierarchical:
            hierarchical_config = ChunkingConfig(
                strategy="hierarchical",
                chunk_size=self.chunker.config.chunk_size,
                chunk_overlap=self.chunker.config.chunk_overlap
            )
            chunker = HierarchicalMarkdownSplitter(config=hierarchical_config)
            logger.info(f"📊 Using HierarchicalMarkdownSplitter for {document.id}")
        else:
            chunker = self.chunker
            logger.debug(f"Using default RecursiveTextSplitter for {document.id}")

        chunk_texts = chunker.split_text(document.content, document.metadata)

        from ..base import Chunk
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            # Metadata should not be saved to vector database
            chunk_metadata = {k: v for k, v in document.metadata.items()
                            if not k.startswith("_")}
            if "index_type" not in chunk_metadata:
                chunk_metadata["index_type"] = "index_content"

            chunk = Chunk(
                id=f"{document.id}_chunk_{i}",
                document_id=document.id,
                content=chunk_text,
                chunk_index=i,
                metadata=chunk_metadata
            )
            chunks.append(chunk)

        if not chunks:
            logger.warning(f"No chunks created for document: {document.id}")
            return 0

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_generator.embed_texts(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        await self.vector_store.add_chunks(chunks)

        logger.info(f"Created {len(chunks)} chunks for {document.id}")
        return len(chunks)

    async def _create_summary_index(
        self, document: Document
    ) -> int:
        """Create summary index vectors (if summary exists) for the given document.

        Concatenate filename and summary, then encode and store as a single vector (labelled as index_type: "index_summary").

        Args:
            document: Document object

        Returns:
            Number of summary index vectors created (0 or 1)
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        summary = document.metadata.get("summary")
        if not summary: 
            summary = ""

        filename = document.metadata.get("source", document.id)
        summary_content = f"{filename}\n{summary}"

        from ..base import Chunk
        summary_chunk = Chunk(
            id=f"{document.id}_summary",
            document_id=document.id,
            content=summary_content,
            chunk_index=-1,  # -1 represents a summary chunk
            metadata={
                **document.metadata,
                "index_type": "index_summary"
            }
        )

        embedding = await self.embedding_generator.embed_texts([summary_content])
        summary_chunk.embedding = embedding[0]

        await self.vector_store.add_chunks([summary_chunk])

        logger.info(f"Created summary index for {document.id}")
        return 1

    async def _store_to_sqlite(
        self, df: pd.DataFrame, table_name: str
    ) -> str:
        """Save DataFrame to SQLite.

        Args:
            df: DataFrame
            table_name: table name

        Returns:
            Table name
        """
        os.makedirs(os.path.dirname(self.relational_db_path), exist_ok=True)

        conn = sqlite3.connect(self.relational_db_path)

        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} rows to SQLite table: {table_name}")
            return table_name

        finally:
            conn.close()

    async def _get_file_data(
        self, source_identifier: str, config: dict[str, Any]
    ) -> bytes:
        """Get file data from local file or MinIO.
        
        This method is shared by all file processors, with priority:
        1. Load local file (if file_path exists in config and file exists)
        2. Download from MinIO (using source_identifier)
        3. If all fail, raise FileNotFoundError

        Args:
            source_identifier: File name or object name
            config: Configuration dictionary

        Returns:
            File bytes data
        """
        file_path = config.get("file_path", source_identifier)

        if os.path.exists(file_path):
            logger.debug(f"Reading file from local path: {file_path}")
            with open(file_path, "rb") as f:
                return f.read()

        if self.minio_client:
            try:
                logger.info(f"Downloading file from MinIO: {source_identifier}")
                file_stream = self.minio_client.download_file(source_identifier)
                if file_stream:
                    file_data = file_stream.read()
                    logger.info(f"✓ Downloaded {len(file_data)} bytes from MinIO: {source_identifier}")
                    return file_data
            except Exception as e:
                logger.warning(f"Failed to download from MinIO ({source_identifier}): {e}")

        raise FileNotFoundError(f"File not found: {file_path}")

    async def _process_simple_document(
        self, 
        source_identifier: str, 
        config: dict[str, Any],
        file_type: str
    ) -> ProcessResult:
        """General process for simple documents (PDF/DOCX/TXT).

        Load -> chunk and store -> create summary index.
        
        Args:
            source_identifier: Source identifier
            config: Configuration dictionary
            file_type: File type (pdf/docx/txt)
            
        Returns:
            Process result
        """
        try:
            file_data = await self._get_file_data(source_identifier, config)

            minio_metadata = await self._get_file_metadata(source_identifier)
            etag = await self._get_file_etag(source_identifier)

            document = await self._load_document(
                file_data, source_identifier, file_type,
                minio_metadata=minio_metadata,
                etag=etag
            )

            derived_etags = document.metadata.pop("_derived_files_etags", [])

            chunks_count = await self._chunk_and_store(document)

            summary_count = await self._create_summary_index(document)

            return ProcessResult(
                source_identifier=source_identifier,
                source_type=file_type,
                status="completed",
                chunks_created=chunks_count + summary_count,
                metadata={"derived_files_etags": derived_etags} if derived_etags else {}
            )

        except Exception as e:
            logger.error(f"{file_type.upper()} processing failed for {source_identifier}: {str(e)}")
            return ProcessResult(
                source_identifier=source_identifier,
                source_type=file_type,
                status="failed",
                chunks_created=0,
                error_message=str(e),
            )

    async def _create_column_vectors_shared(
        self,
        df: pd.DataFrame,
        table_name: str,
        collection_name: str
    ) -> int:
        """Create column vectors (used for value link) - shared by Excel and Database
        
        Supports two strategies:
        1. individual: each value vectorized independently (precise matching, more vectors)
        2. concatenate: all values concatenated and vectorized (space-efficient, semantic dilution)

        Args:
            df: DataFrame
            table_name: table name
            collection_name: vector store collection name

        Returns:
            Number of created chunks
        """
        total_chunks = 0

        for col in df.columns:
            # Create value vectors for text columns only
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                unique_values = df[col].dropna().unique()

                top_n = self.column_value_top_n
                if len(unique_values) > top_n:
                    value_counts = df[col].value_counts().head(top_n)
                    values_to_vectorize = value_counts.index.tolist()
                else:
                    values_to_vectorize = unique_values.tolist()

                if self.column_value_vectorization_strategy == "individual":
                    success_count = 0
                    failed_values = []

                    for value in values_to_vectorize:
                        try:
                            doc = Document(
                                id=f"column_value_{table_name}_{col}_{value}",
                                content=str(value),
                                metadata={
                                    "table_name": table_name,
                                    "column_name": col,
                                    "type": "column_value",
                                    "index_type": "index_element",
                                    "value": str(value)
                                }
                            )

                            chunks = await self._chunk_and_store(doc)
                            total_chunks += chunks
                            success_count += 1

                        except Exception as e:
                            # Failure on an individual value won't interrupt the overall process
                            failed_values.append(str(value)[:50])
                            logger.warning(
                                f"⚠️  Failed to vectorize column value '{str(value)[:50]}...' "
                                f"in {table_name}.{col}: {str(e)[:100]}"
                            )
                            continue

                    logger.info(
                        f"✓ Created column-level vectors (individual) for: {table_name}.{col} "
                        f"({success_count}/{len(values_to_vectorize)} values succeeded)"
                    )
                    if failed_values:
                        logger.warning(
                            f"⚠️  {len(failed_values)} values failed for {table_name}.{col}: "
                            f"{', '.join(failed_values[:5])}{'...' if len(failed_values) > 5 else ''}"
                        )

                else:  # concatenate
                    try:
                        values_text = "\n".join(str(v) for v in values_to_vectorize)

                        doc = Document(
                            id=f"column_values_{table_name}_{col}",
                            content=values_text,
                            metadata={
                                "table_name": table_name,
                                "column_name": col,
                                "type": "column_values",
                                "index_type": "index_element",
                                "value_count": len(values_to_vectorize)
                            }
                        )

                        chunks = await self._chunk_and_store(doc)
                        total_chunks += chunks

                        logger.info(
                            f"✓ Created column-level vector (concatenate) for: {table_name}.{col} "
                            f"({len(values_to_vectorize)} values → {chunks} chunks)"
                        )

                    except Exception as e:
                        logger.warning(
                            f"⚠️  Failed to vectorize column values (concatenate) for {table_name}.{col}: "
                            f"{str(e)[:100]}"
                        )

        return total_chunks


# ==================== Concrete Processors ====================


class PDFProcessor(BaseProcessor):

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        return await self._process_simple_document(source_identifier, config, "pdf")


class WordProcessor(BaseProcessor):

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        return await self._process_simple_document(source_identifier, config, "docx")


class TextProcessor(BaseProcessor):

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        return await self._process_simple_document(source_identifier, config, "txt")


class ExcelProcessor(BaseProcessor):
    """Supports structured table detection and two-level vectorization (table-level + column-level)"""

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        """Process Excel file.

        Workflow:
        1. Read all sheets of the Excel file;
        2. Detect structured tables in each sheet;
        3. For structured tables: store to SQLite, create table-level and column-level vectors;
        4. For unstructured tables: vectorize by rows.
        5. (Added) Textify tables.
        6. Create summary vectors.
        """
        try:
            file_data = await self._get_file_data(source_identifier, config)

            minio_metadata = await self._get_file_metadata(source_identifier)
            etag = await self._get_file_etag(source_identifier)

            collection_name = config.get("collection_name", "default")  # Used for vectorization
            kb_id = config.get("kb_id", 0)
            source_id = config.get("source_id")  # Associated with kb_source_configs.id

            excel_file = pd.ExcelFile(io.BytesIO(file_data))

            tables_created = []
            total_chunks_count = 0

            for sheet_name in excel_file.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(excel_file, sheet_name=sheet_name)

                is_structured = await self._detect_structured(df)

                if is_structured:
                    table_name = self._generate_table_name(source_identifier, sheet_name, kb_id)

                    table_metadata = await self._store_to_sqlite_enhanced(
                        df, table_name, source_identifier, sheet_name, kb_id, source_id
                    )
                    tables_created.append(table_name)
                    try:
                        table_chunks = await self._create_table_vector(
                            table_name, table_metadata, source_identifier, sheet_name
                        )
                        total_chunks_count += table_chunks
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to create table vector for {table_name}: {str(e)[:100]}")

                    try:
                        column_chunks = await self._create_column_vectors_shared(
                            df, table_name, collection_name
                        )
                        total_chunks_count += column_chunks
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to create column vectors for {table_name}: {str(e)[:100]}")

                else:
                    document = self._create_row_document(
                        df, f"{source_identifier}#{sheet_name}",
                        minio_metadata=minio_metadata,
                        etag=etag
                    )
                    chunks_count = await self._chunk_and_store(document)
                    total_chunks_count += chunks_count

            # Add: Excel textification - generate text chunks whether structured or not (index_type=index_content)
            logger.info(f"📄 Creating text chunks for Excel file: {source_identifier}")
            try:
                from ..document_loaders.excel_loader import ExcelLoader

                excel_loader = ExcelLoader()
                text_content = excel_loader.load(file_data, source_identifier)

                if text_content:
                    text_doc = Document(
                        id=f"{source_identifier}_text",
                        content=text_content,
                        metadata={
                            "source": source_identifier,
                            "file_type": "excel",
                            **{k: v for k, v in (minio_metadata or {}).items() if k not in ["summary"]}
                        }
                    )
                    if etag:
                        text_doc.metadata["etag"] = etag

                    # Chunk the text, and index_type will be set to "index_content" automatically
                    text_chunks_count = await self._chunk_and_store(text_doc)
                    total_chunks_count += text_chunks_count
                    logger.info(f"✅ Created {text_chunks_count} text chunks (index_type=index_content) for {source_identifier}")
                else:
                    logger.warning(f"⚠️  No text content extracted from {source_identifier}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to create text chunks for {source_identifier}: {e}")
                # Textification failure won't interrupt the main processing

            if minio_metadata and minio_metadata.get("summary"):
                summary_doc = Document(
                    id=source_identifier,
                    content="",  # Empty, only used to pass metadata
                    metadata={
                        "source": source_identifier,
                        "file_type": "excel",
                        "summary": minio_metadata.get("summary"),
                        **{k: v for k, v in (minio_metadata or {}).items() if k not in ["summary"]}
                    }
                )
                if etag:
                    summary_doc.metadata["etag"] = etag
                summary_count = await self._create_summary_index(summary_doc)
                total_chunks_count += summary_count

            return ProcessResult(
                source_identifier=source_identifier,
                source_type="excel",
                status="completed",
                chunks_created=total_chunks_count,
                tables_created=tables_created,
            )

        except Exception as e:
            logger.error(f"Excel processing failed for {source_identifier}: {str(e)}", exc_info=True)
            return ProcessResult(
                source_identifier=source_identifier,
                source_type="excel",
                status="failed",
                chunks_created=0,
                error_message=str(e),
            )



    async def _detect_structured(self, df: pd.DataFrame) -> bool:
        """Determine if the given DataFrame is a structured table.

        Rules:
        1. Has explicit column names (not Unnamed);
        2. Column count >= 2;
        3. Row count >= 5.
        """
        if len(df.columns) < 2:
            return False

        if len(df) < 5:
            return False

        unnamed_count = sum(1 for col in df.columns if str(col).startswith("Unnamed"))
        if unnamed_count > len(df.columns) / 2:
            return False

        return True

    def _generate_table_name(self, source_identifier: str, sheet_name: str = "", kb_id: int = 0) -> str:
        """Generate a unique table name, including kb_id to avoid cross-KB conflicts.

        Args:
            source_identifier: Source identifier
            sheet_name: Sheet name
            kb_id: Knowledge base ID

        Returns:
            Unique table name (`excel_{kb_id}_{filename}_{sheet}`)
        """
        from .excel_table_manager import ExcelTableManager

        filename = Path(source_identifier).name

        return ExcelTableManager.generate_table_name(kb_id, filename, sheet_name or "default")

    def _create_row_document(
        self,
        df: pd.DataFrame,
        source_identifier: str,
        minio_metadata: dict[str, Any] | None = None,
        etag: str | None = None
    ) -> Document:
        """Create row document for unstructured Excel (including MinIO metadata)."""
        rows_text = []
        for _, row in df.iterrows():
            row_text = " | ".join(str(v) for v in row.values)
            rows_text.append(row_text)

        content = "\n".join(rows_text)

        metadata = {
            "source": source_identifier,
            "type": "excel_rows"
        }

        if etag:
            metadata["etag"] = etag

        if minio_metadata:
            if "char_length" in minio_metadata:
                metadata["char_length"] = minio_metadata["char_length"]
            if "publish_date" in minio_metadata:
                metadata["publish_date"] = minio_metadata["publish_date"]
            if "key_timepoints" in minio_metadata:
                metadata["key_timepoints"] = minio_metadata["key_timepoints"]
            if "summary" in minio_metadata:
                metadata["summary"] = minio_metadata["summary"]

            standard_fields = {"char_length", "publish_date", "key_timepoints", "summary"}
            custom_fields = {k: v for k, v in minio_metadata.items() if k not in standard_fields}
            if custom_fields:
                metadata.update(custom_fields)

        return Document(
            id=source_identifier,
            content=content,
            metadata=metadata,
        )

    async def _store_to_sqlite_enhanced(
        self,
        df: pd.DataFrame,
        table_name: str,
        source_identifier: str,
        sheet_name: str,
        kb_id: int = 0,
        source_id: int | None = None
    ) -> dict[str, Any]:
        """Enhanced SQLite storage for Excel tables.

        Refer to excel2sqlite.py

        Supports:
        1. Store DataFrame to SQLite;
        2. Record detailed table metadata (column types, sample values, etc.);
        3. Register table mapping to kb_excel_tables;
        4. Return metadata dictionary for vectorization.

        Args:
            df: DataFrame data
            table_name: Table name
            source_identifier: Source identifier
            sheet_name: Sheet name
            kb_id: Knowledge base ID
            source_id: Source config ID (associated with kb_source_configs.id)

        Returns:
            Dictionary containing table metadata
        """
        os.makedirs(os.path.dirname(self.relational_db_path), exist_ok=True)

        conn = sqlite3.connect(self.relational_db_path)

        try:
            columns = list(df.columns)
            col_types = []
            cells_list = []

            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ('

            for col, dtype in zip(columns, df.dtypes):
                if pd.api.types.is_string_dtype(dtype):
                    sqlite_type = "TEXT"
                elif pd.api.types.is_integer_dtype(dtype):
                    sqlite_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sqlite_type = "REAL"
                elif pd.api.types.is_bool_dtype(dtype):
                    sqlite_type = "INTEGER"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sqlite_type = "TEXT"
                else:
                    sqlite_type = "TEXT"

                col_types.append(sqlite_type)
                create_sql += f'\n  "{col}" {sqlite_type},'

            create_sql = create_sql.rstrip(",") + "\n);"

            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            conn.execute(create_sql)

            data_to_insert = []
            column_value_sets = [set() for _ in range(len(columns))]

            for _, row in df.iterrows():
                row_data = []
                for i, val in enumerate(row):
                    if str(val) == "-" or pd.isna(val):
                        converted_val = None
                    elif isinstance(val, (np.int64, np.int32)):
                        converted_val = int(val)
                    elif isinstance(val, (np.float64, np.float32)):
                        converted_val = float(val)
                    elif isinstance(val, pd.Timestamp):
                        converted_val = str(val)
                    else:
                        converted_val = str(val)

                    row_data.append(converted_val)
                    column_value_sets[i].add(converted_val)

                data_to_insert.append(tuple(row_data))

            placeholders = ", ".join(["?"] * len(columns))
            column_names = ",".join(f'"{c}"' for c in columns)
            insert_sql = f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})'
            conn.executemany(insert_sql, data_to_insert)

            # Get top values for each column, used for value link
            for i, (col, col_type) in enumerate(zip(columns, col_types)):
                if col_type == 'TEXT':
                    # For text columns, sort by frequency and get top 100
                    try:
                        cursor = conn.execute(f'''
                            SELECT "{col}", COUNT(*) as cnt
                            FROM "{table_name}"
                            WHERE "{col}" IS NOT NULL
                            GROUP BY "{col}"
                            ORDER BY cnt DESC
                            LIMIT 100
                        ''')
                        cells_list.append([row[0] for row in cursor.fetchall()])
                    except Exception:
                        # Fallback to first 100 unique values
                        cells_list.append(list(column_value_sets[i])[:100])
                else:
                    # For non-text columns, get first 100 unique values
                    cells_list.append(list(column_value_sets[i])[:100])

            conn.commit()

            metadata = {
                "table_name": table_name,
                "file_name": Path(source_identifier).name,
                "sheet_name": sheet_name,
                "columns_name": columns,
                "columns_types": col_types,
                "table_sql": create_sql,
                "cells": cells_list,  # Samples for each column
                "row_count": len(df),
                "column_count": len(columns)
            }

            logger.info(f"✓ Stored {len(df)} rows to SQLite table: {table_name}")

        except Exception as e:
            logger.error(f"Failed to store to SQLite: {e}", exc_info=True)
            raise
        finally:
            conn.close()

        # Note: Registration should be done after the connection is closed
        if kb_id > 0:
            try:
                from ..api.database import get_db_session
                from .excel_table_manager import ExcelTableManager

                db = get_db_session()
                try:
                    ExcelTableManager.register_table(
                        db=db,
                        kb_id=kb_id,
                        source_id=source_id,  # Associated with kb_source_configs.id
                        source_file=source_identifier,
                        sheet_name=sheet_name,
                        table_name=table_name,
                        row_count=len(df),
                        column_count=len(columns)
                    )
                    logger.info(
                        f"✅ Registered table mapping: {table_name} "
                        f"(kb_id={kb_id}, source_id={source_id})"
                    )
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Failed to register table mapping: {e}")
                # Registration failure won't interrupt the main process

        return metadata

    async def _create_table_vector(
        self,
        table_name: str,
        table_metadata: dict[str, Any],
        source_identifier: str,
        sheet_name: str
    ) -> int:
        """Create table-level vector (used for schema link).

        Vectorize table structure information (table name, CREATE TABLE SQL) in JSON format.
        Format matches example script: {"table_name": "...", "table_sql": "..."}

        Args:
            table_name: Table name
            table_metadata: Table metadata
            source_identifier: Source file identifier
            sheet_name: Sheet name

        Returns:
            Number of created chunks
        """
        table_info = {
            "table_name": table_name,
            "table_sql": table_metadata["table_sql"].replace("CREATE TABLE IF NOT EXISTS","TABLE").replace("CREATE TABLE","TABLE")
        }
        table_json = json.dumps(table_info, ensure_ascii=False)

        doc = Document(
            id=f"table_schema_{table_name}",
            content=table_json,
            metadata={
                "source": source_identifier,
                "table_name": table_name,
                "type": "table_schema",
                "index_type": "index_element",
                "db_type": "excel",  # Excel tables stored in SQLite
                "sheet_name": sheet_name,
                "file_name": table_metadata.get('file_name', ''),
                "row_count": table_metadata.get('row_count', 0),
                "column_count": table_metadata.get('column_count', 0)
            }
        )

        logger.info(f"📝 Table-level vector (JSON format) for '{table_name}':")
        logger.info(f"{table_json}")

        chunks_count = await self._chunk_and_store(doc)
        logger.info(f"✓ Created table-level vector for: {table_name}")
        return chunks_count


class DatabaseProcessor(BaseProcessor):
    """Supports two-level vectorization (table-level + column-level).

    Workflow:
    1. Connect to database and read table schema (CREATE TABLE SQL)
    2. Read table data (for column value statistics)
    3. Create table-level vector (schema link) using table structure information
    4. Create column-level vector (value link) using high-frequency values in columns

    Note: No mirroring to local SQLite, table already exists in source database
    """

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        """Process a database table."""
        try:
            connection_string = config.get("connection_string", "")
            table_name = config.get("table_name", "")
            db_type = config.get("db_type", "mysql")
            collection_name = config.get("collection_name", "default")

            # For SQLite, if connection_string is empty, try to build from file_path
            if db_type == "sqlite" and not connection_string:
                file_path = config.get("file_path", "")
                if file_path:
                    connection_string = f"sqlite:///{file_path}"
                    logger.info(f"Built SQLite connection string from file_path: {connection_string}")

            df = await self._fetch_table_data(connection_string, table_name, db_type, config)

            table_sql = await self._fetch_table_schema(connection_string, table_name, db_type, config)

            total_chunks = 0

            table_chunks = await self._create_table_vector(
                table_name, table_sql, df, source_identifier, db_type
            )
            total_chunks += table_chunks

            column_chunks = await self._create_column_vectors_shared(
                df, table_name, collection_name
            )
            total_chunks += column_chunks

            logger.info(
                f"Processed {db_type} table '{table_name}': "
                f"created {total_chunks} vector chunks "
                f"(table: {table_chunks}, columns: {column_chunks})"
            )

            return ProcessResult(
                source_identifier=source_identifier,
                source_type="database",
                status="completed",
                chunks_created=total_chunks,
                tables_created=[],  # Do not create mirror tables for database
            )

        except Exception as e:
            logger.error(f"Database processing failed for {source_identifier}: {str(e)}")
            return ProcessResult(
                source_identifier=source_identifier,
                source_type="database",
                status="failed",
                chunks_created=0,
                error_message=str(e),
            )

    async def _fetch_table_data(
        self, connection_string: str, table_name: str, db_type: str, config: dict[str, Any] = None
    ) -> pd.DataFrame:
        """Fetch table data from database.

        Args:
            connection_string: Database connection string
            table_name: Table name
            db_type: Database type (sqlite/mysql)
            config: Config dictionary (for MySQL connection)

        Returns:
            Table data (all rows, for column value statistics), returned as a pandas DataFrame
        """
        import sqlite3

        if db_type == "sqlite":
            conn_path = connection_string.replace("sqlite:///", "").replace("sqlite://", "")
            conn = sqlite3.connect(conn_path)
            try:
                query = f'SELECT * FROM "{table_name}"'
                df = pd.read_sql(query, conn)
                logger.info(f"Fetched {len(df)} rows from SQLite table '{table_name}'")
                return df
            finally:
                conn.close()

        elif db_type == "mysql":
            try:
                import pymysql
            except ImportError:
                logger.error("pymysql not installed. Please install it: pip install pymysql")
                raise ImportError("pymysql is required for MySQL support")

            if not config:
                logger.error("MySQL connection requires config dict with host, port, database, username, password")
                raise ValueError("Missing config for MySQL connection")

            host = config.get("host", "")
            port = int(config.get("port", 3306))
            database = config.get("database", "")
            username = config.get("username", "")
            password = config.get("password", "")

            if not all([host, database, username]):
                logger.error(f"Missing MySQL connection params: host={host}, database={database}, username={username}")
                raise ValueError("Incomplete MySQL connection information")

            conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                charset='utf8mb4'
            )

            try:
                query = f'SELECT * FROM `{table_name}`'
                df = pd.read_sql(query, conn)
                logger.info(f"Fetched {len(df)} rows from MySQL table '{table_name}'")
                return df
            finally:
                conn.close()

        else:
            logger.warning(f"Unsupported db_type: {db_type}, using mock data")
            return pd.DataFrame({"id": [1, 2, 3], "placeholder": ["A", "B", "C"]})

    async def _fetch_table_schema(
        self, connection_string: str, table_name: str, db_type: str, config: dict[str, Any] = None
    ) -> str:
        """Fetch table schema from database.

        Args:
            connection_string: Database connection string
            table_name: Table name
            db_type: Database type (sqlite/mysql)
            config: Config dictionary (for MySQL connection)

        Returns:
            str: CREATE TABLE SQL statement
        """
        import sqlite3

        if db_type == "sqlite":
            conn_path = connection_string.replace("sqlite:///", "").replace("sqlite://", "")
            conn = sqlite3.connect(conn_path)
            try:
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:  # Fallback to DataFrame if not found
                    logger.warning(f"Table '{table_name}' not found in sqlite_master, will generate schema")
                    return ""
            finally:
                conn.close()

        elif db_type == "mysql":
            try:
                import pymysql
            except ImportError:
                logger.error("pymysql not installed. Please install it: pip install pymysql")
                return ""

            if not config:
                logger.warning("MySQL schema fetch requires config, will generate from DataFrame")
                return ""

            host = config.get("host", "")
            port = int(config.get("port", 3306))
            database = config.get("database", "")
            username = config.get("username", "")
            password = config.get("password", "")

            if not all([host, database, username]):
                logger.warning("Incomplete MySQL connection info, will generate schema from DataFrame")
                return ""

            try:
                conn = pymysql.connect(
                    host=host,
                    port=port,
                    user=username,
                    password=password,
                    database=database,
                    charset='utf8mb4'
                )

                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
                    result = cursor.fetchone()
                    if result and len(result) >= 2:
                        # SHOW CREATE TABLE returns (table_name, create_sql)
                        create_sql = result[1]
                        logger.info(f"Fetched MySQL table schema for '{table_name}'")
                        return create_sql
                    else:
                        logger.warning(f"Failed to fetch MySQL schema for '{table_name}'")
                        return ""
                finally:
                    conn.close()

            except Exception as e:
                logger.error(f"Error fetching MySQL schema for '{table_name}': {e}")
                return ""

        return ""

    async def _create_table_vector(
        self,
        table_name: str,
        table_sql: str,
        df: pd.DataFrame,
        source_identifier: str,
        db_type: str
    ) -> int:
        """Create table-level vector (used for schema link).

        Vectorize table structure (table name, CREATE TABLE SQL) as JSON format.
        Format matches the example script: {"table_name": "...", "table_sql": "..."}

        Args:
            table_name: Table name
            table_sql: CREATE TABLE SQL statement
            df: Table data (used to get column info, only used when table_sql is empty)
            source_identifier: Source identifier
            db_type: Database type

        Returns:
            Number of created chunks
        """
        if not table_sql:
            columns_sql = ", ".join(
                f"{col} {self._pandas_dtype_to_sql(df[col].dtype)}"
                for col in df.columns
            )
            table_sql = f"CREATE TABLE {table_name} ({columns_sql})"

        # Avoid "CREATE TABLE IF NOT EXISTS" since it may be recognized as SQL injection
        table_info = {
            "table_name": table_name,
            "table_sql": table_sql.replace("CREATE TABLE IF NOT EXISTS","TABLE").replace("CREATE TABLE","TABLE")
        }
        table_json = json.dumps(table_info, ensure_ascii=False)

        doc = Document(
            id=f"table_schema_{table_name}",
            content=table_json,
            metadata={
                "source": source_identifier,
                "table_name": table_name,
                "type": "table_schema",
                "index_type": "index_element",
                "db_type": db_type,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )

        logger.info(f"📝 Table-level vector (JSON format) for '{table_name}':")
        logger.info(f"{table_json}")

        chunks_count = await self._chunk_and_store(doc)
        logger.info(f"✓ Created table-level vector for: {table_name}")
        return chunks_count

    def _pandas_dtype_to_sql(self, dtype) -> str:
        """Translate pandas dtype to SQL type."""
        dtype_str = str(dtype)
        if "int" in dtype_str:
            return "INTEGER"
        elif "float" in dtype_str:
            return "REAL"
        elif "bool" in dtype_str:
            return "INTEGER"
        elif "datetime" in dtype_str:
            return "TEXT"
        else:
            return "TEXT"


# ==================== Factory ====================


class QAProcessor(BaseProcessor):
    """Associates QA with source files."""

    def __init__(
        self,
        kb_id: int,
        **kwargs
    ):
        """Initialize QA processor.

        Args:
            kb_id: Knowledge base ID
            kwargs: Other parameters passed to BaseProcessor
        """
        super().__init__(**kwargs)
        self.kb_id = kb_id

    async def process(
        self, source_identifier: str, config: dict[str, Any]
    ) -> ProcessResult:
        """Process a QA file.

        Steps:
        1. Read Excel file (question, answer, howtofind columns)
        2. Create qa_associations table to store associations

        Args:
            source_identifier: QA file identifier
            config: Configuration information

        Returns:
            Processing result
        """
        try:
            file_data = await self._get_file_data(source_identifier, config)

            sheet_name = config.get('sheet_name', 0)  
            df = pd.read_excel(io.BytesIO(file_data), sheet_name=sheet_name)

            column_mapping = {}
            header_aliases = {
                'question': ['question', '问题', '题目'],
                'answer': ['answer', '答案', '回答'],
                'howtofind': ['howtofind', 'how to find', '查找方式']
            }

            df_columns_lower = [str(col).lower().strip() for col in df.columns]
            for required_col, aliases in header_aliases.items():
                found = False
                for col_idx, col_name in enumerate(df_columns_lower):
                    if col_name in aliases:
                        column_mapping[df.columns[col_idx]] = required_col
                        found = True
                        break
                if not found:
                    raise ValueError(f"QA file missing required column: {required_col}. Expected one of: {aliases}")

            df = df.rename(columns=column_mapping)

            required_cols = ['question', 'answer', 'howtofind']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"QA file missing required columns: {missing_cols}")

            qa_count = await self._create_qa_associations(df, source_identifier)

            logger.info(f"Created {qa_count} QA associations for KB {self.kb_id}")

            return ProcessResult(
                source_identifier=source_identifier,
                source_type="qa_file",
                status="completed",
                chunks_created=qa_count,  # 记录QA对的数量
                metadata={"qa_pairs": qa_count}
            )

        except Exception as e:
            logger.error(f"QA processing failed for {source_identifier}: {str(e)}", exc_info=True)
            return ProcessResult(
                source_identifier=source_identifier,
                source_type="qa_file",
                status="failed",
                chunks_created=0,
                error_message=str(e)
            )

    async def _create_qa_associations(self, df: pd.DataFrame, source_file: str = None) -> int:
        """Create QA associations table (qa_associations).

        Table structure:
        - qa_id: primary key
        - kb_id: knowledge base ID
        - question: question
        - answer: answer
        - howtofind: how to find description
        - source_file: source file name (for precise cleanup)

        Args:
            df: QA DataFrame
            source_file: Source file name; clear the entire knowledge base if not provided; otherwise, clear only the specified source file

        Returns:
            Number of QA associations created
        """
        os.makedirs(os.path.dirname(self.relational_db_path), exist_ok=True)

        conn = sqlite3.connect(self.relational_db_path)

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_associations (
                    qa_id INTEGER PRIMARY KEY,
                    kb_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    howtofind TEXT,
                    source_file TEXT,  -- 来源文件名
                    learning_status TEXT DEFAULT 'pending',  -- 学习状态: pending, learning, completed, failed
                    memory_status TEXT DEFAULT 'pending',  -- 记忆状态: pending, memorized, failed
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            if source_file:
                conn.execute(
                    "DELETE FROM qa_associations WHERE kb_id = ? AND source_file = ?",
                    (self.kb_id, source_file)
                )
            else:
                conn.execute("DELETE FROM qa_associations WHERE kb_id = ?", (self.kb_id,))

            inserted_count = 0
            for _, row in df.iterrows():
                conn.execute(
                    """
                    INSERT INTO qa_associations
                    (kb_id, question, answer, howtofind, source_file)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.kb_id,
                        row.get('question', ''),
                        row.get('answer', ''),
                        row.get('howtofind', ''),
                        source_file
                    )
                )
                inserted_count += 1

            conn.commit()
            logger.info(f"Inserted {inserted_count} QA pairs into qa_associations table")
            return inserted_count

        finally:
            conn.close()


class FileProcessorFactory:
    """Wrap file processor as a factory to avoid massive if-else."""

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        relational_db_path: str | None = None,
        minio_client = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        column_value_vectorization_strategy: str = "individual",
        column_value_top_n: int = 10,
        batch_delay: float = 3.0,
        batch_size: int = 50,
        embedding_type: str = "api",  # "local" or "api"
        embedding_base_url: str | None = None,  # Base URL for embedding service
    ):
        """Initialize.

        Args:
            vector_store: Vector store
            relational_db_path: Relational database path
            minio_client: MinIO client
            chunk_size: Text chunk size
            chunk_overlap: Text chunk overlap
            column_value_vectorization_strategy: Column value vectorization strategy (for Excel/CSV/Database)
            column_value_top_n: Top N frequent values per column to vectorize (for Excel/CSV/Database)
            batch_delay: Batch delay (seconds)
            batch_size: Embedding API batch size
            embedding_type: Embedding type ("local" or "api")
            embedding_base_url: Embedding service base URL
        """
        self.vector_store = vector_store
        self.relational_db_path = relational_db_path
        self.minio_client = minio_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.column_value_vectorization_strategy = column_value_vectorization_strategy
        self.column_value_top_n = column_value_top_n
        self.batch_delay = batch_delay
        self.batch_size = batch_size
        self.embedding_type = embedding_type
        self.embedding_base_url = embedding_base_url

        # Only Excel/CSV/Database need column value vectorization configuration
        self.processors = {
            "pdf": PDFProcessor(vector_store, relational_db_path, minio_client, chunk_size=chunk_size, chunk_overlap=chunk_overlap, batch_delay=batch_delay, batch_size=batch_size, embedding_type=embedding_type, embedding_base_url=embedding_base_url),
            "docx": WordProcessor(vector_store, relational_db_path, minio_client, chunk_size=chunk_size, chunk_overlap=chunk_overlap, batch_delay=batch_delay, batch_size=batch_size, embedding_type=embedding_type, embedding_base_url=embedding_base_url),
            "txt": TextProcessor(vector_store, relational_db_path, minio_client, chunk_size=chunk_size, chunk_overlap=chunk_overlap, batch_delay=batch_delay, batch_size=batch_size, embedding_type=embedding_type, embedding_base_url=embedding_base_url),
            "excel": ExcelProcessor(
                vector_store, relational_db_path, minio_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                column_value_vectorization_strategy=column_value_vectorization_strategy,
                column_value_top_n=column_value_top_n,
                batch_delay=batch_delay,
                batch_size=batch_size,
                embedding_type=embedding_type,
                embedding_base_url=embedding_base_url
            ),
            "xlsx": ExcelProcessor(
                vector_store, relational_db_path, minio_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                column_value_vectorization_strategy=column_value_vectorization_strategy,
                column_value_top_n=column_value_top_n,
                batch_delay=batch_delay,
                batch_size=batch_size,
                embedding_type=embedding_type,
                embedding_base_url=embedding_base_url
            ),
            "xls": ExcelProcessor(
                vector_store, relational_db_path, minio_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                column_value_vectorization_strategy=column_value_vectorization_strategy,
                column_value_top_n=column_value_top_n,
                batch_delay=batch_delay,
                batch_size=batch_size,
                embedding_type=embedding_type,
                embedding_base_url=embedding_base_url
            ),
            "csv": ExcelProcessor(  # CSV is processed using ExcelProcessor
                vector_store, relational_db_path, minio_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                column_value_vectorization_strategy=column_value_vectorization_strategy,
                column_value_top_n=column_value_top_n,
                batch_delay=batch_delay,
                batch_size=batch_size,
                embedding_type=embedding_type,
                embedding_base_url=embedding_base_url
            ),
        }

        # Support image files, using TextProcessor to process OCR derived files.
        # BaseProcessor._load_document() will load OCR derived markdown files first.
        text_processor = self.processors["txt"]
        image_types = ["png", "jpg", "jpeg", "bmp", "webp", "gif"]
        for img_type in image_types:
            self.processors[img_type] = text_processor
        self.processors["md"] = text_processor

    def create(self, file_type: str) -> BaseProcessor:
        """Create the processor.

        Args:
            file_type: File type.

        Returns:
            Processor instance.
        """
        processor = self.processors.get(file_type.lower())

        if not processor:
            raise ValueError(f"Unsupported file type: {file_type}")

        return processor
