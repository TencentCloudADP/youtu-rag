"""Knowledge Builder implementation."""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any

from ..base import BaseKnowledgeBuilder, BaseVectorStore, BuildStatus, Chunk, Document
from ..config import KnowledgeBuilderConfig
from .chunker import RecursiveTextSplitter
from ..embeddings.factory import EmbedderFactory

logger = logging.getLogger(__name__)


class KnowledgeBuilder(BaseKnowledgeBuilder):
    """Main implementation of knowledge builder."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        config: KnowledgeBuilderConfig | None = None,
    ):
        """Initialize knowledge builder.

        Args:
            vector_store: Vector store for storing chunks
            config: Knowledge builder configuration
        """
        self.vector_store = vector_store
        self.config = config or KnowledgeBuilderConfig()

        # Initialize components
        self.text_splitter = RecursiveTextSplitter(config=self.config.chunking)

        # Use service backend for local embeddings and openai backend for other providers
        if self.config.embedding.provider == "local":
            self.embedder = EmbedderFactory.create(
                backend="service",
                service_url=self.config.embedding.base_url,
                batch_size=self.config.embedding.batch_size,
                batch_delay=self.config.batch_delay,
            )
        else:
            self.embedder = EmbedderFactory.create(
                backend="openai",
                model=self.config.embedding.model,
                api_key=self.config.embedding.api_key,
                base_url=self.config.embedding.base_url,
                batch_size=self.config.embedding.batch_size,
                batch_delay=self.config.batch_delay,
            )

        self._build_status = BuildStatus(status="idle")
        self._lock = asyncio.Lock()

    async def build_from_documents(
        self, documents: list[Document], rebuild: bool = False
    ) -> BuildStatus:
        """Build knowledge base from documents.

        Args:
            documents: List of documents to build from
            rebuild: If True, clear existing data before building

        Returns:
            BuildStatus with progress information
        """
        async with self._lock:
            try:
                self._build_status = BuildStatus(
                    status="running",
                    total_documents=len(documents),
                    processed_documents=0,
                    total_chunks=0,
                    start_time=datetime.now().isoformat(),
                )

                if rebuild:
                    logger.info("Clearing existing knowledge base...")
                    await self.vector_store.clear()

                all_chunks = []
                for i, doc in enumerate(documents):
                    try:
                        chunks = await self._process_document(doc)
                        all_chunks.extend(chunks)
                        self._build_status.processed_documents = i + 1
                        self._build_status.total_chunks = len(all_chunks)
                        logger.info(
                            f"Processed document {i + 1}/{len(documents)}: {doc.id}, "
                            f"generated {len(chunks)} chunks"
                        )
                    except Exception as e:
                        error_msg = f"Error processing document {doc.id}: {str(e)}"
                        logger.error(error_msg)
                        self._build_status.errors.append(error_msg)

                if all_chunks:
                    logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
                    await self.vector_store.add_chunks(all_chunks)

                self._build_status.status = "completed"
                self._build_status.end_time = datetime.now().isoformat()
                logger.info(
                    f"Knowledge base build completed: {self._build_status.total_chunks} chunks from "
                    f"{self._build_status.processed_documents} documents"
                )

            except Exception as e:
                self._build_status.status = "failed"
                self._build_status.errors.append(str(e))
                self._build_status.end_time = datetime.now().isoformat()
                logger.error(f"Knowledge base build failed: {str(e)}")
                raise

            return self._build_status

    async def add_documents(self, documents: list[Document]) -> BuildStatus:
        """Add documents to existing knowledge base.

        Args:
            documents: List of documents to add

        Returns:
            BuildStatus with progress information
        """
        return await self.build_from_documents(documents, rebuild=False)

    async def get_build_status(self) -> BuildStatus:
        """Get current build status.

        Returns:
            Current build status
        """
        return self._build_status

    async def _process_document(self, document: Document) -> list[Chunk]:
        """Process a single document into chunks with embeddings.

        Args:
            document: Document to process

        Returns:
            List of processed chunks
        """
        chunk_texts = self.text_splitter.split_text(document.content, document.metadata)

        embeddings = await self.embedder.embed_texts(chunk_texts)

        chunks = []
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings, strict=False)):
            chunk_id = self._generate_chunk_id(document.id, i)
            chunk = Chunk(
                id=chunk_id,
                document_id=document.id,
                content=text,
                chunk_index=i,
                metadata={
                    **(document.metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                },
                embedding=embedding,
            )
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID.

        Args:
            document_id: Document ID
            chunk_index: Chunk index

        Returns:
            Unique chunk ID
        """
        content = f"{document_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
