"""RAG Toolkit for integration with agents."""

import logging
from typing import Any

from ..config import ToolkitConfig
from ..tools.base import AsyncBaseToolkit, register_tool
from .base import Document
from .config import RAGConfig
from .knowledge_builder import KnowledgeBuilder
from .knowledge_retrieval import ContextAssembler, VectorRetriever
from .storage import VectorStoreFactory
from .monitoring import StorageMonitor
from .embeddings.factory import EmbedderFactory

logger = logging.getLogger(__name__)


class RAGToolkit(AsyncBaseToolkit):
    """RAG tools for knowledge building and retrieval."""

    def __init__(self, config: ToolkitConfig | dict | None = None):
        """Initialize RAG toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        rag_config_dict = self.config.config.get("rag_config", {})
        self.rag_config = RAGConfig(**rag_config_dict) if rag_config_dict else RAGConfig()

        self.vector_store = VectorStoreFactory.create(self.rag_config.vector_store)
        self.embedder = EmbedderFactory.create(
            backend="openai",
            model=self.rag_config.knowledge_builder.embedding.model,
            api_key=self.rag_config.knowledge_builder.embedding.api_key,
            base_url=self.rag_config.knowledge_builder.embedding.base_url,
            batch_size=self.rag_config.knowledge_builder.embedding.batch_size,
        )
        self.knowledge_builder = KnowledgeBuilder(
            vector_store=self.vector_store, config=self.rag_config.knowledge_builder
        )
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            config=self.rag_config.retriever,
        )
        self.context_assembler = ContextAssembler(max_context_length=4000)
        self.monitor = StorageMonitor(
            vector_store=self.vector_store, config=self.rag_config.monitor
        )

        logger.info(f"RAG Toolkit initialized with backend: {self.rag_config.vector_store.backend}")

    @register_tool
    async def build_knowledge_base(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None, rebuild: bool = False
    ) -> str:
        """Build or update knowledge base from texts.

        Args:
            texts: List of text strings to add to knowledge base
            metadatas: Optional list of metadata dicts for each text
            rebuild: If True, clear existing data before building

        Returns:
            Build status message
        """
        try:
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = Document(id=f"doc_{i}", content=text, metadata=metadata)
                documents.append(doc)

            build_status = await self.knowledge_builder.build_from_documents(
                documents, rebuild=rebuild
            )

            return (
                f"Knowledge base build {build_status.status}. "
                f"Processed {build_status.processed_documents} documents, "
                f"created {build_status.total_chunks} chunks."
            )

        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
            return f"Error building knowledge base: {str(e)}"

    @register_tool
    async def retrieve_knowledge(
        self, query: str, top_k: int = 5, format_style: str = "markdown"
    ) -> str:
        """Retrieve relevant knowledge for a query.

        Args:
            query: Query string
            top_k: Number of results to retrieve
            format_style: Format of returned context ("markdown", "plain", "json")

        Returns:
            Retrieved and formatted context
        """
        try:
            results = await self.retriever.retrieve(query=query, top_k=top_k)

            if not results:
                return "No relevant knowledge found for the query."

            context = self.context_assembler.assemble(
                results, include_metadata=True, format_style=format_style
            )

            if self.rag_config.monitor.enable_query_logging:
                await self.monitor.log_query(
                    query=query, latency_ms=0.0, result_count=len(results)
                )

            return context

        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return f"Error retrieving knowledge: {str(e)}"

    @register_tool
    async def get_knowledge_base_stats(self) -> str:
        """Get statistics about the knowledge base.

        Returns:
            Knowledge base statistics
        """
        try:
            total_chunks = await self.vector_store.count()
            health = await self.monitor.check_health()

            stats = (
                f"Knowledge Base Statistics:\n"
                f"- Backend: {self.rag_config.vector_store.backend}\n"
                f"- Collection: {self.rag_config.vector_store.collection_name}\n"
                f"- Total Chunks: {total_chunks}\n"
                f"- Health Status: {'Healthy' if health.is_healthy else 'Unhealthy'}\n"
            )

            if health.warnings:
                stats += f"- Warnings: {', '.join(health.warnings)}\n"

            return stats

        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {str(e)}")
            return f"Error getting knowledge base stats: {str(e)}"

    @register_tool
    async def clear_knowledge_base(self) -> str:
        """Clear all data from the knowledge base.

        Returns:
            Confirmation message
        """
        try:
            await self.vector_store.clear()
            return "Knowledge base cleared successfully."

        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return f"Error clearing knowledge base: {str(e)}"
