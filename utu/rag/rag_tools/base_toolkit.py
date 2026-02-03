"""Base toolkit for RAG operations with shared utility methods."""

import logging
from typing import Optional

from ...config import ToolkitConfig
from ...tools.base import AsyncBaseToolkit
from ..storage import VectorStoreFactory
from ..embeddings.factory import EmbedderFactory
from ..knowledge_retrieval import VectorRetriever
from ..config import VectorStoreConfig, RetrieverConfig
from ..api.database import get_db, KnowledgeBase

logger = logging.getLogger(__name__)


class BaseRAGToolkit(AsyncBaseToolkit):
    """Base class for RAG toolkits with shared utility methods.
    
    Provides common functionality for:
    - Knowledge base collection name retrieval
    - Embedder parameter building
    - Retriever creation
    """

    def __init__(self, config: ToolkitConfig = None):
        """Initialize base RAG toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        self.embedding_config = self.config.config.get("embedding", {})
        self.vector_store_base_config = self.config.config.get("vector_store", {})
        logger.info(f"Embedding config: {self.embedding_config}")
        logger.info(f"Vector store base config: {self.vector_store_base_config}")

        self._embedder_cache = None  # Shared embedder for all KBs
        self._vector_store_cache = {}  # Cache by collection_name

    async def _get_kb_collection_name(self, kb_id: int) -> tuple[str, str]:
        db = next(get_db())
        try:
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
            logger.info(f"Find KB '{kb.name}' (collection: {kb.collection_name})")
            if not kb:
                raise ValueError(f"Knowledge base with ID {kb_id} not found")
            return kb.collection_name, kb.name
        finally:
            db.close()

    def _build_embedder_params(self, embedding_config: Optional[dict] = None) -> dict:
        config = embedding_config or self.embedding_config
        backend = config.get("backend", "openai")
        logger.info(f"build_embedder_params = '{backend}'")

        embedder_params = {}

        if backend == "service":
            embedder_params["service_url"] = config.get("base_url")
            embedder_params["batch_size"] = config.get("batch_size", 16)
        else:
            embedder_params["model"] = config.get("model")
            embedder_params["api_key"] = config.get("api_key")
            embedder_params["base_url"] = config.get("base_url")
            embedder_params["batch_size"] = config.get("batch_size", 16)

        return embedder_params

    def _get_or_create_embedder(self):
        if self._embedder_cache is None:
            backend = self.embedding_config.get("backend", "openai")
            logger.info(f"Creating and caching embedder with backend '{backend}'")
            embedder_params = self._build_embedder_params()
            self._embedder_cache = EmbedderFactory.create(backend=backend, **embedder_params)
        return self._embedder_cache

    def _get_or_create_vector_store(self, collection_name: str, persist_directory: str):
        if collection_name not in self._vector_store_cache:
            logger.info(f"Creating and caching vector store for collection '{collection_name}'")
            vector_store_config = VectorStoreConfig(
                backend=self.vector_store_base_config.get("backend", "chroma"),
                persist_directory=persist_directory,
                collection_name=collection_name,
                distance_metric=self.vector_store_base_config.get("distance_metric", "cosine"),
            )
            self._vector_store_cache[collection_name] = VectorStoreFactory.create(vector_store_config)
        else:
            logger.info(f"Using cached vector store for collection '{collection_name}'")
        return self._vector_store_cache[collection_name]

    async def _create_retriever(
        self,
        kb_id: int,
        top_k: int,
        embedder=None,
        persist_directory: Optional[str] = None,
        similarity_threshold: float = 0.0,
    ) -> VectorRetriever:
        """Create a retriever for a specific knowledge base.

        Args:
            kb_id: Knowledge base ID
            top_k: Number of results to retrieve
            embedder: Pre-initialized embedder (if None, uses cached embedder)
            persist_directory: Vector store persist directory (uses config default if None)
            similarity_threshold: Minimum similarity threshold for results

        Returns:
            Configured VectorRetriever
        """
        collection_name, kb_name = await self._get_kb_collection_name(kb_id)

        persist_dir = persist_directory or self.vector_store_base_config.get(
            "persist_directory", "./rag_data/vector_store"
        )

        vector_store = self._get_or_create_vector_store(collection_name, persist_dir)

        if embedder is None:
            embedder = self._get_or_create_embedder()
        else:
            logger.info(f"Using provided embedder for KB '{kb_name}'")

        retriever_config = RetrieverConfig(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_reranking=False
        )

        retriever = VectorRetriever(
            vector_store=vector_store, embedder=embedder, config=retriever_config
        )

        logger.info(f"Get retriever for KB '{kb_name}' (collection: {collection_name})")
        return retriever
