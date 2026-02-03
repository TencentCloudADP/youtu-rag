"""Knowledge retrieval implementations."""

import logging
from typing import Any

from ..base import BaseEmbedder, BaseRetriever, BaseVectorStore, RetrievalResult
from ..config import RetrieverConfig
from ..rerankers.factory import RerankerFactory


logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Vector-based retrieval using semantic similarity."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        config: RetrieverConfig | None = None,
    ):
        """Initialize vector retriever.

        Args:
            vector_store: Vector store for retrieving chunks
            embedder: Embedding generator for query encoding
            config: Retriever configuration
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or RetrieverConfig()

        # Initialize reranker if enabled
        self.reranker = None
        if self.config.enable_reranking:
            self.reranker = RerankerFactory.create(
                backend="auto",
                model=self.config.reranker_model
            )

    async def retrieve(self, query: str, top_k: int | None = None, **kwargs) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string
            top_k: Number of results to return (overrides config if provided)
            kwargs: Additional arguments (filters, similarity_threshold, etc.)

        Returns:
            List of retrieval results
        """
        top_k = top_k or self.config.top_k
        filters = kwargs.get("filters")
        similarity_threshold = kwargs.get("similarity_threshold", self.config.similarity_threshold)

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k * 2 if self.reranker else top_k, filters=filters
        )

        # Convert to RetrievalResult objects
        retrieval_results = []
        for i, (chunk, score) in enumerate(results):
            # Apply threshold filtering only if threshold > 0
            # When threshold <= 0.0, return all results (no filtering)
            # This handles cases where ChromaDB returns negative similarity scores (distance > 1.0)
            if similarity_threshold <= 0.0 or score >= similarity_threshold:
                retrieval_results.append(RetrievalResult(chunk=chunk, score=score, rank=i + 1))

        # Apply reranking if enabled
        if self.reranker and retrieval_results:
            retrieval_results = await self.reranker.rerank(
                query=query, results=retrieval_results, top_k=top_k
            )

        return retrieval_results[:top_k]

    async def batch_retrieve(
        self, queries: list[str], top_k: int | None = None, **kwargs
    ) -> list[list[RetrievalResult]]:
        """Batch retrieve for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query
            **kwargs: Additional arguments (filters, similarity_threshold, etc.)

        Returns:
            List of retrieval results for each query
        """
        results = []
        for query in queries:
            query_results = await self.retrieve(query=query, top_k=top_k, **kwargs)
            results.append(query_results)
        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining vector and keyword search."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        config: RetrieverConfig | None = None,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: Vector store for retrieving chunks
            embedder: Embedding generator for query encoding
            config: Retriever configuration
        """
        self.vector_retriever = VectorRetriever(
            vector_store=vector_store, embedder=embedder, config=config
        )
        self.config = config or RetrieverConfig()

    async def retrieve(self, query: str, top_k: int | None = None, **kwargs) -> list[RetrievalResult]:
        """Retrieve relevant chunks using hybrid approach.

        Note: This is a simplified implementation. A full hybrid retrieval would
        include keyword-based search (e.g., BM25) and combine scores.

        Args:
            query: Query string
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of retrieval results
        """
        # For now, delegate to vector retriever
        # TODO: Implement keyword search and score fusion
        return await self.vector_retriever.retrieve(query=query, top_k=top_k, **kwargs)

    async def batch_retrieve(
        self, queries: list[str], top_k: int | None = None, **kwargs
    ) -> list[list[RetrievalResult]]:
        """Batch retrieve for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query
            **kwargs: Additional arguments

        Returns:
            List of retrieval results for each query
        """
        return await self.vector_retriever.batch_retrieve(queries=queries, top_k=top_k, **kwargs)
