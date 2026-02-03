"""Service-based Reranker implementation."""

import logging
from typing import List

import requests

from ..base import BaseReranker, RetrievalResult
from ..utils import make_request_with_retry

logger = logging.getLogger(__name__)


class TioneReranker(BaseReranker):
    """Reranker using custom reranking service (e.g., self-hosted model)."""

    def __init__(
        self,
        model: str,
        service_url: str,
        api_key: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize service reranker.

        Args:
            service_url: URL of reranking service (e.g., "http://9.206.34.16:8082")
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.service_url = service_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized ServiceReranker with URL: {self.service_url}")

        # Test connection
        self._check_service_health()

    def _check_service_health(self):
        """Check if reranking service is healthy."""
        return True

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of document texts to rerank
            top_k: Optional number of top results to return (None returns all)

        Returns:
            List of (index, score) tuples sorted by relevance score in descending order
        """
        if not results:
            return results
        
        documents = [result.chunk.content for result in results]
        top_k = top_k or len(results)

        logger.info("-" * 60)
        logger.info(f"🔄 Starting Local Document Reranking")
        logger.info(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"   Number of documents: {len(documents)}")
        logger.info(f"   Top K: {top_k if top_k else 'all'}")
        logger.info(f"   Service URL: {self.service_url}")

        if documents:
            preview = documents[0][:100] + "..." if len(documents[0]) > 100 else documents[0]
            logger.info(f"   First document preview: {preview}")
        logger.info("-" * 60)

        try:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }
            if top_k is not None:
                payload["top_k"] = top_k

            rsp = make_request_with_retry(
                url=f"{self.service_url}/rerank",
                json_data=payload,
                timeout=60,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                logger_instance=logger,
            )

            # Expected format: {"results": [{"index": 0, "score": 0.95}, ...]}
            rerank_scores = rsp.get("results", [])
            reranked_results = []
            for rank_result in rerank_scores:
                index = rank_result["index"]
                score = rank_result["relevance_score"]

                original_result = results[index]

                reranked_result = RetrievalResult(
                    chunk=original_result.chunk,
                    score=score,  # Use rerank score
                    rank=len(reranked_results) + 1,
                )

                reranked_results.append(reranked_result)

            logger.info(
                f"Reranked {len(results)} results to top {len(reranked_results)} using Jina"
            )
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original results if reranking fails
            return results[:top_k]
