"""Service-based Reranker implementation."""

import logging
from typing import List

import requests

from ..base import BaseReranker, RetrievalResult
from ..utils import make_request_with_retry

logger = logging.getLogger(__name__)


class ServiceReranker(BaseReranker):
    """Reranker using custom reranking service (e.g., self-hosted model)."""

    def __init__(
        self,
        service_url: str,
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
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized ServiceReranker with URL: {self.service_url}")

        # Test connection
        self._check_service_health()

    def _check_service_health(self):
        """Check if reranking service is healthy."""
        try:
            response = requests.get(f"{self.service_url}/model_id", timeout=5)
            response.raise_for_status()
            model_id = response.json()
            logger.info(f"✓ Reranking service is healthy. Model ID: {model_id}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to reranking service at {self.service_url}")
            logger.error("  Please check if the service is running and the URL is correct")
            raise ConnectionError(f"Reranking service unreachable: {self.service_url}")
        except requests.exceptions.Timeout:
            logger.error(f"✗ Reranking service timeout at {self.service_url}")
            raise TimeoutError(f"Reranking service timeout: {self.service_url}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to reranking service: {str(e)}")
            raise

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results based on relevance to query.

        Args:
            query: Query text
            results: List of retrieval results to rerank
            top_k: Optional number of top results to return (None returns all)

        Returns:
            Reranked list of retrieval results with updated scores and ranks
        """
        if not results:
            return results

        top_k = top_k or len(results)

        documents = [result.chunk.content for result in results]

        logger.info("-" * 60)
        logger.info(f"🔄 Starting Local Document Reranking")
        logger.info(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"   Number of documents: {len(documents)}")
        logger.info(f"   Top K: {top_k}")
        logger.info(f"   Service URL: {self.service_url}")

        if documents:
            preview = documents[0][:100] + "..." if len(documents[0]) > 100 else documents[0]
            logger.info(f"   First document preview: {preview}")
        logger.info("-" * 60)

        try:
            payload = {
                "query": query,
                "documents": documents,
                "top_k": top_k,
            }

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
                score = rank_result.get("score") or rank_result.get("relevance_score")

                original_result = results[index]

                reranked_result = RetrievalResult(
                    chunk=original_result.chunk,
                    score=score,  # Use rerank score
                    rank=len(reranked_results) + 1,
                )

                reranked_results.append(reranked_result)

            logger.info("-" * 60)
            logger.info(f"✅ Successfully reranked {len(reranked_results)} documents")
            if reranked_results:
                logger.info(f"   Top score: {reranked_results[0].score:.4f}")
                logger.info(f"   Lowest score: {reranked_results[-1].score:.4f}")
            logger.info("-" * 60)

            return reranked_results

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Error reranking documents: {str(e)}")
            logger.error("=" * 60)
            # Return original results if reranking fails
            return results[:top_k]
