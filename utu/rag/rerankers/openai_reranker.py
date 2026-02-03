"""OpenAI-compatible Reranker implementation.

Supports all reranking services using OpenAI-compatible /rerank API:
- Jina AI (https://api.jina.ai/v1/rerank)
- Cohere (via OpenAI-compatible proxies)
- Self-hosted rerankers with OpenAI-compatible interfaces
"""

import asyncio
import logging
from typing import Any

import httpx

from ..base import BaseReranker, RetrievalResult

logger = logging.getLogger(__name__)


class OpenAIReranker(BaseReranker):
    """Reranker using OpenAI-compatible reranking API.

    This reranker works with any service that implements the standard rerank API:
    - Request: POST /rerank with {"model": "...", "query": "...", "documents": [...], "top_n": N}
    - Response: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}

    Examples:
        - Jina AI: base_url="https://api.jina.ai/v1"
        - Self-hosted: base_url="http://your-server:8080/v1"
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
        base_url: str = "https://api.jina.ai/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize OpenAI-compatible reranker.

        Args:
            api_key: API key (optional for some self-hosted services)
            model: Reranker model name
                Examples:
                - jina-reranker-v2-base-multilingual
                - jina-reranker-v1-base-en
                - rerank-3 (Cohere)
                - bce-reranker-base (self-hosted)
            base_url: API base URL (without /rerank endpoint)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(
            f"Initialized OpenAIReranker: model={self.model}, base_url={self.base_url}"
        )

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

        try:
            documents = [result.chunk.content for result in results]

            rerank_scores = await self._call_rerank_api(
                query=query,
                documents=documents,
                top_k=top_k,
            )

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
                f"Reranked {len(results)} results to top {len(reranked_results)} "
                f"using {self.model}"
            )
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original results if reranking fails
            return results[:top_k]

    async def _call_rerank_api(
        self,
        query: str,
        documents: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Call rerank API with retry logic.

        Args:
            query: Query string
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of rerank results with scores

        Raises:
            httpx.HTTPError: Raised if all retries failed
        """
        if self.base_url.endswith("/rerank"):
            url = self.base_url
        else:
            url = f"{self.base_url}/rerank"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()

                    data = response.json()

                    results = data.get("results", [])

                    logger.info(
                        f"Rerank API returned {len(results)} results for query: "
                        f"'{query[:50]}{'...' if len(query) > 50 else ''}'"
                    )

                    return results

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                last_exception = e

                # Retryable errors
                if status_code in (502, 503, 429):
                    error_names = {502: "Bad Gateway", 503: "Service Unavailable", 429: "Rate Limit"}
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries}: "
                        f"{status_code} {error_names.get(status_code, 'Error')}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                else:
                    # Non-retryable error
                    logger.error(f"HTTP Error {status_code}: {str(e)}")
                    raise

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: Request timeout. "
                    f"Retrying in {self.retry_delay}s..."
                )

            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries}: Connection error. "
                    f"Retrying in {self.retry_delay}s..."
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        logger.error(f"All {self.max_retries} attempts failed")
        raise last_exception
