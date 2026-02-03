"""OpenAI Embeddings implementation."""

import asyncio
import logging
import os
from typing import List

from openai import AsyncOpenAI

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_delay: float = 3.0,
    ):
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
                - text-embedding-3-small (default, 1536 dims, cheaper)
                - text-embedding-3-large (3072 dims, better performance)
                - text-embedding-ada-002 (legacy, 1536 dims)
            api_key: OpenAI API key (defaults to UTU_EMBEDDING_API_KEY env var)
            base_url: Custom base URL for OpenAI-compatible services
            batch_size: Maximum batch size for API calls
            max_retries: Maximum number of retries on failure
            timeout: Request timeout in seconds
            batch_delay: Delay in seconds between batches to avoid rate limiting
        """
        self.model = model
        self.batch_size = batch_size
        self.batch_delay = batch_delay

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("UTU_EMBEDDING_API_KEY"),
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

        logger.info(
            f"Initialized OpenAIEmbedder with model: {self.model}, "
            f"batch_size: {self.batch_size}, batch_delay: {self.batch_delay}s"
        )

    def _batched(self, iterable, n):
        """Split iterable into batches of size n."""
        from itertools import islice

        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        try:
            # Calculate total batches for progress tracking
            batches = list(self._batched(texts, self.batch_size))
            total_batches = len(batches)

            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i + 1}/{total_batches} with {len(batch)} texts...")

                # Log request details
                logger.info(f"📤 Embedding API Request Details:")
                logger.info(f"  - Model: {self.model}")
                logger.info(f"  - Batch size: {len(batch)} texts")
                logger.info(f"  - Base URL: {self.client.base_url}")
                for idx, text in enumerate(batch):
                    text_preview = text[:200] if len(text) > 200 else text
                    logger.info(f"  - Text {idx + 1}: length={len(text)} chars, preview=\"{text_preview}...\"")

                # Retry logic for handling WAF blocks and rate limits
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Call OpenAI embedding API
                        response = await self.client.embeddings.create(
                            model=self.model, input=list(batch), encoding_format="float"
                        )

                        # Extract embeddings
                        batch_embeddings = [item.embedding for item in response.data]
                        embeddings.extend(batch_embeddings)
                        logger.info(f"✅ Successfully received {len(batch_embeddings)} embeddings from API")
                        break  # Success, exit retry loop

                    except Exception as e:
                        error_msg = str(e)
                        is_waf_block = "waf-static.tencent.com" in error_msg.lower() or "501page.html" in error_msg.lower()
                        is_rate_limit = "rate" in error_msg.lower() or "429" in error_msg

                        # Log error details
                        logger.error(f"❌ Embedding API error on batch {i + 1}:")
                        logger.error(f"  - Error type: {'WAF block' if is_waf_block else 'Rate limit' if is_rate_limit else 'Other'}")
                        logger.error(f"  - Batch size: {len(batch)} texts")
                        logger.error(f"  - Total chars in batch: {sum(len(t) for t in batch)}")
                        logger.error(f"  - Error message: {error_msg[:500]}")

                        if retry < max_retries - 1 and (is_waf_block or is_rate_limit):
                            # Exponential backoff: 5s, 10s, 20s
                            wait_time = 5 * (2 ** retry)
                            logger.warning(
                                f"⚠️ {'WAF block' if is_waf_block else 'Rate limit'} detected on batch {i + 1}. "
                                f"Retrying in {wait_time}s... (attempt {retry + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            # Last retry or non-retryable error
                            raise

                # Add delay between batches to avoid rate limiting (except for last batch)
                if i < total_batches - 1 and self.batch_delay > 0:
                    logger.debug(f"Waiting {self.batch_delay}s before next batch...")
                    await asyncio.sleep(self.batch_delay)

            logger.info(f"✓ Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"✗ Error generating embeddings: {str(e)}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            # Call OpenAI embedding API for single query
            response = await self.client.embeddings.create(
                model=self.model, input=query, encoding_format="float"
            )

            embedding = response.data[0].embedding

            logger.info(f"✓ Successfully generated embedding for query")
            return embedding

        except Exception as e:
            logger.error(f"✗ Error generating query embedding: {str(e)}")
            raise
