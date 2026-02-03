"""Service-based Embedding implementation."""

import base64
import logging
from typing import List

import numpy as np
import requests

from ..base import BaseEmbedder
from ..utils import make_request_with_retry

logger = logging.getLogger(__name__)


class ServiceEmbedder(BaseEmbedder):
    """Embedder using custom embedding service (e.g., self-hosted model)."""

    def __init__(
        self,
        service_url: str,
        batch_size: int = 64,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize service embedder.

        Args:
            service_url: URL of embedding service (e.g., "http://9.206.34.16:8081")
            batch_size: Batch size for processing texts
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.service_url = service_url.rstrip("/")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized ServiceEmbedder with URL: {self.service_url}")

        # Test connection
        self._check_service_health()

    def _check_service_health(self):
        """Check if embedding service is healthy."""
        try:
            response = requests.get(f"{self.service_url}/model_id", timeout=5)
            response.raise_for_status()
            model_id = response.json()
            logger.info(f"✓ Embedding service is healthy. Model ID: {model_id}")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to embedding service at {self.service_url}")
            logger.error("  Please check if the service is running and the URL is correct")
            raise ConnectionError(f"Embedding service unreachable: {self.service_url}")
        except requests.exceptions.Timeout:
            logger.error(f"✗ Embedding service timeout at {self.service_url}")
            raise TimeoutError(f"Embedding service timeout: {self.service_url}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to embedding service: {str(e)}")
            raise

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

        logger.info("-" * 60)
        logger.info(f"📝 Starting Local Embedding Generation")
        logger.info(f"   Total texts to embed: {len(texts)}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Service URL: {self.service_url}")

        # Show a preview of first text
        if texts:
            preview = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            logger.info(f"   First text preview: {preview}")
        logger.info("-" * 60)

        embeddings = []

        try:
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            for i, batch in enumerate(self._batched(texts, self.batch_size)):
                logger.info(f"🔄 Processing batch {i + 1}/{total_batches} with {len(batch)} texts...")

                # Call embedding service with retry
                rsp = make_request_with_retry(
                    url=f"{self.service_url}/embed_docs",
                    json_data={"docs": list(batch)},
                    timeout=60,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    logger_instance=logger,
                )

                # Decode base64 encoded embeddings
                embed = np.frombuffer(
                    base64.b64decode(rsp["embedding"].encode("ascii")),
                    dtype=np.float32,
                ).reshape(rsp["shape"]).tolist()

                embeddings.extend(embed)
                logger.info(f"   ✓ Batch {i + 1} completed. Shape: {rsp['shape']}, Dimension: {rsp['shape'][1] if len(rsp['shape']) > 1 else 'N/A'}")

            logger.info("-" * 60)
            logger.info(f"✅ Successfully generated {len(embeddings)} embeddings for {len(texts)} texts")
            logger.info(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            logger.info("-" * 60)
            return embeddings

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Error generating embeddings: {str(e)}")
            logger.error("=" * 60)
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            logger.info("-" * 60)
            logger.info(f"🔍 Generating Local Embedding for Query")
            preview = query[:100] + "..." if len(query) > 100 else query
            logger.info(f"   Query: {preview}")
            logger.info(f"   Service URL: {self.service_url}/embed_query")
            logger.info("-" * 60)

            # Call embedding service for query with retry
            rsp = make_request_with_retry(
                url=f"{self.service_url}/embed_query",
                json_data={"query": query},
                timeout=30,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
                logger_instance=logger,
            )

            # Decode base64 encoded embedding
            embed = np.frombuffer(
                base64.b64decode(rsp["embedding"].encode("ascii")),
                dtype=np.float32,
            ).reshape(rsp["shape"]).tolist()

            logger.info(f"✅ Successfully generated query embedding")
            logger.info(f"   Embedding dimension: {len(embed)}")
            logger.info("-" * 60)
            return embed

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Error generating query embedding: {str(e)}")
            logger.error("=" * 60)
            raise
