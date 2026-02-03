"""Factory for creating embedder instances."""

import logging
import os
from typing import Any

from ..base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .service_embedder import ServiceEmbedder

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def create(backend: str = "auto", **kwargs) -> BaseEmbedder:
        """Create embedder instance based on backend type.

        Args:
            backend: Embedder backend type:
                - "auto": Auto-detect from environment variables (UTU_EMBEDDING_URL)
                - "openai": OpenAI-compatible embedding API (requires api_key)
                - "service": Local embedding service (no api_key needed)
            **kwargs: Additional parameters for the embedder

        Returns:
            Embedder instance

        Examples:
            ```
            # Auto-detect from environment (UTU_EMBEDDING_URL, UTU_API_KEY, UTU_EMBEDDING_MODEL)
            embedder = EmbedderFactory.create("auto")

            # OpenAI-compatible API (requires api_key)
            embedder = EmbedderFactory.create(
                "openai",
                model="text-embedding-3-small",
                base_url="https://api.openai.com/v1",
                api_key="sk-..."
            )

            # Local embedding service (no api_key needed)
            embedder = EmbedderFactory.create(
                "service",
                service_url="http://9.206.34.16:8081"
            )
            ```
        """
        if backend == "auto":
            return EmbedderFactory._create_auto(**kwargs)
        elif backend == "openai":
            return EmbedderFactory._create_openai(**kwargs)
        elif backend == "service":
            return EmbedderFactory._create_service(**kwargs)
        else:
            raise ValueError(
                f"Unknown embedder backend: {backend}. "
                f"Supported backends: auto, openai, service"
            )

    @staticmethod
    def _create_auto(**kwargs) -> BaseEmbedder:
        """Auto-detect embedder from environment variables.

        Uses unified configuration:
        - UTU_EMBEDDING_URL: Embedding service endpoint (OpenAI-compatible API)
        - UTU_EMBEDDING_API_KEY: API key (optional, depends on service)
        - UTU_EMBEDDING_MODEL: Model name (e.g., youtu-embedding-2b, hunyuan-embedding)
        """
        # Priority 1: Check for unified UTU_EMBEDDING_URL
        embedding_url = os.getenv("UTU_EMBEDDING_URL")
        if embedding_url:
            model = kwargs.pop("model", None) or os.getenv("UTU_EMBEDDING_MODEL", "youtu-embedding-2b")
            # Try UTU_EMBEDDING_API_KEY first, then fall back to UTU_API_KEY
            api_key = os.getenv("UTU_EMBEDDING_API_KEY")

            logger.info(f"Auto-detected unified embedding config: model={model}, url={embedding_url}")

            # Use OpenAI-compatible interface
            # Note: api_key can be None for services that don't require authentication
            return OpenAIEmbedder(
                model=model,
                base_url=embedding_url,
                api_key=api_key,
                **kwargs
            )

        # No valid configuration found
        raise ValueError(
            "Could not auto-detect embedder configuration. "
            "Please set UTU_EMBEDDING_URL environment variable."
        )

    @staticmethod
    def _create_service(**kwargs) -> ServiceEmbedder:
        """Create service-based embedder (Youtu-Embedding-2B)."""
        service_url = kwargs.pop("service_url", None)

        # Remove batch_delay since ServiceEmbedder doesn't support it (local service doesn't need rate limiting)
        kwargs.pop("batch_delay", None)

        if not service_url:
            service_url = os.getenv("UTU_EMBEDDING_URL")

        if not service_url:
            raise ValueError(
                "service_url is required for service embedder. "
                "Provide it as parameter or set UTU_EMBEDDING_URL/EMBEDDING_URL environment variable."
            )
        logger.info(f"🚀    Service URL: {service_url}")

        logger.info("=" * 60)
        logger.info("🚀 Creating Local Embedding Service (ServiceEmbedder)")
        logger.info(f"   Service URL: {service_url}")
        logger.info(f"   Batch Size: {kwargs.get('batch_size', 64)}")
        logger.info(f"   Type: Local service (no API key required)")
        logger.info("=" * 60)

        return ServiceEmbedder(service_url=service_url, **kwargs)

    @staticmethod
    def _create_openai(**kwargs) -> OpenAIEmbedder:
        """Create OpenAI-compatible embedder.

        Uses unified configuration or falls back to OpenAI-specific env vars.
        """
        # Try unified config first
        api_key = kwargs.pop("api_key", None) or os.getenv("UTU_EMBEDDING_API_KEY")
        model = kwargs.pop("model", None) or os.getenv("UTU_EMBEDDING_MODEL") or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        base_url = kwargs.pop("base_url", None) or os.getenv("UTU_EMBEDDING_URL")

        logger.info(f"Creating OpenAI-compatible embedder: model={model}, base_url={base_url}")

        return OpenAIEmbedder(model=model, api_key=api_key, base_url=base_url, **kwargs)


def create_embedder(backend: str = "auto", **kwargs) -> BaseEmbedder:
    """Convenience function to create embedder instance.

    Args:
        backend: Embedder backend type (auto, service, openai)
        **kwargs: Additional parameters for the embedder

    Returns:
        Embedder instance

    Examples:
        # Auto-detect from environment
        embedder = create_embedder()

        # Service-based embedder
        embedder = create_embedder("service", service_url="http://9.206.34.16:8081")

        # OpenAI embedder
        embedder = create_embedder("openai", model="text-embedding-3-large")
    """
    return EmbedderFactory.create(backend, **kwargs)
