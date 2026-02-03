"""Factory for creating reranker instances."""

import logging
import os
from typing import Any

from ..base import BaseReranker
from .openai_reranker import OpenAIReranker
from .service_reranker import ServiceReranker
from .tione_reranker import TioneReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """Factory for creating reranker instances."""

    @staticmethod
    def create(backend: str = "auto", **kwargs) -> BaseReranker:
        """Create reranker instance based on backend type.

        Args:
            backend: Reranker backend type:
                - "auto": Auto-detect from environment variables (UTU_RERANK_URL)
                - "openai": OpenAI-compatible reranking API (requires api_key)
                - "service": Local reranking service (no api_key needed)
                - "tione": Tione platform reranking service
                - "jina": Jina AI reranking service (requires api_key)
            **kwargs: Additional parameters for the reranker

        Returns:
            Reranker instance

        Examples:
            ```
            # Auto-detect from environment (UTU_RERANK_URL, UTU_API_KEY, UTU_RERANK_MODEL)
            reranker = RerankerFactory.create("auto")

            # OpenAI-compatible API (requires api_key)
            reranker = RerankerFactory.create(
                "openai",
                model="rerank-3",
                base_url="https://api.openai.com/v1",
                api_key="sk-..."
            )

            # Local reranking service (no api_key needed)
            reranker = RerankerFactory.create(
                "service",
                service_url="http://9.206.34.16:8082"
            )

            # Tione platform reranking service
            reranker = RerankerFactory.create(
                "tione",
                service_url="http://tione-rerank-service:8082"
            )

            # Jina AI reranking service
            reranker = RerankerFactory.create(
                "jina",
                api_key="jina_xxx",
                model="jina-reranker-v2-base-multilingual"
            )
            ```
        """
        if backend == "auto":
            return RerankerFactory._create_auto(**kwargs)
        elif backend == "openai":
            return RerankerFactory._create_openai(**kwargs)
        elif backend == "service":
            return RerankerFactory._create_service(**kwargs)
        elif backend == "tione":
            return RerankerFactory._create_tione(**kwargs)
        elif backend == "jina":
            return RerankerFactory._create_jina(**kwargs)
        else:
            raise ValueError(
                f"Unknown reranker backend: {backend}. "
                f"Supported backends: auto, openai, service, tione, jina"
            )

    @staticmethod
    def _create_auto(**kwargs) -> BaseReranker:
        """Auto-detect reranker from environment variables.

        Uses unified configuration:
        - UTU_RERANK_URL: Reranking service endpoint (OpenAI-compatible API)
        - UTU_RERANK_API_KEY: API key (optional, depends on service)
        - UTU_RERANK_MODEL: Model name (e.g., rerank-3, bce-reranker-base)
        """
        rerank_url = os.getenv("UTU_RERANKER_URL")
        if rerank_url:
            model = kwargs.pop("model", None) or os.getenv("UTU_RERANKER_MODEL", "bce-reranker-base")
            # Try UTU_RERANK_API_KEY first, then fall back to UTU_API_KEY
            api_key = os.getenv("UTU_RERANKER_API_KEY")

            logger.info(f"Auto-detected unified reranking config: model={model}, url={rerank_url}")

            # Note: api_key can be None for services that don't require authentication
            return OpenAIReranker(
                model=model,
                base_url=rerank_url,
                api_key=api_key,
                **kwargs
            )

        raise ValueError(
            "Could not auto-detect reranker configuration. "
            "Please set UTU_RERANK_URL environment variable."
        )

    @staticmethod
    def _create_service(**kwargs) -> ServiceReranker:
        """Create service-based reranker."""
        service_url = kwargs.pop("service_url", None)

        if not service_url:
            service_url = os.getenv("UTU_RERANKER_URL")

        if not service_url:
            raise ValueError(
                "service_url is required for service reranker. "
                "Provide it as parameter or set UTU_RERANK_URL environment variable."
            )
        logger.info(f"🚀    Service URL: {service_url}")

        return ServiceReranker(service_url=service_url, **kwargs)

    @staticmethod
    def _create_openai(**kwargs) -> OpenAIReranker:
        """Create OpenAI-compatible reranker.

        Uses unified configuration or falls back to OpenAI-specific env vars.
        """
        api_key = kwargs.pop("api_key", None) or os.getenv("UTU_RERANKER_API_KEY")
        model = kwargs.pop("model", None) or os.getenv("UTU_RERANKER_MODEL", "rerank-3")
        base_url = kwargs.pop("base_url", None) or os.getenv("UTU_RERANKER_URL")

        logger.info(f"Creating OpenAI-compatible reranker: model={model}, base_url={base_url}")

        return OpenAIReranker(model=model, api_key=api_key, base_url=base_url, **kwargs)

    @staticmethod
    def _create_tione(**kwargs) -> TioneReranker:
        """Create Tione platform reranker."""
        api_key = kwargs.pop("api_key", None) or os.getenv("UTU_RERANKER_API_KEY")
        model = kwargs.pop("model", None) or os.getenv("UTU_RERANKER_MODEL")
        service_url = kwargs.pop("service_url", None) or os.getenv("UTU_RERANKER_URL")

        if not service_url:
            raise ValueError(
                "service_url is required for Tione reranker. "
                "Provide it as parameter or set UTU_RERANKER_URL environment variable."
            )
        logger.info(f"🚀    Tione Service URL: {service_url}")

        return TioneReranker(
            service_url=service_url,
            model=model,
            api_key=api_key,
            **kwargs
        )

    @staticmethod
    def _create_jina(**kwargs) -> OpenAIReranker:
        """Create Jina AI reranker."""
        api_key = kwargs.pop("api_key", None) or os.getenv("UTU_RERANKER_API_KEY")
        model = kwargs.pop("model", None) or os.getenv("UTU_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
        base_url = kwargs.pop("base_url", None) or os.getenv("UTU_RERANKER_URL", "https://api.jina.ai/v1")

        if not api_key:
            raise ValueError(
                "api_key is required for Jina reranker. "
                "Provide it as parameter or set UTU_RERANKER_API_KEY environment variable."
            )

        logger.info(f"Creating Jina reranker: model={model}, base_url={base_url}")

        return OpenAIReranker(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs
        )


def create_reranker(backend: str = "auto", **kwargs) -> BaseReranker:
    """Convenience function to create reranker instance.

    Args:
        backend: Reranker backend type (auto, service, openai, tione, jina)
        **kwargs: Additional parameters for the reranker

    Returns:
        Reranker instance

    Examples:
        ```
        # Auto-detect from environment
        reranker = create_reranker()

        # Service-based reranker
        reranker = create_reranker("service", service_url="http://9.206.34.16:8082")

        # Tione reranker
        reranker = create_reranker("tione", service_url="http://tione-rerank:8082")

        # OpenAI reranker
        reranker = create_reranker("openai", model="rerank-3")

        # Jina AI reranker
        reranker = create_reranker("jina", api_key="jina_xxx")
        ```
    """
    return RerankerFactory.create(backend, **kwargs)
