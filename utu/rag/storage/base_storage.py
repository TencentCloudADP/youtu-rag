"""Base storage implementation and factory."""

import logging
from typing import Any

from ..base import BaseVectorStore
from ..config import VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create(config: VectorStoreConfig) -> BaseVectorStore:
        """Create a vector store instance based on configuration.

        Args:
            config: Vector store configuration

        Returns:
            Vector store instance

        Raises:
            ValueError: If backend is not supported
        """
        backend = config.backend.lower()

        if backend == "chroma":
            from .implementations.chroma_store import ChromaVectorStore

            return ChromaVectorStore(config=config)

        # elif backend == "faiss":
        #     from .implementations.faiss_store import FAISSVectorStore

        #     return FAISSVectorStore(config=config)

        else:
            msg = f"Unsupported vector store backend: {backend}"
            raise ValueError(msg)

    @staticmethod
    def list_backends() -> list[str]:
        """List available vector store backends.

        Returns:
            List of backend names
        """
        # return ["chroma", "faiss"]
        return ["chroma"]
