"""Health checking for storage systems."""

import logging
from datetime import datetime

from ..base import BaseVectorStore, HealthStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Check health of storage systems."""

    def __init__(self, vector_store: BaseVectorStore):
        """Initialize health checker.

        Args:
            vector_store: Vector store to check
        """
        self.vector_store = vector_store

    async def check_health(self) -> HealthStatus:
        """Check storage health.

        Returns:
            Health status
        """
        errors = []
        warnings = []
        is_healthy = True

        try:
            # Check if we can count documents
            total_chunks = await self.vector_store.count()

            backend = self.vector_store.config.backend
            collection_name = self.vector_store.config.collection_name

            if total_chunks == 0:
                warnings.append("Vector store is empty")

            health_status = HealthStatus(
                is_healthy=is_healthy,
                backend=backend,
                collection_name=collection_name,
                total_chunks=total_chunks,
                last_check_time=datetime.now().isoformat(),
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_status = HealthStatus(
                is_healthy=False,
                backend="unknown",
                collection_name="unknown",
                last_check_time=datetime.now().isoformat(),
                errors=[str(e)],
            )

        return health_status
