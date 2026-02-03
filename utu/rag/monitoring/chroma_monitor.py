"""ChromaDB vector store monitoring."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ChromaMonitor:
    """Monitor ChromaDB vector store health and metrics."""

    def __init__(self, persist_directory: str):
        """Initialize ChromaDB monitor.

        Args:
            persist_directory: Path to ChromaDB persistent storage
        """
        self.persist_directory = persist_directory
        self.client = None

    def _init_client(self) -> bool:
        """Initialize ChromaDB client.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            return True
        except ImportError:
            logger.error("chromadb package not installed. Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            return False

    def check_health(self) -> dict[str, Any]:
        """Check ChromaDB storage health.

        Returns:
            Health status dictionary
        """
        health = {
            "type": "vector_store",
            "is_healthy": False,
            "timestamp": datetime.now().isoformat(),
            "persist_directory": self.persist_directory,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            if not self._init_client():
                health["errors"].append("Failed to initialize ChromaDB client")
                return health

            collections = self.client.list_collections()
            collection_count = len(collections)

            total_chunks = 0
            collection_stats = []

            for collection in collections:
                try:
                    count = collection.count()
                    total_chunks += count
                    collection_stats.append({
                        "name": collection.name,
                        "count": count,
                    })
                except Exception as e:
                    logger.warning(f"Failed to get count for collection {collection.name}: {e}")
                    collection_stats.append({
                        "name": collection.name,
                        "count": 0,
                        "error": str(e),
                    })

            collection_stats.sort(key=lambda x: x.get("count", 0), reverse=True)

            health["metrics"]["collection_count"] = collection_count
            health["metrics"]["total_chunks"] = total_chunks
            health["metrics"]["collections"] = collection_stats

            if collection_count == 0:
                health["warnings"].append("No collections found in ChromaDB")
            elif total_chunks == 0:
                health["warnings"].append("All collections are empty")

            health["is_healthy"] = True
            logger.info(f"ChromaDB health check passed: {self.persist_directory}")

        except ImportError:
            health["errors"].append("chromadb package not installed")
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {str(e)}")
            health["errors"].append(str(e))

        return health

    def get_metrics(self) -> dict[str, Any]:
        """Get detailed ChromaDB metrics.

        Returns:
            Metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "type": "vector_store",
            "persist_directory": self.persist_directory,
        }

        try:
            if not self._init_client():
                metrics["error"] = "Failed to initialize ChromaDB client"
                return metrics

            collections = self.client.list_collections()
            collection_details = []

            for collection in collections:
                try:
                    count = collection.count()
                    metadata = collection.metadata

                    sample = collection.peek(limit=1)
                    has_data = len(sample["ids"]) > 0

                    collection_details.append({
                        "name": collection.name,
                        "count": count,
                        "metadata": metadata,
                        "has_data": has_data,
                    })
                except Exception as e:
                    logger.warning(f"Failed to get details for collection {collection.name}: {e}")
                    collection_details.append({
                        "name": collection.name,
                        "error": str(e),
                    })

            metrics["collection_count"] = len(collections)
            metrics["collections"] = collection_details
            metrics["total_chunks"] = sum(c.get("count", 0) for c in collection_details)

        except ImportError:
            metrics["error"] = "chromadb package not installed"
        except Exception as e:
            logger.error(f"Error collecting ChromaDB metrics: {str(e)}")
            metrics["error"] = str(e)

        return metrics
