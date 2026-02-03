"""MinIO object storage monitoring."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class MinIOMonitor:
    """Monitor MinIO object storage health and metrics."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = True,
    ):
        """Initialize MinIO monitor.

        Args:
            endpoint: MinIO server endpoint (e.g., 'localhost:9000')
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name to monitor
            secure: Use HTTPS if True, HTTP if False
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.secure = secure
        self.client = None

    def _init_client(self) -> bool:
        """Initialize MinIO client.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            from minio import Minio

            self.client = Minio(
                self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=self.secure
            )
            return True
        except ImportError:
            logger.error("minio package not installed. Install with: pip install minio")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {str(e)}")
            return False

    def check_health(self) -> dict[str, Any]:
        """Check MinIO storage health.

        Returns:
            Health status dictionary
        """
        health = {
            "type": "minio",
            "is_healthy": False,
            "timestamp": datetime.now().isoformat(),
            "endpoint": self.endpoint,
            "bucket": self.bucket_name,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            if not self._init_client():
                health["errors"].append("Failed to initialize MinIO client")
                return health

            if not self.client.bucket_exists(self.bucket_name):
                health["errors"].append(f"Bucket '{self.bucket_name}' does not exist")
                return health

            objects = list(self.client.list_objects(self.bucket_name, recursive=True))

            total_size = sum(obj.size for obj in objects)
            object_count = len(objects)

            health["metrics"]["object_count"] = object_count
            health["metrics"]["total_size_bytes"] = total_size
            health["metrics"]["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            health["metrics"]["total_size_gb"] = round(total_size / (1024 * 1024 * 1024), 2)

            try:
                policy = self.client.get_bucket_policy(self.bucket_name)
                health["metrics"]["has_policy"] = bool(policy)
            except Exception:
                health["metrics"]["has_policy"] = False

            try:
                tags = self.client.get_bucket_tags(self.bucket_name)
                health["metrics"]["tags"] = dict(tags) if tags else {}
            except Exception:
                health["metrics"]["tags"] = {}

            if object_count == 0:
                health["warnings"].append("No objects found in bucket")

            if total_size > 10 * 1024 * 1024 * 1024:  # > 10GB
                health["warnings"].append(
                    f"Large bucket size: {total_size / (1024**3):.2f} GB"
                )

            health["is_healthy"] = True
            logger.info(f"MinIO health check passed: {self.endpoint}/{self.bucket_name}")

        except ImportError:
            health["errors"].append("minio package not installed")
        except Exception as e:
            logger.error(f"MinIO health check failed: {str(e)}")
            health["errors"].append(str(e))

        return health

    def get_metrics(self) -> dict[str, Any]:
        """Get detailed MinIO metrics.

        Returns:
            Metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "type": "minio",
            "endpoint": self.endpoint,
            "bucket": self.bucket_name,
        }

        try:
            if not self._init_client():
                metrics["error"] = "Failed to initialize MinIO client"
                return metrics

            if not self.client.bucket_exists(self.bucket_name):
                metrics["error"] = f"Bucket '{self.bucket_name}' does not exist"
                return metrics

            objects = list(self.client.list_objects(self.bucket_name, recursive=True))

            total_size = 0
            object_types = {}
            largest_objects = []

            for obj in objects:
                total_size += obj.size

                if "." in obj.object_name:
                    ext = obj.object_name.split(".")[-1].lower()
                    object_types[ext] = object_types.get(ext, 0) + 1

                largest_objects.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    }
                )

            largest_objects.sort(key=lambda x: x["size"], reverse=True)
            largest_objects = largest_objects[:10]

            metrics["object_count"] = len(objects)
            metrics["total_size_bytes"] = total_size
            metrics["total_size_gb"] = round(total_size / (1024**3), 2)
            metrics["object_types"] = object_types
            metrics["largest_objects"] = largest_objects

            try:
                versioning = self.client.get_bucket_versioning(self.bucket_name)
                metrics["versioning_enabled"] = versioning.status == "Enabled" if versioning else False
            except Exception:
                metrics["versioning_enabled"] = False

            try:
                encryption = self.client.get_bucket_encryption(self.bucket_name)
                metrics["encryption_enabled"] = bool(encryption)
            except Exception:
                metrics["encryption_enabled"] = False

        except ImportError:
            metrics["error"] = "minio package not installed"
        except Exception as e:
            logger.error(f"Error collecting MinIO metrics: {str(e)}")
            metrics["error"] = str(e)

        return metrics

    def list_objects(self, prefix: str = "", max_objects: int = 100) -> list[dict[str, Any]]:
        """List objects in bucket.

        Args:
            prefix: Object name prefix to filter
            max_objects: Maximum number of objects to return

        Returns:
            List of object information dictionaries
        """
        objects_info = []

        try:
            if not self._init_client():
                return objects_info

            objects = self.client.list_objects(
                self.bucket_name, prefix=prefix, recursive=True
            )

            for i, obj in enumerate(objects):
                if i >= max_objects:
                    break

                objects_info.append(
                    {
                        "name": obj.object_name,
                        "size": obj.size,
                        "size_mb": round(obj.size / (1024 * 1024), 2),
                        "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                        "etag": obj.etag,
                    }
                )

        except Exception as e:
            logger.error(f"Error listing MinIO objects: {str(e)}")

        return objects_info

    def cleanup_old_objects(self, days: int = 30) -> dict[str, Any]:
        """Delete objects older than specified days.

        Args:
            days: Delete objects older than this many days

        Returns:
            Cleanup result dictionary
        """
        result = {
            "success": False,
            "deleted_count": 0,
            "space_freed_mb": 0,
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            if not self._init_client():
                result["errors"].append("Failed to initialize MinIO client")
                return result

            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days)

            objects = list(self.client.list_objects(self.bucket_name, recursive=True))
            objects_to_delete = []
            total_size = 0

            for obj in objects:
                if obj.last_modified and obj.last_modified < cutoff_date:
                    objects_to_delete.append(obj.object_name)
                    total_size += obj.size

            if objects_to_delete:
                errors = self.client.remove_objects(
                    self.bucket_name, objects_to_delete
                )
                for error in errors:
                    result["errors"].append(str(error))

            result["success"] = len(result["errors"]) == 0
            result["deleted_count"] = len(objects_to_delete)
            result["space_freed_mb"] = round(total_size / (1024 * 1024), 2)

            logger.info(
                f"Cleanup completed: deleted {len(objects_to_delete)} objects, "
                f"freed {result['space_freed_mb']} MB"
            )

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            result["errors"].append(str(e))

        return result
