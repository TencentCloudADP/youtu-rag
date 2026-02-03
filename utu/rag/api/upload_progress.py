"""Track file upload progress."""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum


class UploadStatus(str, Enum):
    PENDING = "pending"                          # Waiting for processing
    UPLOADING = "uploading"                      # Uploading
    OCR_PROCESSING = "ocr_processing"            # OCR processing
    METADATA_EXTRACTING = "metadata_extracting"  # Metadata extraction
    UPLOADING_TO_MINIO = "uploading_to_minio"    # Uploading to MinIO
    CHUNK_PROCESSING = "chunk_processing"        # Chunk-level recognition processing
    COMPLETED = "completed"                      # Completed
    FAILED = "failed"                            # Failed


class UploadProgressTracker:
    """The tracker for file upload progress (memory-based, suitable for single-machine deployment)."""

    def __init__(self):
        # Format: {task_id: {status, progress, message, result, error, ...}}
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, filename: str) -> str:
        """Create new task and return task ID."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "task_id": task_id,
            "filename": filename,
            "status": UploadStatus.PENDING,
            "progress": 0,
            "message": "等待处理...",
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return task_id

    def update_progress(
        self,
        task_id: str,
        status: UploadStatus,
        progress: int,
        message: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Update task progress."""
        if task_id not in self._tasks:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[ProgressTracker] Task {task_id} not found in tracker!")
            return

        old_progress = self._tasks[task_id].get("progress", 0)
        self._tasks[task_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        if result:
            self._tasks[task_id]["result"] = result
        if error:
            self._tasks[task_id]["error"] = error

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[Progress] Task {task_id[:8]}: {old_progress}% → {progress}% ({status}) - {message}")

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the progress of a task."""
        return self._tasks.get(task_id)

    def cleanup_completed_tasks(self, max_age_hours: int = 1):
        """Clean up old completed tasks (to avoid memory leaks)."""
        now = datetime.now(timezone.utc)
        to_delete = []

        for task_id, task in self._tasks.items():
            if task["status"] in [UploadStatus.COMPLETED, UploadStatus.FAILED]:
                updated_at = datetime.fromisoformat(task["updated_at"])
                age_hours = (now - updated_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_delete.append(task_id)

        for task_id in to_delete:
            del self._tasks[task_id]


# Create a global singleton
upload_tracker = UploadProgressTracker()
