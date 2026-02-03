"""Utils for hash computations."""
import hashlib
import json
from typing import Any


def calculate_metadata_hash(metadata: dict[str, Any]) -> str:
    """Calculate the MD5 hash of metadata for version detection.
    
    The hash is stored in the kb_source_configs table (not in MinIO) to track metadata changes and trigger rebuilds when metadata is updated.
    
    Args:
        metadata: Metadata dictionary from MinIO (only original fields).
        
    Returns:
        MD5 hash string.
    """
    metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(metadata_json.encode('utf-8')).hexdigest()
