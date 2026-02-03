"""File-related models"""
from typing import Any, Dict, List
from pydantic import BaseModel


class FileMetadata(BaseModel):
    """Metadata of a file"""
    name: str
    size: int
    last_modified: str
    content_type: str
    metadata: dict[str, Any]


class MetadataImportRow(BaseModel):
    """Metadata import row"""
    filename: str
    etag: str
    metadata: Dict[str, Any]


class MetadataImportResult(BaseModel):
    """Metadata import result"""
    total_rows: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]]
