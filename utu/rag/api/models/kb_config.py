"""Models related to knowledge base configurations"""
from typing import Any, Optional, List
from pydantic import BaseModel


class ToolConfig(BaseModel):
    """Tool configuration"""
    enabled: bool
    settings: dict[str, Any]


class KBConfiguration(BaseModel):
    """Knowledge base configuration"""
    tools: dict[str, ToolConfig]
    selectedFiles: list[str] = []
    selectedQAFiles: list[str] = []
    dbConnections: list[dict[str, Any]] = []


class KBConfigurationUpdate(BaseModel):
    """Request for updating knowledge base configuration"""
    configuration: KBConfiguration


class KBBuildRequest(BaseModel):
    """Request for building knowledge base"""
    force_rebuild: bool = False
    file_filter: Optional[list[str]] = None


class KBBuildResponse(BaseModel):
    """Response to building knowledge base"""
    status: str
    message: str
    kb_id: int
    kb_name: str
    total_files: int
    processed_files: int
    skipped_files: int = 0  # Number of skipped files (unchanged)
    total_chunks: int
    errors: list[str] = []


class QAValidationResult(BaseModel):
    """QA file validation result"""
    valid: bool
    filename: str
    sheet_name: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[list[str]] = None
    errors: list[str] = []
    sample_data: Optional[list[dict[str, str]]] = None


class DBConnectionTestRequest(BaseModel):
    """Request for testing database connection"""
    db_type: str  # mysql, postgresql, sqlite
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    file_path: Optional[str] = None  # For SQLite


class DBConnectionTestResponse(BaseModel):
    """Response to testing database connection"""
    success: bool
    message: str
    tables: list[str] = []
    error: Optional[str] = None
