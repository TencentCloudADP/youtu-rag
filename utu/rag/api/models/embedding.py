"""Embedding-related models"""
from typing import Any, List, Optional
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    """Request for embedding texts"""
    texts: List[str]
    backend: Optional[str] = "auto"  # auto, service, openai
    config: Optional[dict[str, Any]] = None


class EmbedQueryRequest(BaseModel):
    """Request for embedding a single query"""
    query: str
    backend: Optional[str] = "auto"
    config: Optional[dict[str, Any]] = None


class EmbedResponse(BaseModel):
    """Response to embedding texts"""
    embeddings: List[List[float]]
    count: int
    backend_used: str


class EmbedQueryResponse(BaseModel):
    """Response to embedding a single query"""
    embedding: List[float]
    backend_used: str


class ModelInfo(BaseModel):
    """Model info"""
    backend: str
    available: bool
    config: dict[str, Any]
    error: Optional[str] = None


class TestConnectionRequest(BaseModel):
    """Request for testing connection"""
    backend: str
    config: dict[str, Any]


class TestConnectionResponse(BaseModel):
    """Response to testing connection"""
    success: bool
    backend: str
    message: str
    details: Optional[dict[str, Any]] = None
