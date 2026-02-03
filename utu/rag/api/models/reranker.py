"""Reranker-related models"""
from typing import Any, List, Optional
from pydantic import BaseModel


class RerankRequest(BaseModel):
    """Request for reranking"""
    query: str
    documents: List[str]
    top_n: Optional[int] = 3
    return_documents: Optional[bool] = False


class RerankResult(BaseModel):
    """Single reranking result"""
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Response to reranking"""
    results: List[RerankResult]
    count: int
    model_used: str


class RerankerModelInfo(BaseModel):
    """Reranker model information"""
    model: str
    available: bool
    config: dict[str, Any]
    error: Optional[str] = None


class RerankerTestConnectionRequest(BaseModel):
    """Request for testing connection to a reranker model"""
    api_key: str
    model: Optional[str] = "jina-reranker-v2-base-multilingual"


class RerankerTestConnectionResponse(BaseModel):
    """Response to testing connection to a reranker model"""
    success: bool
    model: str
    message: str
    details: Optional[dict[str, Any]] = None
