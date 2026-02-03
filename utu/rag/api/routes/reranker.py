"""Reranker routes"""
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from ..models.reranker import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankerModelInfo,
    RerankerTestConnectionRequest,
    RerankerTestConnectionResponse,
)
from ..services.reranker_service import RerankerService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Use Jina API to rerank documents based on query relevance.
    
    Args:
        request: Rerank request containing a query and multiple documents.
        
    Returns:
        Reranked documents with relevance scores.
        
    Example:
        ```
        POST /api/reranker/rerank
        {
            "query": "What is RAG?",
            "documents": ["doc1", "doc2", "doc3"],
            "top_n": 2
        }
        ```
    """
    try:
        results, model_used = await RerankerService.rerank_documents(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            return_documents=request.return_documents
        )
        
        return RerankResponse(
            results=[RerankResult(**r) for r in results],
            count=len(results),
            model_used=model_used
        )
    
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to rerank documents: {str(e)}")


@router.get("/models", response_model=List[RerankerModelInfo])
async def list_models():
    """List available reranker models and their configurations.
    
    Returns:
        List of available reranker models with their status.
        
    Example:
        ```
        GET /api/reranker/models
        ```
    """
    try:
        models = RerankerService.list_available_models()
        return models
    except Exception as e:
        logger.error(f"List models error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-connection", response_model=RerankerTestConnectionResponse)
async def test_connection(request: RerankerTestConnectionRequest):
    """Test connection to Jina reranker API.
    
    Args:
        request: Test connection request containing API key.
        
    Returns:
        Connection test result.
        
    Example:
        ```
        POST /api/reranker/test-connection
        {
            "api_key": "jina_xxx",
            "model": "jina-reranker-v2-base-multilingual"
        }
        ```
    """
    try:
        result = await RerankerService.test_connection(
            api_key=request.api_key,
            model=request.model
        )
        
        return RerankerTestConnectionResponse(**result)
    
    except Exception as e:
        logger.error(f"Test connection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
