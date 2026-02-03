"""Embedding routes, embedding vector generation and management"""
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from ..models.embedding import (
    EmbedRequest,
    EmbedQueryRequest,
    EmbedResponse,
    EmbedQueryResponse,
    ModelInfo,
    TestConnectionRequest,
    TestConnectionResponse,
)
from ..services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for multiple texts.
    
    Args:
        request: Embed request containing texts and configuration.
        
    Returns:
        All text embeddings.
        
    Example:
        ```
        POST /api/embedding/embed
        {
            "texts": ["Hello world", "RAG system"],
            "backend": "auto"
        }
        ```
    """
    try:
        embeddings, backend_used = await EmbeddingService.embed_texts(
            texts=request.texts,
            backend=request.backend,
            config=request.config
        )
        
        return EmbedResponse(
            embeddings=embeddings,
            count=len(embeddings),
            backend_used=backend_used
        )
    
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")


@router.post("/embed_query", response_model=EmbedQueryResponse)
async def embed_query(request: EmbedQueryRequest):
    """Generate embedding for a single query.
    
    Args:
        request: Embed query request containing query and configuration.
        
    Returns:
        Query embedding.
        
    Example:
        ```
        POST /api/embedding/embed_query
        {
            "query": "What is RAG?",
            "backend": "auto"
        }
        ```
    """
    try:
        embedding, backend_used = await EmbeddingService.embed_query(
            query=request.query,
            backend=request.backend,
            config=request.config
        )
        
        return EmbedQueryResponse(
            embedding=embedding,
            backend_used=backend_used
        )
    
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query embedding error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available embedding models and their configurations.
    
    Returns:
        Available embedding backends and their status.
        
    Example:
        ```
        GET /api/embedding/models
        ```
    """
    try:
        models = EmbeddingService.list_available_models()
        return models
    except Exception as e:
        logger.error(f"List models error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-connection", response_model=TestConnectionResponse)
async def test_connection(request: TestConnectionRequest):
    """Test connection to embedding backend.
    
    Args:
        request: Test connection request, containing backend and configuration.
        
    Returns:
        Test connection result.
        
    Example:
        ```
        POST /api/embedding/test-connection
        {
            "backend": "service",
            "config": {
                "service_url": "http://ip:port"
            }
        }
        ```
    """
    try:
        result = await EmbeddingService.test_connection(
            backend=request.backend,
            config=request.config
        )
        
        return TestConnectionResponse(**result)
    
    except Exception as e:
        logger.error(f"Test connection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
