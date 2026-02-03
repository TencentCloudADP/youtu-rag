"""Embedding service"""
import logging
import os
from typing import List, Tuple

from ..models.embedding import ModelInfo

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service interface for embedding."""
    
    @staticmethod
    async def embed_texts(texts: List[str], backend: str = "auto", config: dict = None) -> Tuple[List[List[float]], str]:
        """Generating embeddings for multiple texts.
        
        Args:
            texts: List of texts.
            backend: Type of the backend.
            config: Configuration dictionary.
            
        Returns:
            A tuple, (embeddings, backend_used).
        """
        from ...embeddings import create_embedder
        
        if not texts:
            raise ValueError("No texts provided")
        
        config = config or {}
        embedder = create_embedder(backend=backend, **config)
        
        embeddings = await embedder.embed_texts(texts)
        
        backend_used = type(embedder).__name__.replace("Embedder", "").lower()
        
        logger.info(f"Generated {len(embeddings)} embeddings using {backend_used}")
        
        return embeddings, backend_used
    
    @staticmethod
    async def embed_query(query: str, backend: str = "auto", config: dict = None) -> Tuple[List[float], str]:
        """Generate embedding for a single query.
        
        Args:
            query: The input query string.
            backend: Type of the backend.
            config: Configuration dictionary.
            
        Returns:
            A tuple, (embedding, backend_used).
        """
        from ...embeddings import create_embedder
        
        if not query or not query.strip():
            raise ValueError("Empty query provided")
        
        config = config or {}
        embedder = create_embedder(backend=backend, **config)
        
        embedding = await embedder.embed_query(query)
        
        backend_used = type(embedder).__name__.replace("Embedder", "").lower()
        
        logger.info(f"Generated query embedding using {backend_used}")
        
        return embedding, backend_used
    
    @staticmethod
    def list_available_models() -> List[ModelInfo]:
        """List available embedding models and their configurations.
        
        Returns:
            List of available models with their configurations and status.
        """
        from ...embeddings import create_embedder
        
        models = []
        
        # Check unified UTU_EMBEDDING configuration
        embedding_url = os.getenv("UTU_EMBEDDING_URL")
        embedding_model = os.getenv("UTU_EMBEDDING_MODEL", "youtu-embedding-2b")
        api_key = os.getenv("UTU_API_KEY")
        
        if embedding_url:
            try:
                embedder = create_embedder("auto")
                
                # Determine embedding dimensions based on model name
                dimensions = 2304  # Default for youtu-embedding-2b
                if "hunyuan" in embedding_model.lower():
                    dimensions = 1024
                elif "openai" in embedding_model.lower() or "text-embedding" in embedding_model.lower():
                    dimensions = 1536
                
                models.append(ModelInfo(
                    backend="unified",
                    available=True,
                    config={
                        "base_url": embedding_url,
                        "model": embedding_model,
                        "type": "Unified Embedding Service (OpenAI-compatible)",
                        "dimensions": dimensions,
                        "has_api_key": bool(api_key)
                    }
                ))
            except Exception as e:
                models.append(ModelInfo(
                    backend="unified",
                    available=False,
                    config={"base_url": embedding_url, "model": embedding_model},
                    error=str(e)
                ))


        if not models:
            models.append(ModelInfo(
                backend="none",
                available=False,
                config={},
                error="No embedding backend configured. Set UTU_EMBEDDING_URL, UTU_API_KEY, and UTU_EMBEDDING_MODEL"
            ))
        
        return models
    
    @staticmethod
    async def test_connection(backend: str, config: dict) -> dict:
        """Test connection to the embedding backend.
        
        Args:
            backend: Type of the backend.
            config: Configuration dictionary.
            
        Returns:
            Connection test results.
        """
        from ...embeddings import create_embedder
        
        try:
            embedder = create_embedder(backend=backend, **config)
            
            # Test with a simple query
            test_query = "Test connection"
            embedding = await embedder.embed_query(test_query)
            
            embedding_dim = len(embedding) if embedding else 0
            
            backend_name = type(embedder).__name__.replace("Embedder", "")
            
            return {
                "success": True,
                "backend": backend,
                "message": f"Successfully connected to {backend_name} backend",
                "details": {
                    "embedding_dimension": embedding_dim,
                    "test_query": test_query
                }
            }
        
        except ValueError as e:
            logger.error(f"Configuration error during connection test: {str(e)}")
            return {
                "success": False,
                "backend": backend,
                "message": f"Configuration error: {str(e)}"
            }
        except ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            return {
                "success": False,
                "backend": backend,
                "message": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Test connection error: {str(e)}")
            return {
                "success": False,
                "backend": backend,
                "message": f"Connection test failed: {str(e)}"
            }
