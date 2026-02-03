"""Reranker service."""
import logging
import os
from typing import List, Tuple

from ..models.reranker import RerankerModelInfo

logger = logging.getLogger(__name__)


class RerankerService:
    """Service interface for reranking."""
    
    @staticmethod
    async def rerank_documents(
        query: str,
        documents: List[str],
        top_n: int = 3,
        return_documents: bool = False,
        api_key: str = None,
        model: str = None
    ) -> Tuple[List[dict], str]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Query text.
            documents: List of documents.
            top_n: Return top N results.
            return_documents: Whether to return document content.
            api_key: Jina API key.
            model: Model name.
            
        Returns:
            A tuple, (results, model_used).
        """
        from ...rerankers import create_reranker
        from ...base import RetrievalResult, Chunk
        
        if not documents:
            raise ValueError("No documents provided")

        reranker = create_reranker(backend="jina")
        
        retrieval_results = [
            RetrievalResult(
                chunk=Chunk(
                    id=f"doc_{i}",
                    document_id=f"doc_{i}",
                    content=doc,
                    chunk_index=i,
                    metadata={}
                ),
                score=0.0,
                rank=i + 1
            )
            for i, doc in enumerate(documents)
        ]
        
        reranked_results = await reranker.rerank(
            query=query,
            results=retrieval_results,
            top_n=top_n,
            return_documents=return_documents,
        )
        
        results = [
            {
                "index": documents.index(result.chunk.content),
                "relevance_score": result.score,
                "document": result.chunk.content if return_documents else None,
            }
            for result in reranked_results
        ]
        
        logger.info(f"Reranked {len(documents)} documents to top {len(results)}")
        
        return results, model
    
    @staticmethod
    def list_available_models() -> List[RerankerModelInfo]:
        """List available reranker models and their configurations.
        
        Returns:
            List of available reranker models and their statuses.
        """
        models = []
        
        # Check Jina reranker
        jina_api_key = os.getenv("UTU_RERANKER_API_KEY")
        jina_model = os.getenv("UTU_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
        jina_base_url = os.getenv("UTU_RERANKER_BASE_URL", "https://api.jina.ai/v1")
        
        if jina_api_key:
            models.append(
                RerankerModelInfo(
                    model=jina_model,
                    available=True,
                    config={
                        "base_url": jina_base_url,
                        "type": "Jina Reranker API",
                        "multilingual": "multilingual" in jina_model,
                    },
                )
            )
        else:
            models.append(
                RerankerModelInfo(
                    model=jina_model,
                    available=False,
                    config={"base_url": jina_base_url},
                    error="JINA_API_KEY not configured",
                )
            )
        
        if not models:
            models.append(
                RerankerModelInfo(
                    model="none",
                    available=False,
                    config={},
                    error="No reranker configured. Set JINA_API_KEY in .env file",
                )
            )
        
        return models
    
    @staticmethod
    async def test_connection(api_key: str, model: str = "jina-reranker-v2-base-multilingual") -> dict:
        """Test connection to Jina reranker API.
        
        Args:
            api_key: Jina API key.
            model: Model name.
            
        Returns:
            Connection test result.
        """
        from ...rerankers import create_reranker
        from ...base import RetrievalResult, Chunk
        
        try:
            reranker = create_reranker(backend="jina")
            
            # Use a simple query for testing
            test_query = "Test connection"
            test_docs = ["This is a test document"]
            
            test_results = [
                RetrievalResult(
                    chunk=Chunk(
                        id="test_0",
                        document_id="test_0",
                        content=doc,
                        chunk_index=0,
                        metadata={}
                    ),
                    score=0.0,
                    rank=1
                )
                for doc in test_docs
            ]
            
            # Test rerank
            reranked = await reranker.rerank(query=test_query, results=test_results)
            
            return {
                "success": True,
                "model": model,
                "message": f"Successfully connected to Jina reranker",
                "details": {
                    "model": model,
                    "test_query": test_query,
                    "relevance_score": reranked[0].score if reranked else 0.0,
                },
            }
        
        except ValueError as e:
            logger.error(f"Configuration error during connection test: {str(e)}")
            return {
                "success": False,
                "model": model,
                "message": f"Configuration error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Test connection error: {str(e)}")
            return {
                "success": False,
                "model": model,
                "message": f"Connection test failed: {str(e)}",
            }
