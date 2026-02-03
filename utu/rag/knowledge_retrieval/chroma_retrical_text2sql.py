"""Generate demo course data and store in ChromaDB.

This script creates 10 course documents with metadata and stores them in ChromaDB.
Each course includes:
- Course title and detailed content
- Created timestamp
- Certification level (company/department/personal)
- Popularity score (0-100)
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import random
import json
import shutil
from typing import List, Dict, Any
from tqdm import tqdm

# Add current directory to path for local_embedder import
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utu.rag import Document, VectorStoreConfig, Chunk
from utu.rag.storage import VectorStoreFactory
from utu.rag.knowledge_builder import RecursiveTextSplitter
from utu.rag.config import ChunkingConfig
from utu.utils.log import get_logger
from ..embeddings.factory import EmbedderFactory

logger = get_logger("utu.rag.chroma_retrival_text2sql")

class CourseSearcher:
    """Course searcher with metadata filtering."""

    def __init__(self, collection_name: str = "demo_knowledge_base", vector_save_path: str = None):
        """Initialize course searcher.
        Args:
            collection_name: ChromaDB collection name
            vector_save_path: Path to the vector store directory
        """
        data_dir = Path(vector_save_path)
        vector_store_config = VectorStoreConfig(
            backend="chroma",
            collection_name=collection_name,
            persist_directory=str(data_dir),
            distance_metric="cosine",
        )

        self.vector_store = VectorStoreFactory.create(vector_store_config)
        self.embedder = self._init_embedder()

        # Cache for query embeddings to avoid redundant embedding calls
        self._embedding_cache = {}  # {query_text: embedding_vector}

    def _init_embedder(self):
        """Initialize embedder with hardcoded service configuration."""
        # Hardcoded embedding configuration (from meta_retrieval.yaml)
        backend = "service"
        service_url = os.getenv("UTU_EMBEDDING_URL")  # e.g., http://9.206.34.16:8081
        batch_size = 16
        
        embedder_params = {
            "service_url": service_url,
            "batch_size": batch_size,
        }
        
        embedder = EmbedderFactory.create(backend=backend, **embedder_params)
        return embedder

    def clear_embedding_cache(self):
        """Clear the embedding cache. Call this when starting a new query session."""
        self._embedding_cache.clear()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search courses with filters.

        Args:
            query: Search query text
            top_k: Number of results to return
        Returns:
            List of search results with metadata
        """
        # Check cache first to avoid redundant embedding
        if query in self._embedding_cache:
            query_embedding = self._embedding_cache[query]
        else:
            query_embedding = await self.embedder.embed_query(query)
            self._embedding_cache[query] = query_embedding

        chroma_filters = None
        if filter_conditions:
            if len(filter_conditions) == 1:
                chroma_filters = filter_conditions[0]
            else:
                chroma_filters = {"$and": filter_conditions}
        
        # chroma_filters = {"$and": chroma_filters}
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k, 
            filters=chroma_filters,
        )

        processed_results = []
        for chunk, score in results:
            temp_dict = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "score": score,
            }
            processed_results.append(temp_dict)
        return processed_results




def main():
    """Main entry point."""
    # vector_save_path = "/Users/_jie-wang_/Desktop/agent/git_project/youtu-rag-alpha-v1/rag_data/vector_store"
    # collection_name = "kb_wj_1_20251210_192006"
    # searcher = CourseSearcher(collection_name, vector_save_path)
    # top_docs = asyncio.run(searcher.search(
    #     query="北京小学",
    #     top_k=5,
    #     ))
    # print(json.dumps(top_docs, ensure_ascii=False, indent=4))

    vector_save_path = "/Users/_jie-wang_/Desktop/agent/git_project/youtu-rag-alpha-v1-wj/rag_data/vector_store"
    collection_name = "kb_mysql_test2_20251211_213324"
    searcher = CourseSearcher(collection_name, vector_save_path)
    top_docs = asyncio.run(searcher.search(
        query="在20_2024年2月11日全渠道库存（含云仓）.xlsx中的Z0K0库存表，K000库存中，Z0K0库存表中手机类的占比，比K000库存表中手机类的占比高还是低，两个占比之和是否超出100%",
        top_k=5,
        ))
    print(json.dumps(top_docs, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()
