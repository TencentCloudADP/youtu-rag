"""Vector memory store implementation for agent memory management.

Includes:
- MemoryVectorStore: ChromaDB-based vector store for memory storage
- EmbeddingService: Text embedding generation service
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None
    Settings = None

from utu.rag.base import BaseVectorStore, Chunk, BaseEmbedder
from utu.rag.config import VectorStoreConfig
from utu.rag.embeddings.factory import EmbedderFactory


logger = logging.getLogger(__name__)

# ============== Embedding Service ==============

class EmbeddingConfig(BaseModel):
    """Configuration for embedding service."""

    embedding_type: str = Field(default="local", description="Embedding type: 'local' or 'api'")
    url: str = Field(default="", description="Embedding service URL")
    model: str = Field(default="", description="Embedding model name")
    api_key: str | None = Field(default=None, description="API key for API-based embedding")
    batch_size: int = Field(default=32, description="Batch size for embedding requests")

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        return cls(
            embedding_type=os.getenv("UTU_EMBEDDING_TYPE", "local"),
            url=os.getenv("UTU_EMBEDDING_URL", "http://localhost:8081"),
            model=os.getenv("UTU_EMBEDDING_MODEL", "youtu-embedding-2b"),
            api_key=os.getenv("UTU_EMBEDDING_API_KEY"),
            batch_size=int(os.getenv("UTU_EMBEDDING_BATCH_SIZE", "32")),
        )


class EmbeddingService:
    """Service for generating text embeddings.

    Supports two modes:
    1. Local mode: Calls a local embedding service (e.g., youtu-embedding-2B)
    2. API mode: Calls an OpenAI-compatible embedding API (e.g., HunyuanEmbedding)

    Usage:
        ```python
        # Using default config from environment
        service = EmbeddingService()

        # Generate embeddings
        embeddings = await service.embed(["Hello world", "Another text"])
        ```
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """Initialize embedding service.

        Args:
            config: Optional configuration. If None, loads from environment.
        """
        self.config = config or EmbeddingConfig.from_env()
        self._embedder: BaseEmbedder | None = None

    @property
    def embedder(self) -> BaseEmbedder:
        """Lazy initialization of embedder."""
        if self._embedder is None:
            if self.config.embedding_type == "local":
                logger.info(f"Creating local embedder with URL: {self.config.url}")
                self._embedder = EmbedderFactory.create(
                    backend="service",
                    service_url=self.config.url,
                    batch_size=self.config.batch_size,
                )
            else:
                logger.info(f"Creating API embedder with URL: {self.config.url}")
                self._embedder = EmbedderFactory.create(
                    backend="openai",
                    model=self.config.model,
                    base_url=self.config.url,
                    api_key=self.config.api_key,
                    batch_size=self.config.batch_size,
                )
        return self._embedder

    async def close(self) -> None:
        """Close the embedder if needed."""
        # BaseEmbedder doesn't have a close method, so nothing to do
        pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        return await self.embedder.embed_texts(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self.embedder.embed_query(text)

    @property
    def dimension(self) -> int | None:
        """Get embedding dimension (available after first embed call)."""
        # Not implemented in base embedder interface
        return None

    async def __aenter__(self) -> "EmbeddingService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

# ============== Memory Vector Store ==============
class MemoryVectorStore(BaseVectorStore):
    """ChromaDB-based vector store specialized for agent memory management.

    Features:
        1. Store vectorized representations of memory content
        2. Support semantic similarity retrieval
        3. Create collections by user_id or memory_type
        4. Provide persistent storage capability

    This store extends the base vector store with memory-specific features:
        - User isolation via collection naming
        - Memory type filtering (episodic, procedural, semantic, working)
        - Importance and recency scoring
        - Tool sequence tracking for procedural memories
    """

    def __init__(self, config: VectorStoreConfig | None = None, persist_directory: str | None = None):
        """Initialize Memory vector store.

        Args:
            config: Optional vector store configuration
            persist_directory: Directory for persistent storage (overrides config)

        Raises:
            ImportError: If chromadb is not installed
        """
        if not CHROMA_AVAILABLE:
            msg = "chromadb is not installed. Install it with: pip install chromadb"
            raise ImportError(msg)

        self.config = config
        self._persist_directory = persist_directory or (config.persist_directory if config else "./data/memory")

        if self._persist_directory:
            self.client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Initialized persistent Memory store at: {self._persist_directory}")
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            logger.info("Initialized in-memory Memory store")

        self._collections: dict[str, chromadb.Collection] = {}
        self._default_collection_name = config.collection_name if config else "agent_memory"

    def get_collection_name(self, user_id: str, memory_type: str | None = None) -> str:
        """Generate collection name based on user_id and optional memory_type.

        Args:
            user_id: User identifier for isolation
            memory_type: Optional memory type for further segmentation

        Returns:
            Collection name string
        """
        base = f"memory_{user_id}"
        if memory_type:
            return f"{base}_{memory_type}"
        return base

    def get_or_create_collection(self, collection_name: str | None = None) -> chromadb.Collection:
        """Get or create a ChromaDB collection.

        Args:
            collection_name: Optional collection name, uses default if not provided

        Returns:
            ChromaDB Collection object
        """
        name = collection_name or self._default_collection_name
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(f"Created/loaded memory collection: {name}")
        return self._collections[name]

    async def add_chunks(self, chunks: list[Chunk], collection_name: str | None = None) -> None:
        """Add chunks to the memory store.

        Args:
            chunks: List of chunks to add
            collection_name: Optional collection name
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(collection_name)

        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
            }
            # Add memory-specific metadata, filtering None values
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if v is not None:
                        if isinstance(v, datetime):
                            metadata[k] = v.isoformat()
                        elif isinstance(v, (list, dict)):
                            import json
                            metadata[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            metadata[k] = v
            metadatas.append(metadata)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"Added {len(chunks)} memory chunks to collection {collection.name}")

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar memory chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            collection_name: Optional collection name

        Returns:
            List of (chunk, similarity_score) tuples
        """
        collection = self.get_or_create_collection(collection_name)

        where = None
        if filters:
            if any(key.startswith("$") for key in filters.keys()):
                where = filters
            elif any(
                isinstance(value, dict) and any(k.startswith("$") for k in value.keys())
                for value in filters.values()
            ):
                where = filters
            else:
                where = {key: {"$eq": value} for key, value in filters.items()}

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

        chunks_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                embedding = results["embeddings"][0][i] if results["embeddings"] else None
                distance = results["distances"][0][i]

                similarity = 1.0 - distance

                parsed_metadata = self._deserialize_metadata(metadata)

                chunk = Chunk(
                    id=chunk_id,
                    document_id=parsed_metadata.get("document_id", ""),
                    content=document,
                    chunk_index=parsed_metadata.get("chunk_index", 0),
                    metadata=parsed_metadata,
                    embedding=embedding,
                )
                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def _deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON-encoded metadata fields.

        Args:
            metadata: Raw metadata from ChromaDB

        Returns:
            Parsed metadata dict
        """
        import json

        result = {}
        json_fields = {"tool_sequence", "metadata"}

        for k, v in metadata.items():
            if k in json_fields and isinstance(v, str):
                try:
                    result[k] = json.loads(v)
                except json.JSONDecodeError:
                    result[k] = v
            else:
                result[k] = v
        return result

    async def search_memories(
        self,
        query_embedding: list[float],
        user_id: str,
        memory_type: str | None = None,
        session_id: str | None = None,
        top_k: int = 10,
        min_importance: float = 0.0,
        include_outdated: bool = False,
    ) -> list[tuple[Chunk, float]]:
        """Search memories with memory-specific filters.

        Args:
            query_embedding: Query embedding vector
            user_id: User ID to search within
            memory_type: Optional memory type filter
            session_id: Optional session ID filter
            top_k: Maximum number of results
            min_importance: Minimum importance score
            include_outdated: Whether to include low success_rate procedural memories

        Returns:
            List of (chunk, similarity_score) tuples
        """
        collection_name = self.get_collection_name(user_id, memory_type)

        filter_conditions = []
        if session_id:
            filter_conditions.append({"session_id": {"$eq": session_id}})
        if memory_type:
            filter_conditions.append({"memory_type": {"$eq": memory_type}})
        if min_importance > 0:
            filter_conditions.append({"importance_score": {"$gte": min_importance}})
        if not include_outdated:
            filter_conditions.append({"success_rate": {"$gte": 0.2}})

        filters = None
        if len(filter_conditions) == 1:
            filters = filter_conditions[0]
        elif len(filter_conditions) > 1:
            filters = {"$and": filter_conditions}

        return await self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=collection_name,
        )

    async def get_working_memory(
        self,
        user_id: str,
        session_id: str,
        max_turns: int = 10,
    ) -> list[Chunk]:
        """Get working memory (short-term) for a specific session.

        Args:
            user_id: User ID
            session_id: Session ID
            max_turns: Maximum number of turns to retrieve

        Returns:
            List of working memory chunks sorted by creation time
        """
        collection_name = self.get_collection_name(user_id)

        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.get(
                where={
                    "$and": [
                        {"session_id": {"$eq": session_id}},
                        {"memory_type": {"$eq": "working"}},
                    ]
                },
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                document = results["documents"][i] if results["documents"] else ""
                metadata = self._deserialize_metadata(results["metadatas"][i]) if results["metadatas"] else {}
                embedding = results["embeddings"][i] if results.get("embeddings") else None

                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=document,
                    chunk_index=metadata.get("chunk_index", 0),
                    metadata=metadata,
                    embedding=embedding,
                )
                chunks.append(chunk)

        chunks.sort(key=lambda x: x.metadata.get("created_at", ""))
        return chunks[-max_turns:]

    async def delete(self, chunk_ids: list[str], collection_name: str | None = None) -> None:
        """Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs to delete
            collection_name: Optional collection name
        """
        if not chunk_ids:
            return

        collection = self.get_or_create_collection(collection_name)
        collection.delete(ids=chunk_ids)
        logger.debug(f"Deleted {len(chunk_ids)} memory chunks")

    async def delete_by_document_id(self, document_id: str, collection_name: str | None = None) -> int:
        """Delete all chunks associated with a document.

        Args:
            document_id: Document ID
            collection_name: Optional collection name

        Returns:
            Number of chunks deleted
        """
        collection = self.get_or_create_collection(collection_name)

        count_result = collection.get(where={"document_id": document_id}, include=[])
        count = len(count_result["ids"]) if count_result and count_result["ids"] else 0

        if count > 0:
            collection.delete(where={"document_id": document_id})
            logger.debug(f"Deleted {count} memory chunks for document_id: {document_id}")

        return count

    async def delete_by_metadata(self, metadata_filter: dict[str, Any], collection_name: str | None = None) -> int:
        """Delete all chunks matching metadata filter.

        Args:
            metadata_filter: Metadata filter dict
            collection_name: Optional collection name

        Returns:
            Number of chunks deleted
        """
        collection = self.get_or_create_collection(collection_name)

        if len(metadata_filter) > 1:
            where_clause = {"$and": [{key: value} for key, value in metadata_filter.items()]}
        else:
            where_clause = metadata_filter

        try:
            count_result = collection.get(where=where_clause, include=[])
            count = len(count_result["ids"]) if count_result and count_result["ids"] else 0

            if count > 0:
                collection.delete(where=where_clause)
                logger.debug(f"Deleted {count} memory chunks matching filter: {metadata_filter}")
            return count
        except Exception as e:
            logger.error(f"Failed to delete by metadata {metadata_filter}: {e}")
            return 0

    async def cleanup_outdated_memories(
        self,
        user_id: str,
        success_rate_threshold: float = 0.2,
    ) -> int:
        """Clean up outdated procedural memories with low success rate.

        Args:
            user_id: User ID to clean up
            success_rate_threshold: Memories below this threshold are removed

        Returns:
            Number of memories cleaned up
        """
        collection_name = self.get_collection_name(user_id, "procedural")
        return await self.delete_by_metadata(
            {"success_rate": {"$lt": success_rate_threshold}},
            collection_name=collection_name,
        )

    async def get_by_id(self, chunk_id: str, collection_name: str | None = None) -> Chunk | None:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk ID
            collection_name: Optional collection name

        Returns:
            Chunk object or None if not found
        """
        collection = self.get_or_create_collection(collection_name)
        results = collection.get(ids=[chunk_id], include=["embeddings", "documents", "metadatas"])

        if results["ids"]:
            metadata = self._deserialize_metadata(results["metadatas"][0]) if results["metadatas"] else {}
            return Chunk(
                id=chunk_id,
                document_id=metadata.get("document_id", ""),
                content=results["documents"][0] if results["documents"] else "",
                chunk_index=metadata.get("chunk_index", 0),
                metadata=metadata,
                embedding=results["embeddings"][0] if results.get("embeddings") else None,
            )
        return None

    async def count(self, collection_name: str | None = None) -> int:
        """Get total number of chunks.

        Args:
            collection_name: Optional collection name

        Returns:
            Total number of chunks
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    async def clear(self, collection_name: str | None = None) -> None:
        """Clear all data from a collection.

        Args:
            collection_name: Optional collection name
        """
        name = collection_name or self._default_collection_name
        self.client.delete_collection(name=name)
        if name in self._collections:
            del self._collections[name]
        # Recreate empty collection
        self._collections[name] = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared memory collection: {name}")

    def list_collections(self) -> list[str]:
        """List all collection names.

        Returns:
            List of collection names
        """
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete a collection permanently.

        Args:
            collection_name: Collection name to delete

        Returns:
            True if successful
        """
        name = collection_name or self._default_collection_name
        try:
            self.client.delete_collection(name=name)
            if name in self._collections:
                del self._collections[name]
            logger.info(f"Deleted memory collection: {name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete collection {name}: {e}")
            return False