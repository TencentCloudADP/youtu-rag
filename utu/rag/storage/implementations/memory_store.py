"""Vector memory store implementation for agent memory management.

This module provides MemoryVectorStore, a ChromaDB-based vector store
specialized for agent memory management.

Embedding configuration is loaded from `configs/rag/default.yaml`.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

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

# Re-export create_embedder for backward compatibility
from utu.rag.embeddings import create_embedder


logger = logging.getLogger(__name__)


def _load_rag_embedding_config(config_name: str = "default") -> dict[str, Any]:
    """Load embedding configuration from RAG config file.

    Args:
        config_name: Config file name (without .yaml extension).
                    Defaults to "default" which loads configs/rag/default.yaml.

    Returns:
        Embedding configuration dictionary with resolved environment variables.
    """
    try:
        # Get project root (this file is in utu/rag/storage/implementations/)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        config_dir = project_root / "configs" / "rag"

        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            config_path = config_dir / "default.yaml"

        if not config_path.exists():
            logger.warning(f"RAG config not found: {config_path}, using env defaults")
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Resolve environment variables (e.g., ${UTU_EMBEDDING_MODEL} -> actual value)
        def resolve_env_var(value):
            if isinstance(value, str):
                pattern = re.compile(r"\$\{([^}]+)\}")
                matches = pattern.findall(value)
                for var_name in matches:
                    env_value = os.getenv(var_name, "")
                    value = value.replace(f"${{{var_name}}}", env_value)
                return value
            elif isinstance(value, dict):
                return {k: resolve_env_var(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_env_var(item) for item in value]
            return value

        config = resolve_env_var(config)
        embedding_config = config.get("embedding", {})
        logger.info(f"Loaded embedding config from {config_path}: model={embedding_config.get('model')}")

        return embedding_config

    except Exception as e:
        logger.error(f"Error loading RAG embedding config: {e}")
        return {}


# ============== Embedding Service (Lightweight Wrapper) ==============
class EmbeddingService:
    """Embedding service with configuration loaded from configs/rag/default.yaml.

    Example:
        ```python
        service = EmbeddingService()
        embeddings = await service.embed(["text1", "text2"])
        ```
    """

    def __init__(self, config_name: str = "default"):
        """Initialize embedding service from RAG config file.

        Args:
            config_name: Config file name (without .yaml extension).
                        Defaults to "default" which loads configs/rag/default.yaml.
        """
        self._embedder: BaseEmbedder | None = None
        self._config_name = config_name
        self._backend, self._kwargs = self._load_config()

    def _load_config(self) -> tuple[str, dict[str, Any]]:
        """Load configuration from RAG config file."""
        embedding_config = _load_rag_embedding_config(self._config_name)

        if not embedding_config:
            logger.info("No RAG config found, using environment auto-detection")
            return "auto", {}

        # Map YAML config fields to create_embedder parameters
        embedding_type = embedding_config.get("type", "api")
        backend = "service" if embedding_type == "local" else "openai"

        kwargs = {
            "model": embedding_config.get("model"),
            "base_url": embedding_config.get("base_url"),
            "api_key": embedding_config.get("api_key"),
            "batch_size": embedding_config.get("batch_size", 16),
        }

        # For service backend, use service_url instead of base_url
        if backend == "service":
            kwargs["service_url"] = kwargs.pop("base_url")
            kwargs.pop("model", None)
            kwargs.pop("api_key", None)

        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(f"EmbeddingService config: backend={backend}, config={self._config_name}")
        return backend, kwargs

    @property
    def embedder(self) -> BaseEmbedder:
        """Lazy initialization of embedder."""
        if self._embedder is None:
            self._embedder = create_embedder(self._backend, **self._kwargs)
        return self._embedder

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        return await self.embedder.embed_texts(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self.embedder.embed_query(text)


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