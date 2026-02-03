"""ChromaDB vector store implementation."""

import logging
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None
    Settings = None

from ...base import BaseVectorStore, Chunk
from ...config import VectorStoreConfig

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize ChromaDB vector store.

        Args:
            config: Vector store configuration

        Raises:
            ImportError: If chromadb is not installed
        """
        if not CHROMA_AVAILABLE:
            msg = "chromadb is not installed. Install it with: pip install chromadb"
            raise ImportError(msg)

        self.config = config

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Map distance metric to ChromaDB's HNSW space parameter
        # ChromaDB uses "hnsw:space" in metadata to set distance function
        distance_metric_map = {
            "cosine": "cosine",    # Cosine similarity
            "euclidean": "l2",     # L2 distance (Euclidean)
            "dot": "ip",           # Inner product
        }
        hnsw_space = distance_metric_map.get(config.distance_metric, "cosine")

        # Get or create collection with proper distance metric
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": hnsw_space},
        )

        logger.info(f"Initialized ChromaDB vector store at: {config.persist_directory}")
        logger.info(f"Collection name: {config.collection_name}")

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunks to add
        """
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                # Filter out None values - ChromaDB doesn't accept None
                **{k: v for k, v in (chunk.metadata or {}).items() if v is not None},
            }
            for chunk in chunks
        ]

        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    async def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (can be simple dict or ChromaDB where clause)

        Returns:
            List of (chunk, score) tuples
        """
        # Build where clause for filters
        where = None
        if filters:
            # If filters already contains operators ($and, $or, etc.) or nested operators,
            # pass it directly to ChromaDB. Otherwise, convert to $eq format for backward compatibility.
            if any(key.startswith("$") for key in filters.keys()):
                # Already in ChromaDB format (contains $and, $or, etc.)
                where = filters
            elif any(isinstance(value, dict) and any(k.startswith("$") for k in value.keys()) for value in filters.values()):
                # Contains nested operators like {"field": {"$gte": value}}
                where = filters
            else:
                # Simple key-value pairs, convert to $eq
                where = {key: {"$eq": value} for key, value in filters.items()}

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where, include=["embeddings", "documents", "metadatas", "distances"]
        )

        chunks_with_scores = []

        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                embedding = results["embeddings"][0][i] if results["embeddings"] else None
                distance = results["distances"][0][i]

                # Convert distance to similarity score (assuming cosine distance)
                # ChromaDB returns distances, lower is better
                # For cosine: distance = 1 - similarity, so similarity = 1 - distance
                similarity = 1.0 - distance

                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=document,
                    chunk_index=metadata.get("chunk_index", 0),
                    metadata=metadata,
                    embedding=embedding,
                )

                chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs to delete
        """
        if not chunk_ids:
            return

        self.collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks from ChromaDB")

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks associated with a document.

        Args:
            document_id: Document ID (usually the filename/source identifier)

        Returns:
            Number of chunks deleted
        """
        count_result = self.collection.get(
            where={"document_id": document_id},
            include=[]  # We only need the count, not the data
        )
        count = len(count_result["ids"]) if count_result and count_result["ids"] else 0

        if count == 0:
            logger.info(f"No chunks found for document_id: {document_id}")
            return 0

        self.collection.delete(where={"document_id": document_id})
        logger.info(f"Deleted {count} chunks for document_id: {document_id}")
        return count

    async def delete_by_metadata(self, metadata_filter: dict[str, Any]) -> int:
        """Delete all chunks matching metadata filter.

        Args:
            metadata_filter: Metadata filter dict (e.g., {"source": "file.xlsx"})

        Returns:
            Number of chunks deleted
        """
        # Transform metadata_filter for ChromaDB compatibility
        # ChromaDB requires explicit $and operator for multiple conditions
        if len(metadata_filter) > 1:
            where_clause = {
                "$and": [
                    {key: value} for key, value in metadata_filter.items()
                ]
            }
        else:
            where_clause = metadata_filter

        try:
            count_result = self.collection.get(
                where=where_clause,
                include=[]  # We only need the count, not the data
            )
            count = len(count_result["ids"]) if count_result and count_result["ids"] else 0

            if count == 0:
                logger.debug(f"No chunks found matching filter: {metadata_filter}")
                return 0

            self.collection.delete(where=where_clause)
            logger.info(f"Deleted {count} chunks matching filter: {metadata_filter}")
            return count

        except Exception as e:
            logger.error(f"Failed to delete by metadata {metadata_filter}: {e}")
            return 0

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object or None if not found
        """
        results = self.collection.get(ids=[chunk_id], include=["embeddings", "documents", "metadatas"])

        if results["ids"]:
            metadata = results["metadatas"][0]
            chunk = Chunk(
                id=chunk_id,
                document_id=metadata.get("document_id", ""),
                content=results["documents"][0],
                chunk_index=metadata.get("chunk_index", 0),
                metadata=metadata,
                embedding=results["embeddings"][0] if results["embeddings"] else None,
            )
            return chunk

        return None

    async def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Total number of chunks in the store
        """
        return self.collection.count()

    async def clear(self) -> None:
        """Clear all data from the store."""
        # Map distance metric to ChromaDB's HNSW space parameter
        distance_metric_map = {
            "cosine": "cosine",
            "euclidean": "l2",
            "dot": "ip",
        }
        hnsw_space = distance_metric_map.get(self.config.distance_metric, "cosine")

        self.client.delete_collection(name=self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": hnsw_space},
        )
        logger.info(f"Cleared ChromaDB collection: {self.config.collection_name}")

    @staticmethod
    def cleanup_orphaned_directories(persist_directory: str) -> dict:
        """Cleanup orphaned UUID directories - deleted in ChromaDB index but still physically exist.

        Note: ChromaDB uses a two-layer UUID structure:
        1. Collection UUID - the collection's unique identifier;
        2. Segment UUID - the vector storage segment's unique identifier, which is the actual directory name.

        Args:
            persist_directory: ChromaDB persist directory path

        Returns:
            Cleanup statistics {"deleted_count": int, "deleted_dirs": list}
        """
        import shutil
        from pathlib import Path
        import sqlite3

        try:
            persist_path = Path(persist_directory)
            if not persist_path.exists():
                logger.warning(f"Persist directory does not exist: {persist_directory}")
                return {"deleted_count": 0, "deleted_dirs": []}

            sqlite_path = persist_path / "chroma.sqlite3"
            if not sqlite_path.exists():
                logger.warning(f"chroma.sqlite3 not found in {persist_directory}")
                return {"deleted_count": 0, "deleted_dirs": []}

            conn = sqlite3.connect(str(sqlite_path))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM segments WHERE type = 'urn:chroma:segment/vector/hnsw-local-persisted'")
            active_segment_uuids = {row[0] for row in cursor.fetchall()}
            conn.close()

            logger.info(f"Found {len(active_segment_uuids)} active vector segments in ChromaDB")

            deleted_dirs = []
            for item in persist_path.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    if len(item.name) == 36 and item.name.count('-') == 4:  # It is a UUID
                        if item.name not in active_segment_uuids:
                            logger.info(f"🗑️  Deleting orphaned segment directory: {item.name}")
                            shutil.rmtree(item)
                            deleted_dirs.append(item.name)

            logger.info(f"✅ Cleanup completed: deleted {len(deleted_dirs)} orphaned directories")
            return {
                "deleted_count": len(deleted_dirs),
                "deleted_dirs": deleted_dirs
            }

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned directories: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection permanently (including physical segment directories)."""
        import shutil
        from pathlib import Path
        import sqlite3

        try:
            # Fetch collection UUID before deletion
            collection_uuid = None

            # Get collection if not already initialized
            if not self.collection:
                try:
                    self.collection = self.client.get_collection(name=self.config.collection_name)
                    logger.info(f"Loaded collection before deletion: {self.config.collection_name}")
                except Exception as e:
                    logger.warning(f"Collection not found or already deleted: {e}")

            if self.collection:
                try:
                    collection_uuid = str(self.collection.id)
                    logger.info(f"Collection UUID: {collection_uuid}")
                except Exception as e:
                    logger.warning(f"Failed to get collection UUID: {e}")

            # Get all Segment UUIDs for the collection before deletion
            segment_uuids = []
            if collection_uuid and self.config.persist_directory:
                try:
                    persist_path = Path(self.config.persist_directory)
                    sqlite_path = persist_path / "chroma.sqlite3"

                    if sqlite_path.exists():
                        conn = sqlite3.connect(str(sqlite_path))
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT id FROM segments WHERE collection = ? AND type = 'urn:chroma:segment/vector/hnsw-local-persisted'",
                            (collection_uuid,)
                        )
                        segment_uuids = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        logger.info(f"Found {len(segment_uuids)} segments for collection {self.config.collection_name}")
                except Exception as e:
                    logger.warning(f"Failed to query segments: {e}")

            self.client.delete_collection(name=self.config.collection_name)
            logger.info(f"Deleted ChromaDB collection from index: {self.config.collection_name}")
            self.collection = None

            if segment_uuids and self.config.persist_directory:
                persist_path = Path(self.config.persist_directory)
                deleted_count = 0

                for segment_uuid in segment_uuids:
                    segment_dir = persist_path / segment_uuid
                    if segment_dir.exists() and segment_dir.is_dir():
                        shutil.rmtree(segment_dir)
                        deleted_count += 1
                        logger.info(f"✅ Deleted segment directory: {segment_uuid}")

                if deleted_count > 0:
                    logger.info(f"✅ Deleted {deleted_count} segment directories for collection {self.config.collection_name}")
            else:
                logger.debug(f"No segments found for collection {self.config.collection_name} (normal for empty collections)")

        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection: {e}")
            raise
