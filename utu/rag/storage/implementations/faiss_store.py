"""FAISS vector store implementation."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from ...base import BaseVectorStore, Chunk
from ...config import VectorStoreConfig

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation of vector store.

    FAISS only stores vectors, so we maintain a separate mapping
    for chunk IDs, content, and metadata.
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize FAISS vector store.

        Args:
            config: Vector store configuration

        Raises:
            ImportError: Raised if faiss is not installed
        """
        if not FAISS_AVAILABLE:
            msg = "faiss is not installed. Install it with: pip install faiss-cpu"
            raise ImportError(msg)

        self.config = config
        self.persist_dir = Path(config.persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.persist_dir / f"{config.collection_name}.index"
        self.mapping_path = self.persist_dir / f"{config.collection_name}.pkl"

        self.index = None
        self.id_to_idx = {}  # Maps chunk_id to internal index
        self.idx_to_chunk = {}  # Maps internal index to chunk data
        self.next_idx = 0

        self._load_or_create_index()

        logger.info(f"Initialized FAISS vector store at: {config.persist_directory}")
        logger.info(f"Collection name: {config.collection_name}")

    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if self.index_path.exists() and self.mapping_path.exists():
            self.index = faiss.read_index(str(self.index_path))

            with open(self.mapping_path, "rb") as f:
                data = pickle.load(f)
                self.id_to_idx = data["id_to_idx"]
                self.idx_to_chunk = data["idx_to_chunk"]
                self.next_idx = data["next_idx"]

            logger.info(f"Loaded existing FAISS index with {len(self.id_to_idx)} chunks")
        else:
            logger.info("No existing index found, will create new one on first add")

    def _save_index(self):
        """Save index and mappings to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        with open(self.mapping_path, "wb") as f:
            pickle.dump(
                {"id_to_idx": self.id_to_idx, "idx_to_chunk": self.idx_to_chunk, "next_idx": self.next_idx},
                f,
            )

        logger.debug(f"Saved FAISS index to {self.index_path}")

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunks to add
        """
        if not chunks:
            return

        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
        dimension = embeddings.shape[1]

        if self.index is None:
            if self.config.distance_metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)

        if self.config.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            idx = self.next_idx + i
            self.id_to_idx[chunk.id] = idx
            self.idx_to_chunk[idx] = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata or {},
            }

        self.next_idx += len(chunks)

        self._save_index()

        logger.info(f"Added {len(chunks)} chunks to FAISS index")

    async def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters (note: FAISS doesn't support native filtering,
                    so we filter results after retrieval, which may be slower)

        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)

        if self.config.distance_metric == "cosine":
            faiss.normalize_L2(query_vec)

        search_k = top_k * 10 if filters else top_k  # Retrieve more if we need to filter
        search_k = min(search_k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, search_k)

        chunks_with_scores = []

        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            distance = float(distances[0][i])

            chunk_data = self.idx_to_chunk.get(idx)
            if not chunk_data:
                continue

            if filters:
                match = True
                for key, value in filters.items():
                    if chunk_data["metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Convert distance to similarity score
            if self.config.distance_metric == "cosine":
                similarity = distance
            else:
                similarity = 1.0 / (1.0 + distance)

            chunk = Chunk(
                id=chunk_data["id"],
                document_id=chunk_data["document_id"],
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                metadata=chunk_data["metadata"],
                embedding=None,  # Don't load embedding to save memory
            )

            chunks_with_scores.append((chunk, similarity))

            # Stop if we have enough results after filtering
            if len(chunks_with_scores) >= top_k:
                break

        return chunks_with_scores

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Note: FAISS doesn't support efficient deletion, so we need to rebuild the index
        """
        if not chunk_ids:
            return

        indices_to_remove = set()
        for chunk_id in chunk_ids:
            if chunk_id in self.id_to_idx:
                idx = self.id_to_idx[chunk_id]
                indices_to_remove.add(idx)
                del self.id_to_idx[chunk_id]
                del self.idx_to_chunk[idx]

        if not indices_to_remove:
            return

        # Note: FAISS doesn't support efficient deletion
        # We would need to rebuild the entire index, but we don't store embeddings in memory
        # So we can only remove from mappings
        logger.warning("FAISS doesn't support efficient deletion. Chunks removed from mappings but index not rebuilt.")

        self._save_index()
        logger.info(f"Deleted {len(chunk_ids)} chunks from FAISS (mappings only)")

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks associated with a document.

        Args:
            document_id: Document ID (usually the filename/source identifier)

        Returns:
            Number of chunks deleted

        Note: FAISS doesn't support efficient deletion, so we only remove from mappings
        """
        chunk_ids_to_delete = []
        for idx, chunk_data in self.idx_to_chunk.items():
            if chunk_data.get("document_id") == document_id:
                chunk_ids_to_delete.append(chunk_data["id"])

        if not chunk_ids_to_delete:
            logger.info(f"No chunks found for document_id: {document_id}")
            return 0

        await self.delete(chunk_ids_to_delete)

        logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for document_id: {document_id}")
        return len(chunk_ids_to_delete)

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Get chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object or None if not found
        """
        if chunk_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[chunk_id]
        chunk_data = self.idx_to_chunk.get(idx)

        if not chunk_data:
            return None

        return Chunk(
            id=chunk_data["id"],
            document_id=chunk_data["document_id"],
            content=chunk_data["content"],
            chunk_index=chunk_data["chunk_index"],
            metadata=chunk_data["metadata"],
            embedding=None,
        )

    async def count(self) -> int:
        """Get total number of chunks.

        Returns:
            Total number of chunks in the store
        """
        return len(self.id_to_idx)

    async def clear(self) -> None:
        """Clear all data from the store."""
        self.index = None
        self.id_to_idx = {}
        self.idx_to_chunk = {}
        self.next_idx = 0

        if self.index_path.exists():
            self.index_path.unlink()
        if self.mapping_path.exists():
            self.mapping_path.unlink()

        logger.info(f"Cleared FAISS collection: {self.config.collection_name}")
