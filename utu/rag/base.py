"""Base classes and interfaces for RAG module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..db.utu_basemodel import UTUBaseModel


@dataclass
class Document:
    """Represents a document in the knowledge base."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = None

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id}, content='{content_preview}', metadata={self.metadata})"


@dataclass
class Chunk:
    """Represents a chunk of a document."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = None

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index}, content='{content_preview}')"


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""

    chunk: Chunk
    score: float
    rank: int | None = None

    def __repr__(self) -> str:
        return f"RetrievalResult(chunk_id={self.chunk.id}, score={self.score:.4f}, rank={self.rank})"


class QueryRequest(BaseModel):
    """Query request for retrieval."""

    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    enable_reranking: bool = False
    similarity_threshold: float | None = None


class QueryResponse(BaseModel):
    """Query response from retrieval."""

    query: str
    results: list[dict[str, Any]]  # Serialized RetrievalResult
    total_results: int
    retrieval_time_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class BuildStatus(UTUBaseModel):
    """Status of knowledge building process."""

    status: str  # "pending", "running", "completed", "failed"
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    errors: list[str] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(UTUBaseModel):
    """Health status of storage system."""

    is_healthy: bool
    backend: str
    collection_name: str
    total_documents: int = 0
    total_chunks: int = 0
    index_size_bytes: int = 0
    last_check_time: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Abstract Base Classes

class BaseTextSplitter(ABC):
    """Base class for text splitting strategies."""

    @abstractmethod
    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text into chunks."""
        pass


class BaseEmbedder(ABC):
    """Base class for embedding generation."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        pass


class BaseReranker(ABC):
    """Base class for reranking."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results based on relevance to query.

        Args:
            query: Query text
            results: List of retrieval results to rerank
            top_k: Optional number of top results to return (None returns all)

        Returns:
            Reranked list of retrieval results with updated scores and ranks
        """
        pass


class BaseKnowledgeBuilder(ABC):
    """Base class for knowledge building."""

    @abstractmethod
    async def build_from_documents(
        self, documents: list[Document], rebuild: bool = False
    ) -> BuildStatus:
        """Build knowledge base from documents."""
        pass

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> BuildStatus:
        """Add documents to existing knowledge base."""
        pass

    @abstractmethod
    async def get_build_status(self) -> BuildStatus:
        """Get current build status."""
        pass


class BaseRetriever(ABC):
    """Base class for knowledge retrieval."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        pass

    @abstractmethod
    async def batch_retrieve(
        self, queries: list[str], top_k: int = 5, **kwargs
    ) -> list[list[RetrievalResult]]:
        """Batch retrieve for multiple queries."""
        pass


class BaseVectorStore(ABC):
    """Base class for vector storage."""

    @abstractmethod
    async def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store."""
        pass

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by IDs."""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks associated with a document.

        Args:
            document_id: Document ID (usually the filename/source identifier)

        Returns:
            Number of chunks deleted
        """
        pass

    @abstractmethod
    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Get chunk by ID."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of chunks."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all data from the store."""
        pass


class BaseStorageMonitor(ABC):
    """Base class for storage monitoring."""

    @abstractmethod
    async def check_health(self) -> HealthStatus:
        """Check storage health."""
        pass

    @abstractmethod
    async def collect_metrics(self) -> dict[str, Any]:
        """Collect storage metrics."""
        pass

    @abstractmethod
    async def log_query(self, query: str, latency_ms: float, result_count: int) -> None:
        """Log a query execution."""
        pass

    @abstractmethod
    async def get_query_stats(self, time_range_hours: int = 24) -> dict[str, Any]:
        """Get query statistics."""
        pass
