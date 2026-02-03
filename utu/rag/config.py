"""RAG Configuration Classes."""

from typing import Any, Literal

from pydantic import Field

from ..config.base_config import ConfigBaseModel


class ChunkingConfig(ConfigBaseModel):
    """Configuration for document chunking."""

    strategy: Literal["recursive", "hierarchical"] = "recursive"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    separators: list[str] | None = None
    keep_separator: bool = True


class EmbeddingConfig(ConfigBaseModel):
    """Configuration for embedding generation."""

    model: str = "text-embedding-3-small"
    provider: Literal["openai", "local", "huggingface"] = "openai"
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = Field(default=32, ge=1, le=512)
    dimensions: int | None = None  # For models that support dimension reduction


class KnowledgeBuilderConfig(ConfigBaseModel):
    """Configuration for knowledge building."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    max_workers: int = Field(default=4, ge=1, le=16)
    enable_metadata: bool = True
    metadata_fields: list[str] = Field(default_factory=lambda: ["source", "page", "title"])
    batch_delay: float = Field(default=3.0, ge=0.0, le=60.0)  # Delay between batches in seconds


class RetrieverConfig(ConfigBaseModel):
    """Configuration for knowledge retrieval."""

    top_k: int = Field(default=5, ge=1)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_reranking: bool = False
    reranker_model: str | None = None
    reranker_top_k: int = Field(default=3, ge=1, le=50)


class VectorStoreConfig(ConfigBaseModel):
    """Configuration for vector storage."""

    backend: Literal["chroma"] = "chroma"
    collection_name: str = "knowledge_base"
    persist_directory: str = "./data/vector_store"

    host: str | None = None
    port: int | None = None
    api_key: str | None = None
    distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine"

    index_type: str | None = None  # e.g., "IVF_FLAT", "HNSW"
    index_params: dict[str, Any] = Field(default_factory=dict)


class MonitorConfig(ConfigBaseModel):
    """Configuration for storage monitoring."""

    enable_monitoring: bool = True
    health_check_interval: int = Field(default=60, ge=10, le=3600)  # seconds
    metrics_retention_days: int = Field(default=30, ge=1, le=365)
    enable_query_logging: bool = True
    enable_alerts: bool = True
    alert_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "query_latency_ms": 1000.0,
            "error_rate": 0.05,
            "index_size_gb": 100.0,
        }
    )


class RAGConfig(ConfigBaseModel):
    """Main RAG configuration."""

    name: str = "default_rag"
    description: str | None = None

    knowledge_builder: KnowledgeBuilderConfig = Field(default_factory=KnowledgeBuilderConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    enable_cache: bool = True
    cache_ttl: int = Field(default=3600, ge=60, le=86400)  # seconds
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
