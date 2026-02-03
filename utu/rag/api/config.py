"""Application configurations."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Settings of the RAG API application."""
    APP_TITLE: str = "Youtu Agentic RAG - Agent Workflow API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Backend API for agent workflow visualization and chat interface"
    PROJECT_ROOT: Path = project_root
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    FRONTEND_DIR: Path = project_root / "frontend" / "rag_webui"
    CORS_ORIGINS: list = ["*"]
    LOG_LEVEL: str = os.getenv("UTU_LOG_LEVEL", "INFO")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "rag-documents")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    DEFAULT_AGENT_CONFIG: str = "simple/base_search.yaml"

    @property
    def database_url(self) -> str:
        """Get the database URL, supporting environment variable configuration."""
        return os.getenv("UTU_DB_URL", "sqlite:///./rag_data/relational_database/rag_demo.sqlite")

    @property
    def chroma_persist_directory(self) -> str:
        """Get the ChromaDB persistence directory."""
        return os.getenv("VECTOR_STORE_PATH", "rag_data/vector_store")

    @property
    def relational_db_path(self) -> str:
        """Get the relational database path."""
        return os.getenv("RELATIONAL_DB_PATH", "rag_data/relational_database/rag_demo.sqlite")

    @property
    def memory_store_path(self) -> str:
        """
        Get memory store path from environment variable.
        
        Priority:
        1. MEMORY_STORE_PATH if set
        2. Parent directory of VECTOR_STORE_PATH + '/memory_data'
        3. Default: './rag_data/memory_data'
        
        Returns:
            str: Memory store path
        """
        memory_path = os.getenv("MEMORY_STORE_PATH")
        if memory_path:
            return memory_path
        
        # Get VECTOR_STORE_PATH and use its parent directory
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "rag_data/vector_store")
        parent_dir = str(Path(vector_store_path).parent)
        return f"{parent_dir}/memory_data"

    class Config:
        env_file = str(env_path)
        case_sensitive = True
        extra = "ignore"


settings = Settings()
