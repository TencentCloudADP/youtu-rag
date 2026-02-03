"""Router modules"""
from .chat import router as chat_router
from .agent import router as agent_router
from .knowledge_base import router as knowledge_base_router
from .file import router as file_router
from .minio_files import router as minio_files_router
from .embedding import router as embedding_router
from .reranker import router as reranker_router
from .kb_config import router as kb_config_router
from .config import router as config_router
from .monitor import router as monitor_router
from .memory import router as memory_router

__all__ = [
    "chat_router",
    "agent_router",
    "knowledge_base_router",
    "file_router",
    "minio_files_router",
    "embedding_router",
    "reranker_router",
    "kb_config_router",
    "config_router",
    "monitor_router",
    "memory_router",
]
