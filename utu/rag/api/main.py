"""
Youtu Agentic RAG - Main Application Entry Point

A clean and modular FastAPI application for agent workflow visualization.

Run with:
    uv run uvicorn utu.rag.api.main:app --reload --port 8001
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ...utils import setup_logging
from .config import settings

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.info(f"Loading environment from: {settings.PROJECT_ROOT / '.env'}")


# =============================================
# Global Memory Toolkit
# =============================================

# Global memory toolkit instance
_memory_toolkit = None


def get_memory_toolkit():
    """Get the global memory toolkit instance.
    
    Returns:
        VectorMemoryToolkit instance or None if initialization fails.
    """
    global _memory_toolkit
    if _memory_toolkit is None:
        try:
            from utu.tools.memory_toolkit import VectorMemoryToolkit
            _memory_toolkit = VectorMemoryToolkit(
                persist_directory=settings.memory_store_path,
                collection_prefix="rag_chat",
                default_user_id="default_user",
                max_working_memory_turns=20,
            )
            logger.info("✓ Initialized global VectorMemoryToolkit")
        except ImportError as e:
            logger.warning(f"VectorMemoryToolkit not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize VectorMemoryToolkit: {e}")
            return None
    return _memory_toolkit



# =============================================
# Import Dependencies and Routes
# =============================================

from .dependencies import initialize_agent, cleanup_agent, get_agent
from .routes import (
    chat_router,
    agent_router,
    knowledge_base_router,
    file_router,
    minio_files_router,
    embedding_router,
    reranker_router,
    kb_config_router,
    config_router,
    monitor_router,
    memory_router,
)



# =============================================
# Application Lifespan
# =============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_agent()
    
    # Initialize memory toolkit on startup
    memory = get_memory_toolkit()
    if memory:
        logger.info("✓ Memory toolkit initialized on startup")
    else:
        logger.warning("⚠ Memory toolkit not initialized")
    
    yield
    
    await cleanup_agent()


# =============================================
# Application Factory
# =============================================

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    

    app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
    app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
    app.include_router(knowledge_base_router, prefix="/api/knowledge", tags=["Knowledge Base"])
    app.include_router(file_router, prefix="/api/knowledge-base", tags=["Files"])
    app.include_router(minio_files_router, prefix="/api/files", tags=["MinIO Files"])
    app.include_router(embedding_router, prefix="/api/embedding", tags=["Embedding"])
    app.include_router(reranker_router, prefix="/api/reranker", tags=["Reranker"])
    app.include_router(kb_config_router, prefix="/api/knowledge", tags=["KB Config"])
    app.include_router(config_router, prefix="/api/config", tags=["Config"])
    app.include_router(monitor_router, tags=["Monitor"])
    app.include_router(memory_router, prefix="/api/memory", tags=["Memory"])
    

    register_basic_routes(app)

    if settings.FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(settings.FRONTEND_DIR / "assets")), name="assets")
        app.mount("/pages", StaticFiles(directory=str(settings.FRONTEND_DIR / "pages")), name="pages")
        logger.info(f"✓ Frontend static files mounted from: {settings.FRONTEND_DIR}")
    else:
        logger.warning(f"⚠ Frontend directory not found: {settings.FRONTEND_DIR}")

    return app


# =============================================
# Basic Routes Registration
# =============================================

def register_basic_routes(app: FastAPI):
    
    @app.get("/")
    async def root():
        # Directly serve app.html with default route to chat
        app_file = settings.FRONTEND_DIR / "app.html"
        if app_file.exists():
            return FileResponse(app_file)
        return {
            "service": settings.APP_TITLE,
            "version": settings.APP_VERSION,
            "status": "running",
            "endpoints": {
                "chat": "/api/chat",
                "health": "/health",
                "docs": "/docs",
                "api_info": "/api/info"
            }
        }

    @app.get("/api/info")
    async def api_info():
        return {
            "service": settings.APP_TITLE,
            "version": settings.APP_VERSION,
            "status": "running",
            "memory_enabled": get_memory_toolkit() is not None,
            "endpoints": {
                "chat": "/api/chat",
                "health": "/health",
                "ui": "/ui",
                "database": "/database"
            }
        }
    
    @app.get("/health")
    async def health_check():
        try:
            agent = await get_agent()
            agent_status = "healthy" if agent else "unhealthy"
        except Exception as e:
            agent_status = f"error: {str(e)}"
        
        # Check memory toolkit status
        memory = get_memory_toolkit()
        memory_status = "healthy" if memory else "not initialized"

        return {
            "status": "healthy",
            "agent": agent_status,
            "memory": memory_status,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/ui")
    async def serve_ui():
        ui_file = settings.PROJECT_ROOT / "frontend" / "rag_webui" / "agent_workflow.html"
        if not ui_file.exists():
            raise HTTPException(status_code=404, detail="UI file not found")
        return FileResponse(ui_file)
    
    @app.get("/database")
    async def serve_database_ui():
        ui_file = settings.PROJECT_ROOT / "frontend" / "rag_webui" / "index_light.html"
        if not ui_file.exists():
            raise HTTPException(status_code=404, detail="Database UI file not found")
        return FileResponse(ui_file)

    @app.get("/ocr-viewer")
    async def serve_ocr_viewer():
        ui_file = settings.PROJECT_ROOT / "frontend" / "rag_webui" / "pages" / "ocr-viewer.html"
        if not ui_file.exists():
            raise HTTPException(status_code=404, detail="OCR Viewer UI file not found")
        return FileResponse(ui_file)

    @app.get("/chunk-viewer")
    async def serve_chunk_viewer():
        ui_file = settings.PROJECT_ROOT / "frontend" / "rag_webui" / "pages" / "chunk-viewer.html"
        if not ui_file.exists():
            raise HTTPException(status_code=404, detail="Chunk Viewer UI file not found")
        return FileResponse(ui_file)

    @app.get("/docs/assets/rag_logo.png")
    async def serve_rag_logo():
        logo_file = settings.PROJECT_ROOT / "docs" / "assets" / "rag_logo.png"
        if not logo_file.exists():
            raise HTTPException(status_code=404, detail="Logo file not found")
        return FileResponse(logo_file)

    # SPA frontend routing (must be last as catch-all)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Provide frontend SPA application. Returns app.html for all unmatched routes."""
        # If it's an API route but not matched, return 404
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # Check if it's a static file request
        file_path = settings.FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # All other routes return app.html (for SPA hash routing)
        app_file = settings.FRONTEND_DIR / "app.html"
        if not app_file.exists():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return FileResponse(app_file)


# =============================================
# Create Application Instance
# =============================================

app = create_app()

logger.info("✓ Application initialized successfully")