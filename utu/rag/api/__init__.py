"""
RAG API module for Youtu Agentic RAG.

This module provides the FastAPI application for the RAG system.

Usage:
    # Method 1: Direct uvicorn
    uvicorn utu.rag.api.main:app --reload --port 8001
    
    # Method 2: Module execution
    python -m utu.rag.api
"""


def get_app():
    """Deferred import to avoid circular dependencies and initialization order issues"""
    from .main import app
    return app


def __getattr__(name):
    if name == "app":  # For legacy compatibility
        from .main import app
        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["get_app"]
