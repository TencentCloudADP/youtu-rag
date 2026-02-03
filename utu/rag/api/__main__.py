"""
Entry point for running the API module directly with python -m

Usage:
    python -m utu.rag.api
"""
import uvicorn
from .main import app
from .config import settings


def main():
    """Run the FastAPI application using uvicorn"""
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
