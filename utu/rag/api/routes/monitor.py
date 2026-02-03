"""Storage monitoring routes"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..services.monitor_service import MonitorService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global singleton
monitor_service = MonitorService()


@router.get("/health/storage")
async def get_storage_health():
    """
    Get storage backend health status (JSON format)

    Returns:
        Health status of all storage backends (Vector Store, Database, Object Storage).
    """
    try:
        return monitor_service.get_storage_health()
    except Exception as e:
        logger.error(f"Get storage health error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/storage/metrics")
async def get_storage_metrics():
    """
    Get detailed storage backend metrics (JSON format)

    Returns:
        Detailed metrics including capacity, usage rate, etc.
    """
    try:
        return monitor_service.get_storage_metrics()
    except Exception as e:
        logger.error(f"Get storage metrics error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor", response_class=HTMLResponse)
async def get_storage_monitor_dashboard():
    """
    Get storage monitoring dashboard (HTML format)

    Returns:
        A complete HTML page displaying storage health status
    """
    try:
        return monitor_service.get_storage_dashboard_html()
    except Exception as e:
        logger.error(f"Get storage dashboard error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
