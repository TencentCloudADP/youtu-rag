"""Memory config routes"""
import logging
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class MemoryConfig(BaseModel):
    """Memory config, specifies whether memory is enabled"""
    enabled: bool


@router.get("/config")
async def get_memory_config():
    """Get the global memory config (whether memory is enabled).

    It returns the current value of the memoryEnabled environment variable, which is loaded from the .env file at startup.
    Runtime changes will reflect in this interface, but will revert to the default value in the .env file after restart.
    
    Returns:
        dict: {"enabled": bool}, specifying whether memory is enabled.
    """
    is_enabled = os.environ.get("memoryEnabled", "true").lower() == "true"  # Default to True
    return {"enabled": is_enabled}


@router.post("/config")
async def update_memory_config(config: MemoryConfig):
    """Update the global memory config (whether memory is enabled).
    
    It modifies the memoryEnabled environment variable in the current process only.
    
    Note: This operation only affects the current running process. After restart, it will revert to the default value in the .env file.
    To make permanent changes, edit the memoryEnabled configuration in the .env file directly.
    
    Args:
        config: Memory configuration object.
        
    Returns:
        dict: {"status": str, "enabled": bool}, the update status.
        
    Raises:
        HTTPException: raised when config update fails.
    """
    try:
        new_status = str(config.enabled).lower()
        
        os.environ["memoryEnabled"] = new_status
                
        logger.info(f"Memory config updated to: {new_status} (runtime only)")
        return {"status": "updated", "enabled": config.enabled}
        
    except Exception as e:
        logger.error(f"Failed to update memory config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")
