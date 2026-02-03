"""Agent management routes"""
import logging
from fastapi import APIRouter, HTTPException

from ..models.agent import AgentSwitchRequest
from ..dependencies import get_agent
from ..services.agent_service import AgentService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/list")
async def get_agent_list():
    """Get all available agent list for a dropdown list"""
    try:
        return AgentService.get_agent_list()
    except Exception as e:
        logger.error(f"Get agent list error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch")
async def switch_agent(request: AgentSwitchRequest):
    """Switch agent.

    Steps:
    1. Update current agent config path
    2. Set AGENT_CONFIG_FILE environment variable
    3. Reset agent instance to use new config
    4. Return new agent config information
    """
    try:
        return AgentService.switch_agent(request.config_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent switch error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_agent():
    """Get current agent configuration"""
    try:
        return AgentService.get_current_agent()
    except Exception as e:
        logger.error(f"Get current agent error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_agent_status():
    """Get current agent status"""
    try:
        agent = await get_agent()
        return await AgentService.get_agent_status(agent)
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": AgentService.get_current_agent()["timestamp"]
        }


@router.post("/reset")
async def reset_agent():
    """Reset agent state (only used for testing)"""
    return AgentService.reset_agent_state()
