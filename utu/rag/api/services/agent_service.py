"""Agent management service"""
import logging
import os
from datetime import datetime
from fastapi import HTTPException

from ..dependencies import (
    AGENT_CONFIGS,
    get_current_agent_config,
    set_current_agent_config,
    get_current_agent_object,
    set_current_agent_object,
    reset_agent
)

logger = logging.getLogger(__name__)


class AgentService:
    """The service interface for Agent management"""
    
    @staticmethod
    def get_agent_list():
        """Get the list of available agents."""
        return {
            "agents": AGENT_CONFIGS,
            "current": get_current_agent_config(),
            "current_agent_object": get_current_agent_object(),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def switch_agent(config_path: str):
        """Switch agent by specifying the new config path.
        
        Args:
            config_path: Agent config path.
            
        Returns:
            Results of the switch operation.
        """
        valid_paths = [agent['config_path'] for agent in AGENT_CONFIGS]
        if config_path not in valid_paths:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent config: {config_path}. Available configs: {', '.join(valid_paths)}"
            )
        
        # Find the corresponding agent info and agent object
        agent_info = next((agent for agent in AGENT_CONFIGS if agent['config_path'] == config_path), None)
        
        agent_object = agent_info.get('agent_object', 'SimpleAgent') if agent_info else 'SimpleAgent'
        
        os.environ["AGENT_CONFIG_FILE"] = config_path
        
        # Reset the agent to use the new configuration
        reset_agent()
        set_current_agent_config(config_path)
        set_current_agent_object(agent_object)
        
        logger.info(f"Switched to agent config: {config_path} with agent_object: {agent_object}")
        
        return {
            "message": f"Successfully switched to agent: {agent_info['name'] if agent_info else config_path}",
            "config_path": config_path,
            "agent_object": agent_object,
            "agent_info": agent_info,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_current_agent():
        """Get the current agent configuration."""
        current_config = get_current_agent_config()
        current_agent_object = get_current_agent_object()
        agent_info = next(
            (agent for agent in AGENT_CONFIGS if agent['config_path'] == current_config), 
            None
        )
        
        return {
            "config_path": current_config,
            "agent_object": current_agent_object,
            "agent_info": agent_info,
            "available_agents": AGENT_CONFIGS,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    async def get_agent_status(agent):
        """Get the status of the agent."""
        agent_type = type(agent).__name__ if agent else "Unknown"
        return {
            "status": "ready",
            "agent_type": agent_type,
            "config": {
                "has_tools": hasattr(agent, 'tools') and len(agent.tools) > 0 if agent else False,
                "streaming_enabled": True
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def reset_agent_state():
        """Reset the agent state."""
        reset_agent()
        return {
            "message": "Agent reset successfully",
            "timestamp": datetime.now().isoformat()
        }
