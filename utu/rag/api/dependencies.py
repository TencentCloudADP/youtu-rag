"""FastAPI dependencies."""
import logging
import yaml
import sys
import os
from typing import Optional, Union
from pathlib import Path
from fastapi import HTTPException

from ...agents import (
    SimpleAgent,
    OrchestraAgent,
    OrchestratorAgent,
    ParallelOrchestratorAgent,
    WorkforceAgent,
)
from ..rag_agents import OrchestraReactSqlAgent
from ...config import ConfigLoader
from .config import settings

# 导入 ExcelAgent
project_root = Path(__file__).parent.parent.parent.parent
adg_path = project_root / "integrations" / "ADG" / "src" / "model" / "adg_benchmark"
if str(adg_path) not in sys.path:
    sys.path.insert(0, str(adg_path))
from excel_agent import ExcelAgent

logger = logging.getLogger(__name__)

AgentType = Union[SimpleAgent, OrchestraAgent, OrchestraReactSqlAgent, OrchestratorAgent, ParallelOrchestratorAgent, WorkforceAgent, ExcelAgent]

_agent: Optional[AgentType] = None
_current_agent_config: str = settings.DEFAULT_AGENT_CONFIG
_current_agent_object: str = "SimpleAgent"  # 默认 agent 类型


def load_agent_configs():
    """Load agent configurations"""
    agents_config_file = settings.PROJECT_ROOT / "configs" / "rag" / "frontend_agents.yaml"
    try:
        with open(agents_config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('agents', [])
    except Exception as e:
        logger.warning(f"Failed to load agents.yaml: {e}")
        return [
            {
                "name": "Chat",
                "description": "基础对话Agent",
                "config_path": "simple/base.yaml",
                "icon": "💬"
            }
        ]


AGENT_CONFIGS = load_agent_configs()


def _create_agent_instance(agent_object: str, config) -> AgentType:
    """
    Create an agent instance based on its type specified by agent_object.
    
    Args:
        agent_object: Agent type (e.g. "SimpleAgent", "OrchestraAgent" etc.)
        config: Agent configuration object or config file path. Notice that ExcelAgent requires a config file path; otherwise it loads the config file from _current_agent_config.
        
    Returns:
        Agent instance of the corresponding type.
    """
    agent_classes = {
        "SimpleAgent": SimpleAgent,
        "OrchestraAgent": OrchestraAgent,
        "OrchestraReactSqlAgent": OrchestraReactSqlAgent,
        "OrchestratorAgent": OrchestratorAgent,
        "ParallelOrchestratorAgent": ParallelOrchestratorAgent,
        "WorkforceAgent": WorkforceAgent,
        "ExcelAgent": ExcelAgent,
    }
    
    agent_class = agent_classes.get(agent_object)
    if not agent_class:
        logger.error(f"Unknown agent object type: {agent_object}")
        raise ValueError(f"Unknown agent object type: {agent_object}. Available types: {', '.join(agent_classes.keys())}")
    
    # Handle ExcelAgent special case
    if agent_object == "ExcelAgent":
        if isinstance(config, str):
            config_path = config
        else:
            config_path = str(settings.PROJECT_ROOT / "configs" / "agents" / _current_agent_config)
        
        if not Path(config_path).exists():
            raise FileNotFoundError(f"ExcelAgent config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict) or 'config' not in config_data:
                    raise ValueError(f"Invalid ExcelAgent config format in {config_path}: missing 'config' key")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in ExcelAgent config {config_path}: {e}")
        
        logger.info(f"Creating ExcelAgent with config path: {config_path}")
        return ExcelAgent(config=config_path)
    
    return agent_class(config=config)


async def initialize_agent():
    """Initialize the agent, which is called when the application starts."""
    global _agent, _current_agent_object
    try:
        logger.info("Starting agent initialization...")
        
        agent_info = next(
            (agent for agent in AGENT_CONFIGS if agent['config_path'] == _current_agent_config),
            None
        )
        
        if agent_info:
            _current_agent_object = agent_info.get('agent_object', 'SimpleAgent')
        
        if _current_agent_object == "ExcelAgent":
            config_path = str(settings.PROJECT_ROOT / "configs" / "agents" / _current_agent_config)
            _agent = _create_agent_instance(_current_agent_object, config_path)
        else:
            config = ConfigLoader.load_agent_config(_current_agent_config)
            _agent = _create_agent_instance(_current_agent_object, config)
            await _agent.build()
        
        logger.info(f"✓ {_current_agent_object} initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize agent: {e}", exc_info=True)


async def cleanup_agent():
    """Clean up the agent, which is called when the application closes."""
    global _agent
    if _agent:
        logger.info("Shutting down agent...")
        _agent = None
        logger.info("✓ Agent shutdown completed")


async def get_agent() -> AgentType:
    """Get the agent instance."""
    global _agent, _current_agent_object
    if _agent is None:
        try:
            logger.warning("Agent not initialized, initializing now...")
            
            agent_info = next(
                (agent for agent in AGENT_CONFIGS if agent['config_path'] == _current_agent_config),
                None
            )
            
            if agent_info:
                _current_agent_object = agent_info.get('agent_object', 'SimpleAgent')
            
            if _current_agent_object == "ExcelAgent":
                config_path = str(settings.PROJECT_ROOT / "configs" / "agents" / _current_agent_config)
                _agent = _create_agent_instance(_current_agent_object, config_path)
            else:
                config = ConfigLoader.load_agent_config(_current_agent_config)
                _agent = _create_agent_instance(_current_agent_object, config)
                if isinstance(_agent, SimpleAgent):
                    await _agent.build()
            
            logger.info(f"{_current_agent_object} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")
    return _agent


def reset_agent():
    """Reset the agent."""
    global _agent
    _agent = None


def get_current_agent_config() -> str:
    """Get the configuration of the current agent."""
    return _current_agent_config


def set_current_agent_config(config: str):
    """Set the configuration of the current agent."""
    global _current_agent_config
    _current_agent_config = config


def get_current_agent_object() -> str:
    """Get the object type of the current agent."""
    return _current_agent_object


def set_current_agent_object(agent_object: str):
    """Set the object type of the current agent."""
    global _current_agent_object
    _current_agent_object = agent_object
