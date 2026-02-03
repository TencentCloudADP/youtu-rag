"""Agent-related models"""
from pydantic import BaseModel


class AgentSwitchRequest(BaseModel):
    """Agent switch request, storing the config path"""
    config_path: str 
