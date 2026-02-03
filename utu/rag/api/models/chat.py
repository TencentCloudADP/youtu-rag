"""Chat-related models"""
from typing import Optional
from enum import Enum
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request"""
    query: str
    session_id: Optional[str] = None
    stream: bool = True
    kb_id: Optional[int] = None  # Knowledge base ID (required by KB Search Agent)
    file_ids: Optional[list[str]] = None  # File IDs (optional, for filtering specific files)
    auto_select: bool = False  # Whether to use smart agent selection


class ChatResponse(BaseModel):
    """Response to non-streaming chat"""
    answer: str
    session_id: Optional[str] = None
    workflow_steps: list = []


class WorkflowStepType(str, Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    RESULT = "result"
    ERROR = "error"


class StreamEventType(str, Enum):
    START = "start"
    DELTA = "delta"
    THINKING = "thinking"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    TOOL_LOG = "tool_log"  # Add a new event type for tool logs
    RUN_ITEM = "run_item"
    WORKFLOW_UPDATE = "workflow_update"
    DONE = "done"
    ERROR = "error"
    EXCEL_AGENT_EVENT = "excel_agent_event"  # For Excel Agent
