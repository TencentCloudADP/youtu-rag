"""Trajectory model for storing agent execution traces with skill extraction support."""

import json
from typing import TYPE_CHECKING

from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    from ..agents.common import TaskRecorder


class TrajectoryModel(SQLModel, table=True):
    """Model for storing agent execution trajectories and skill extraction metadata."""

    __tablename__ = "trajectory"

    id: int | None = Field(default=None, primary_key=True)

    # Basic trace info
    trace_id: str | None = Field(default=None, index=True)
    trace_url: str | None = Field(default=None)

    # Input/Output
    d_input: str | None = Field(default=None)
    d_output: str | None = Field(default=None)
    trajectories: str | None = Field(default=None)

    # Performance
    time_cost: float | None = Field(default=None)

    # Tool calls summary - JSON string of list[dict]
    # Format: [{"tool_name": "search", "input": {...}, "output": {...}, "success": true}, ...]
    tool_calls_summary: str | None = Field(default=None)

    # Skill extraction status
    skill_extracted: bool = Field(default=False, index=True)
    extracted_skill_id: str | None = Field(default=None, index=True)

    # Additional metadata
    agent_name: str | None = Field(default=None, index=True)
    model_name: str | None = Field(default=None)
    tool_count: int | None = Field(default=None)  # Number of tool calls in this trajectory

    @classmethod
    def from_task_recorder(cls, task_recorder: "TaskRecorder") -> "TrajectoryModel":
        """Create TrajectoryModel from TaskRecorder.

        Args:
            task_recorder: TaskRecorder instance containing execution data.

        Returns:
            TrajectoryModel instance ready for database insertion.
        """
        d_input = getattr(task_recorder, "task", "") or getattr(task_recorder, "input", "")
        trajectories = getattr(task_recorder, "trajectories", [])

        # Extract tool calls summary from trajectories
        tool_calls_summary = cls._extract_tool_calls_summary(trajectories)
        tool_count = len(tool_calls_summary) if tool_calls_summary else 0

        return cls(
            trace_id=task_recorder.trace_id,
            trace_url="",
            d_input=d_input,
            d_output=task_recorder.final_output,
            trajectories=json.dumps(trajectories, ensure_ascii=False),
            time_cost=-1,
            skill_extracted=False,
            extracted_skill_id=None,
            tool_count=tool_count,
        ) #tool_calls_summary=json.dumps(tool_calls_summary, ensure_ascii=False) if tool_calls_summary else None,

    @staticmethod
    def _extract_tool_calls_summary(trajectories: list) -> list[dict]:
        """Extract tool calls summary from trajectory data.

        Args:
            trajectories: List of trajectory items from TaskRecorder.

        Returns:
            List of tool call summaries with tool_name, input, output, success.
        """
        tool_calls = []

        if not trajectories:
            return tool_calls

        for item in trajectories:
            if not isinstance(item, dict):
                continue

            # Check for tool call indicators
            item_type = item.get("type", "")

            # Handle function_call type
            if item_type == "function_call" or "function_call" in item:
                func_call = item.get("function_call", item)
                tool_calls.append({
                    "tool_name": func_call.get("name", "unknown"),
                    "input": func_call.get("arguments", {}),
                    "output": func_call.get("output", None),
                    "success": func_call.get("success", True),
                })

            # Handle tool_call type (OpenAI format)
            elif item_type == "tool_call" or "tool_calls" in item:
                for tc in item.get("tool_calls", [item]):
                    if isinstance(tc, dict):
                        func = tc.get("function", tc)
                        tool_calls.append({
                            "tool_name": func.get("name", "unknown"),
                            "input": func.get("arguments", {}),
                            "output": tc.get("output", None),
                            "success": tc.get("success", True),
                        })

            # Handle action type (some frameworks use this)
            elif item_type == "action" or "action" in item:
                action = item.get("action", item)
                tool_calls.append({
                    "tool_name": action.get("tool", action.get("name", "unknown")),
                    "input": action.get("tool_input", action.get("input", {})),
                    "output": item.get("observation", item.get("output", None)),
                    "success": item.get("success", True),
                })

        return tool_calls

    def get_tool_calls(self) -> list[dict]:
        """Get parsed tool calls summary.

        Returns:
            List of tool call dictionaries.
        """
        if not self.tool_calls_summary:
            return []
        try:
            return json.loads(self.tool_calls_summary)
        except json.JSONDecodeError:
            return []

    def get_tool_names(self) -> list[str]:
        """Get list of tool names used in this trajectory.

        Returns:
            List of unique tool names.
        """
        tool_calls = self.get_tool_calls()
        return list({tc.get("tool_name", "unknown") for tc in tool_calls})

    def mark_skill_extracted(self, skill_id: str) -> None:
        """Mark this trajectory as having its skill extracted.

        Args:
            skill_id: ID of the extracted skill in memory store.
        """
        self.skill_extracted = True
        self.extracted_skill_id = skill_id

    def to_skill_extraction_input(self) -> dict:
        """Prepare data for skill extraction.

        Returns:
            Dictionary containing data needed for skill extraction.
        """
        return {
            "trace_id": self.trace_id,
            "question": self.d_input,
            "answer": self.d_output,
            "tool_calls": self.get_tool_calls(),
            "tool_names": self.get_tool_names(),
        }