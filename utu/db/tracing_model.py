from collections.abc import Mapping, Sequence
from typing import Any, Literal

# from agents.tracing import FunctionSpanData, GenerationSpanData
from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel, String


# FunctionSpanData
class ToolTracingModel(SQLModel, table=True):
    __tablename__ = "tracing_tool"

    id: int | None = Field(default=None, primary_key=True)
    trace_id: str = ""
    span_id: str = ""

    name: str = ""
    input: Any | None = Field(default=None, sa_column=Column(JSON))
    output: Any | None = Field(default=None, sa_column=Column(JSON))
    mcp_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


# GenerationSpanData
class GenerationTracingModel(SQLModel, table=True):
    __tablename__ = "tracing_generation"

    id: int | None = Field(default=None, primary_key=True)
    trace_id: str = ""
    span_id: str = ""
    type: Literal["chat.completions", "responses"] = Field(default="chat.completions", sa_column=Column(String))

    input: Sequence[Mapping[str, Any]] | None = Field(default=None, sa_column=Column(JSON))
    output: Sequence[Mapping[str, Any]] | None = Field(default=None, sa_column=Column(JSON))
    model: str = Field(sa_column=Column(String))
    model_configs: Mapping[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    usage: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    response_id: str | None = Field(default=None, sa_column=Column(String))
