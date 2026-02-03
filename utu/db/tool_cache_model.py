from typing import Any

from sqlalchemy import JSON
from sqlmodel import Column, Field, Float, SQLModel, String


class ToolCacheModel(SQLModel, table=True):
    __tablename__ = "cache_tool"

    id: int | None = Field(default=None, primary_key=True)

    function: str = Field(sa_column=Column(String))
    args: str | None = Field(default=None, sa_column=Column(String))
    kwargs: str | None = Field(default=None, sa_column=Column(String))
    result: Any | None = Field(default=None, sa_column=Column(JSON))

    cache_key: str = Field(sa_column=Column(String))
    timestamp: int = Field(sa_column=Column(Float))
    datetime: str = Field(sa_column=Column(String))
    execution_time: float = Field(sa_column=Column(Float))
