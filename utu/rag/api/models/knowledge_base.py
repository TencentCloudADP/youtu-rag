"""Models related to knowledge base"""
from typing import Optional
from pydantic import BaseModel


class KnowledgeBaseCreate(BaseModel):
    """Request for creating a knowledge base"""
    name: str
    description: Optional[str] = None


class KnowledgeBaseUpdate(BaseModel):
    """Request for updating a knowledge base"""
    name: Optional[str] = None
    description: Optional[str] = None
