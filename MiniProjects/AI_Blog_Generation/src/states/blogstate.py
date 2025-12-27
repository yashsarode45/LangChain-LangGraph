"""
State schema for Blog Generation LangGraph.

Uses TypedDict for type safety and proper state management.
In LangGraph v1.x, state updates are handled through reducers.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class Blog(BaseModel):
    """Blog content structure."""
    title: str = Field(description="The title of the blog post")
    content: str = Field(description="The main content of the blog post")


class BlogState(TypedDict, total=False):
    """
    State schema for blog generation graph.
    
    Attributes:
        topic: The topic for blog generation
        blog: Blog object containing title and content
        current_language: Target language for translation (hindi, french, etc.)
    """
    topic: str
    blog: Optional[Blog]
    current_language: Optional[str]

