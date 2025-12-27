"""
State schema for LangGraph agent.

Uses TypedDict for type safety and add_messages reducer for proper message handling.
In v1.x, add_messages intelligently appends new messages while handling deduplication.
"""

from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    """
    Core state schema for the agent graph.
    
    Attributes:
        messages: Conversation history with automatic message appending via add_messages.
                  The reducer handles both dict and Message object formats.
        news_data: List of news articles (for AI News use case)
        summary: News summary in markdown format (for AI News use case)
        timeframe: Time frame for news fetching - 'daily', 'weekly', or 'monthly'
    """
    messages: Annotated[List, add_messages]
    news_data: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    timeframe: Optional[str]