"""
Search tool implementation using Tavily API.

Uses the official langchain-tavily package (recommended over deprecated langchain_community version).
"""

from langchain_tavily import TavilySearch
from typing import Optional
import os


def get_tavily_search_tool(api_key: Optional[str] = None, max_results: int = 3) -> TavilySearch:
    """
    Initialize and return Tavily search tool.
    
    Tavily provides focused web search optimized for LLMs, returning
    relevant, concise content rather than raw web pages.
    
    Args:
        api_key: Tavily API key (if None, reads from TAVILY_API_KEY env var)
        max_results: Maximum number of search results to return (default: 3)
        
    Returns:
        TavilySearch tool instance configured for use with agents
        
    Raises:
        ValueError: If TAVILY_API_KEY is not provided or found in environment
    """
    # Try to get API key from parameter, then environment
    final_api_key = api_key or os.environ.get("TAVILY_API_KEY")
    
    if not final_api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. "
            "Please provide it via the sidebar or set it as an environment variable."
        )
    
    # TavilySearch from langchain-tavily (new recommended package)
    # This is the official integration that receives continuous updates
    search_tool = TavilySearch(
        api_key=final_api_key,  # Pass API key explicitly
        max_results=max_results,
        topic="general",           # Category: "general", "news", or "finance"
        search_depth="advanced",   # "basic" or "advanced" (more comprehensive)
        include_answer=True,       # Include AI-generated answer summary
        include_raw_content=False, # Exclude full HTML (reduces tokens)
        include_images=False       # Text-only for now
    )
    
    return search_tool