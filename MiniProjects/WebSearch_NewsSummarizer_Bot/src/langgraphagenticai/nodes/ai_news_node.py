"""
AI News node implementation for LangGraph v1.x.

Fetches AI news using Tavily API and summarizes it using an LLM.
Follows v1.x patterns with proper state management.

Uses TavilySearch from langchain_tavily for proper integration.
"""

from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
import os
import json
from src.langgraphagenticai.state.state import State


class AINewsNode:
    """
    Node that fetches and summarizes AI news using Tavily API.
    
    Workflow:
    1. Fetch news based on timeframe (daily/weekly/monthly)
    2. Summarize news using LLM
    3. Return summary in markdown format
    """
    
    def __init__(self, model: BaseChatModel, tavily_api_key: str = None):
        """
        Initialize AI News node.
        
        Args:
            model: Language model for summarizing news
            tavily_api_key: Tavily API key (optional, reads from env if not provided)
        """
        self.llm = model
        # Get API key from parameter or environment
        api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found. "
                "Please provide it via the sidebar or set it as an environment variable."
            )
        
        # Use TavilySearch from langchain_tavily for proper integration
        # According to docs: time_range accepts "day", "week", "month", or "year"
        self.tavily_tool = TavilySearch(
            api_key=api_key,
            max_results=20,
            topic="news",  # Use "news" topic for news-specific results
            search_depth="advanced",  # More comprehensive search
            include_answer=True,  # Include AI-generated answer
            include_raw_content=False,  # Exclude full HTML to reduce tokens
            include_images=False  # Text-only for now
        )
    
    def fetch_news(self, state: State) -> Dict[str, Any]:
        """
        Fetch AI news based on timeframe from state.
        
        Args:
            state: Current graph state containing timeframe
            
        Returns:
            Dict with 'news_data' key containing fetched news articles
        """
        # Extract timeframe from state
        timeframe = state.get('timeframe', 'weekly').lower()
        
        # Map timeframe to Tavily time_range parameter
        # According to docs: time_range accepts "day", "week", "month", or "year"
        time_range_map = {
            'daily': 'day',
            'weekly': 'week', 
            'monthly': 'month'
        }
        
        time_range = time_range_map.get(timeframe, 'week')
        
        # Query for AI news
        query = "Top Artificial Intelligence (AI) technology news India and globally"
        
        try:
            # Invoke TavilySearch tool with query and time_range
            # According to docs: time_range can be set during invocation
            # The tool accepts: query (required), and optionally time_range, include_images, etc.
            tool_result = self.tavily_tool.invoke({
                "query": query,
                "time_range": time_range
            })
            
            # Parse the tool result
            # TavilySearch can return different formats depending on how it's invoked
            result_data = None
            
            if isinstance(tool_result, str):
                # If it's a JSON string, parse it
                try:
                    result_data = json.loads(tool_result)
                except json.JSONDecodeError:
                    # If it's not JSON, it might be plain text content
                    print(f"Warning: Tavily returned non-JSON string: {tool_result[:200]}")
                    result_data = {"results": [], "answer": tool_result}
            elif hasattr(tool_result, 'content'):
                # ToolMessage object - extract content
                try:
                    content = tool_result.content
                    if isinstance(content, str):
                        result_data = json.loads(content)
                    else:
                        result_data = content
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    print(f"Warning: Could not parse ToolMessage content: {e}")
                    result_data = {"results": [], "answer": str(tool_result.content) if hasattr(tool_result, 'content') else str(tool_result)}
            elif isinstance(tool_result, dict):
                # Already a dict
                result_data = tool_result
            else:
                # Unknown format, try to convert
                print(f"Warning: Unexpected tool result type: {type(tool_result)}")
                result_data = {"results": [], "answer": str(tool_result)}
            
            # Extract news results from the response
            # TavilySearch returns results in 'results' key according to docs
            news_data = result_data.get('results', []) if result_data else []
            
            # Debug: print what we got
            print(f"Fetched {len(news_data)} news articles")
            if news_data:
                print(f"First article keys: {news_data[0].keys() if news_data else 'N/A'}")
            
            # If no results but there's an answer, create a result from it
            if not news_data and result_data and result_data.get('answer'):
                news_data = [{
                    'content': result_data.get('answer', ''),
                    'url': '',
                    'title': 'AI News Summary',
                    'published_date': ''
                }]
            
        except Exception as e:
            # Log error with full traceback for debugging
            import traceback
            print(f"Error fetching news: {e}")
            traceback.print_exc()
            news_data = []
        
        return {
            "news_data": news_data,
            "timeframe": timeframe
        }
    
    def summarize_news(self, state: State) -> Dict[str, Any]:
        """
        Summarize fetched news using LLM.
        
        Args:
            state: Current graph state containing 'news_data'
            
        Returns:
            Dict with 'summary' key containing markdown-formatted summary
        """
        news_items = state.get('news_data', [])
        timeframe = state.get('timeframe', 'weekly')
        
        if not news_items:
            return {
                "summary": f"# {timeframe.capitalize()} AI News Summary\n\nNo news articles found for the selected timeframe."
            }
        
        # Create prompt template for summarization
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Summarize AI news articles into markdown format. For each item include:
            - Date in **YYYY-MM-DD** format in IST timezone
            - Concise sentences summary from latest news
            - Sort news by date wise (latest first)
            - Source URL as link
            Use format:
            ### [Date]
            - [Summary](URL)"""),
            ("user", "Articles:\n{articles}")
        ])
        
        # Format articles for prompt
        articles_str = "\n\n".join([
            f"Title: {item.get('title', 'N/A')}\nContent: {item.get('content', '')}\nURL: {item.get('url', 'N/A')}\nDate: {item.get('published_date', 'N/A')}"
            for item in news_items
        ])
        
        try:
            # Generate summary using LLM
            # Use invoke with messages from the prompt template
            messages = prompt_template.format_messages(articles=articles_str)
            response = self.llm.invoke(messages)
            
            # Extract content from AIMessage
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Format summary with title
            summary = f"# {timeframe.capitalize()} AI News Summary\n\n{content}"
            
        except Exception as e:
            # If summarization fails, return a basic summary
            print(f"Error summarizing news: {e}")
            summary = f"# {timeframe.capitalize()} AI News Summary\n\nError generating summary: {str(e)}\n\nFound {len(news_items)} news articles."
        
        return {
            "summary": summary
        }

