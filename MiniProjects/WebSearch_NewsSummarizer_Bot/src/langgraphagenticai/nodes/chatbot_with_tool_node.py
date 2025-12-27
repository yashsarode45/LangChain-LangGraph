"""
Tool-enabled chatbot node using LangChain's create_agent.

In v1.x, create_agent handles the entire tool-calling loop automatically,
including invoking tools and routing between model and tool execution.
"""

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Any, Optional


class ChatbotWithToolNode:
    """
    Agent-based chatbot with tool calling capabilities.
    
    Uses create_agent from v1.x which builds a complete ReAct-style agent
    that can decide when to use tools and synthesize final answers.
    """
    
    def __init__(self, model: BaseChatModel, tools: List[Any], tavily_api_key: Optional[str] = None):
        """
        Initialize tool-enabled chatbot agent.
        
        Args:
            model: Language model for reasoning and generation
            tools: List of tools the agent can use (e.g., search, calculator)
            tavily_api_key: Tavily API key (optional, for passing to tools)
        """
        self.llm = model
        self.tools = tools
        self.tavily_api_key = tavily_api_key
        
        # System prompt guides the agent's behavior
        # Clear instructions improve tool usage decisions
        system_prompt = """You are a helpful AI assistant with access to web search.

When answering questions:
1. If the question requires current information, recent events, or real-time data, use the search tool
2. If you can answer from your training knowledge, respond directly
3. Always cite sources when using search results
4. Be concise but thorough in your responses

Remember: Your knowledge cutoff is January 2025. For anything more recent or requiring real-time data, use search."""

        # create_agent builds a complete graph with:
        # - Model node that can call tools
        # - Tools node that executes tool calls
        # - Automatic routing between them
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
    
    def get_agent(self):
        """
        Return the compiled agent graph.
        
        The agent is already a compiled graph from create_agent,
        so it can be used directly for invocation or streaming.
        
        Returns:
            Compiled LangGraph agent ready for execution
        """
        return self.agent