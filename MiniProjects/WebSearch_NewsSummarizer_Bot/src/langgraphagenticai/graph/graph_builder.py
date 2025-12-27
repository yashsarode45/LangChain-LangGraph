"""
Graph construction module for LangGraph agents.

Builds and compiles state graphs for different agent use cases.
In v1.x, we use START/END constants for manual graphs and create_agent for tool-enabled agents.
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Literal, Any, Optional

from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_tool_node import ChatbotWithToolNode
from src.langgraphagenticai.nodes.ai_news_node import AINewsNode
from src.langgraphagenticai.tools.search_tool import get_tavily_search_tool


class GraphBuilder:
    """
    Constructs and compiles LangGraph state graphs for various agent patterns.
    
    Supports multiple agent architectures:
    - Basic chatbot (no tools)
    - Tool-enabled chatbot (with web search)
    - AI News summarizer (fetches and summarizes news)
    """
    
    def __init__(self, model: BaseChatModel, tavily_api_key: Optional[str] = None):
        """
        Initialize builder with a language model.
        
        Args:
            model: Language model to be used by agent nodes
        """
        self.llm = model
        self.tavily_api_key = tavily_api_key
    def basic_chatbot_build_graph(self) -> Any:
        """
        Construct a basic chatbot graph with single LLM node.
        
        Graph structure:
            START -> chatbot -> END
        
        This is the simplest agent pattern - just a conversational loop
        with no tools, routing, or complex decision making.
        
        Returns:
            Compiled graph for basic chatbot
        """
        # Initialize graph builder with state schema
        graph_builder = StateGraph(State)
        
        # Initialize node with the LLM
        basic_chatbot_node = BasicChatbotNode(self.llm)

        # Add node to graph with its processing function
        graph_builder.add_node("chatbot", basic_chatbot_node.process)
        
        # Define graph flow: START -> chatbot -> END
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        # Compile graph into executable runtime
        return graph_builder.compile()
    
    def chatbot_with_tool_build_graph(self) -> Any:
        """
        Construct a tool-enabled chatbot using create_agent.
        
        Uses LangChain v1.x create_agent which automatically handles:
        - Tool binding to the model
        - ReAct-style reasoning loop
        - Tool execution and result processing
        - Final answer synthesis
        
        Graph structure (handled internally by create_agent):
            User Query -> Model (decides tool use) -> Tools (if needed) -> Model (final answer)
        
        Returns:
            Compiled agent graph with tool calling capabilities
            
        Raises:
            ValueError: If Tavily API key is not provided
        """
        # Initialize Tavily search tool with API key
        # Pass the API key explicitly to avoid environment variable issues
        search_tool = get_tavily_search_tool(
            api_key=self.tavily_api_key,
            max_results=3
        )
        tools = [search_tool]
        
        # Create agent node with tools
        # create_agent returns a compiled graph, not just a node
        chatbot_with_tool = ChatbotWithToolNode(
            model=self.llm,
            tools=tools,
            tavily_api_key=self.tavily_api_key
        )
        
        # Return the compiled agent graph
        # No need to build StateGraph manually - create_agent does it
        return chatbot_with_tool.get_agent()

    def ai_news_build_graph(self) -> Any:
        """
        Construct an AI News summarizer graph.
        
        Graph structure:
            START -> fetch_news -> summarize_news -> END
        
        This graph fetches AI news based on timeframe and summarizes it.
        
        Returns:
            Compiled graph for AI News summarizer
            
        Raises:
            ValueError: If Tavily API key is not provided
        """
        # Initialize AI News node
        ai_news_node = AINewsNode(
            model=self.llm,
            tavily_api_key=self.tavily_api_key
        )
        
        # Initialize graph builder with state schema
        graph_builder = StateGraph(State)
        
        # Add nodes to graph
        graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
        graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
        
        # Define graph flow: START -> fetch_news -> summarize_news -> END
        graph_builder.add_edge(START, "fetch_news")
        graph_builder.add_edge("fetch_news", "summarize_news")
        graph_builder.add_edge("summarize_news", END)
        
        # Compile graph into executable runtime
        return graph_builder.compile()
    
    def setup_graph(self, usecase: Literal["Basic Chatbot", "Chatbot With Web", "AI News"]) -> Any:
        """
        Setup and compile graph for the specified use case.
        
        Acts as a factory method - routes to appropriate graph builder
        based on the use case string.
        
        Args:
            usecase: String identifier for the agent pattern to build.
                    Supported: "Basic Chatbot", "Chatbot With Web", "AI News"
            
        Returns:
            Compiled graph ready for invocation
            
        Raises:
            ValueError: If usecase is not recognized
        """
        if usecase == "Basic Chatbot":
            return self.basic_chatbot_build_graph()
        elif usecase == "Chatbot With Web":
            return self.chatbot_with_tool_build_graph()
        elif usecase == "AI News":
            return self.ai_news_build_graph()
        else:
            raise ValueError(f"Unknown use case: {usecase}. Supported: 'Basic Chatbot', 'Chatbot With Web', 'AI News'")