"""
Graph builder for blog generation workflows.

Constructs LangGraph StateGraph instances for:
1. Basic blog generation (topic only)
2. Blog generation with translation (topic + language)
"""
from src.llms.groqllm import GroqLLM
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Literal, Any

from src.states.blogstate import BlogState
from src.nodes.blog_node import BlogNode


class GraphBuilder:
    """
    Builds and compiles LangGraph workflows for blog generation.
    
    Supports two use cases:
    - "topic": Basic blog generation from topic
    - "language": Blog generation with translation
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize graph builder with LLM.
        
        Args:
            llm: Language model instance for blog generation
        """
        self.llm = llm
        self.blog_node = BlogNode(llm)
    
    def build_topic_graph(self) -> StateGraph:
        """
        Build a graph for basic blog generation (topic only).
        
        Graph structure:
            START -> title_creation -> content_generation -> END
        
        Returns:
            StateGraph: Compiled graph for topic-based blog generation
        """
        workflow = StateGraph(BlogState)
        
        # Add nodes
        workflow.add_node("title_creation", self.blog_node.title_creation)
        workflow.add_node("content_generation", self.blog_node.content_generation)
        
        # Add edges
        workflow.add_edge(START, "title_creation")
        workflow.add_edge("title_creation", "content_generation")
        workflow.add_edge("content_generation", END)
        
        return workflow
    
    def build_language_graph(self) -> StateGraph:
        """
        Build a graph for blog generation with translation.
        
        Graph structure:
            START -> title_creation -> content_generation -> route -> 
            (hindi_translation | french_translation) -> END
        
        Returns:
            StateGraph: Compiled graph for blog generation with translation
        """
        workflow = StateGraph(BlogState)
        
        # Add nodes
        workflow.add_node("title_creation", self.blog_node.title_creation)
        workflow.add_node("content_generation", self.blog_node.content_generation)
        workflow.add_node("route", self.blog_node.route)
        workflow.add_node("hindi_translation", self.blog_node.translation)
        workflow.add_node("french_translation", self.blog_node.translation)
        
        # Add edges
        workflow.add_edge(START, "title_creation")
        workflow.add_edge("title_creation", "content_generation")
        workflow.add_edge("content_generation", "route")
        
        # Conditional edge: route to appropriate translation node
        workflow.add_conditional_edges(
            "route",
            self.blog_node.route_decision,
            {
                "hindi": "hindi_translation",
                "french": "french_translation",
            }
        )
        
        # Both translation paths end
        workflow.add_edge("hindi_translation", END)
        workflow.add_edge("french_translation", END)
        
        return workflow
    
    def setup_graph(self, usecase: Literal["topic", "language"]) -> Any:
        """
        Setup and compile graph for the specified use case.
        
        Args:
            usecase: "topic" for basic blog generation, "language" for translation
            
        Returns:
            Compiled graph ready for invocation
            
        Raises:
            ValueError: If usecase is not recognized
        """
        if usecase == "topic":
            graph = self.build_topic_graph()
        elif usecase == "language":
            graph = self.build_language_graph()
        else:
            raise ValueError(f"Unknown use case: {usecase}. Supported: 'topic', 'language'")
        
        return graph.compile()



    
# Initialize LLM
groq_llm = GroqLLM()
llm = groq_llm.get_llm()

# Build language graph for Studio
graph_builder = GraphBuilder(llm)
graph = graph_builder.build_language_graph().compile()

# Export for LangGraph Studio
# Studio will use this graph for visualization and testing

