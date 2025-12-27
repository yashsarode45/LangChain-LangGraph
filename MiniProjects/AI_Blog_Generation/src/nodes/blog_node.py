"""
Blog generation nodes for LangGraph.

Each node represents a step in the blog generation workflow:
- Title creation
- Content generation
- Translation (for multi-language support)
- Routing (for conditional translation)
"""

from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.graph import END

from src.states.blogstate import BlogState, Blog


class BlogNode:
    """
    Node functions for blog generation workflow.
    
    Each method represents a node in the LangGraph that processes state
    and returns updates.
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize blog node with LLM.
        
        Args:
            llm: Language model instance for generation
        """
        self.llm = llm
    
    def title_creation(self, state: BlogState) -> dict:
        """
        Create a blog title based on the topic.
        
        Args:
            state: Current graph state containing topic
            
        Returns:
            dict: State update with blog title
        """
        if "topic" not in state or not state["topic"]:
            raise ValueError("Topic is required for title creation")
        
        topic = state["topic"]
        
        prompt = f"""You are an expert blog content writer. Use Markdown formatting. 
Generate a creative and SEO-friendly blog title for the topic: {topic}

The title should be:
- Engaging and attention-grabbing
- SEO-optimized
- Clear and descriptive
- Between 50-70 characters

Return only the title, no additional text."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip()
        
        # Initialize blog object if it doesn't exist
        blog = state.get("blog")
        if blog is None:
            blog = Blog(title=title, content="")
        else:
            blog = Blog(title=title, content=blog.content)
        
        return {"blog": blog}
    
    def content_generation(self, state: BlogState) -> dict:
        """
        Generate blog content based on topic and title.
        
        Args:
            state: Current graph state containing topic and blog title
            
        Returns:
            dict: State update with blog content
        """
        if "topic" not in state or not state["topic"]:
            raise ValueError("Topic is required for content generation")
        
        topic = state["topic"]
        blog = state.get("blog")
        
        if not blog or not blog.title:
            raise ValueError("Blog title is required for content generation")
        
        title = blog.title
        
        system_prompt = """You are an expert blog writer. Use Markdown formatting.
Generate detailed, well-structured blog content with:
- Clear headings and subheadings
- Engaging introduction
- Well-organized body paragraphs
- Conclusion
- Proper Markdown formatting (headers, lists, emphasis)

Write in a professional yet accessible tone."""
        
        user_prompt = f"""Write a comprehensive blog post about: {topic}

Title: {title}

Generate detailed content with a breakdown covering:
- Introduction to the topic
- Key concepts and explanations
- Examples and use cases
- Best practices or tips
- Conclusion

Use proper Markdown formatting throughout."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content
        
        # Update blog with content
        updated_blog = Blog(title=title, content=content)
        
        return {"blog": updated_blog}
    
    def translation(self, state: BlogState) -> dict:
        """
        Translate blog content to the specified language.
        
        This node is called for both hindi_translation and french_translation.
        It reads the current_language from state to determine target language.
        
        Args:
            state: Current graph state containing blog and target language
            
        Returns:
            dict: State update with translated blog content
        """
        blog = state.get("blog")
        current_language = state.get("current_language", "").lower()
        
        if not blog or not blog.content:
            raise ValueError("Blog content is required for translation")
        
        if not current_language:
            raise ValueError("Target language is required for translation")
        
        # Map language names to proper display names
        language_map = {
            "hindi": "Hindi",
            "french": "French",
            "spanish": "Spanish",
            "german": "German"
        }
        
        target_language = language_map.get(current_language, current_language.capitalize())
        
        translation_prompt = f"""Translate the following blog content into {target_language}.

Requirements:
- Maintain the original tone, style, and formatting
- Preserve all Markdown formatting (headers, lists, emphasis, etc.)
- Adapt cultural references and idioms appropriately for {target_language}
- Keep technical terms accurate
- Ensure the translation is natural and fluent
- Preserve the original title in the translation

ORIGINAL TITLE:
{blog.title}

ORIGINAL CONTENT:
{blog.content}

Provide the complete translated content in {target_language}. Include both the translated title and content:"""
        
        messages = [HumanMessage(content=translation_prompt)]
        
        # Try structured output first, fallback to regular output
        try:
            translated_response = self.llm.with_structured_output(Blog).invoke(messages)
            # Verify we got valid content
            if translated_response and translated_response.content:
                translated_blog = translated_response
            else:
                # Fallback: extract from regular response
                response = self.llm.invoke(messages)
                translated_content = response.content
                # Try to extract title if present, otherwise use original
                translated_blog = Blog(title=blog.title, content=translated_content)
        except Exception:
            # Fallback to regular output if structured output fails
            response = self.llm.invoke(messages)
            translated_content = response.content
            translated_blog = Blog(title=blog.title, content=translated_content)
        
        return {"blog": translated_blog}
    
    def route(self, state: BlogState) -> dict:
        """
        Route node that returns the current language for conditional routing.
        
        This node is used to prepare state for conditional edge routing.
        
        Args:
            state: Current graph state
            
        Returns:
            dict: State update (no changes, just passes through)
        """
        return {"current_language": state.get("current_language", "")}
    
    def route_decision(self, state: BlogState) -> str:
        """
        Routing function for conditional edges.
        
        Determines which translation node to route to based on language.
        This function is used by add_conditional_edges to route after the route node.
        
        Args:
            state: Current graph state
            
        Returns:
            str: Name of the next node to execute ("hindi" or "french")
        """
        current_language = state.get("current_language", "").lower()
        
        if current_language == "hindi":
            return "hindi"
        elif current_language == "french":
            return "french"
        else:
            # If language not recognized, default to hindi (or could raise error)
            # For robustness, we'll default to hindi
            return "hindi"

