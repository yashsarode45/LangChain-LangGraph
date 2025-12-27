"""
Groq LLM wrapper for blog generation.

Provides a consistent interface for accessing Groq language models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel


class GroqLLM:
    """
    Wrapper for Groq LLM initialization.
    
    Handles environment variable loading from project root.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.7):
        """
        Initialize Groq LLM with API key from environment.
        
        Args:
            model_name: Name of the Groq model to use
            temperature: Temperature setting for generation
        """
        # Load .env from project root (works even if running from subdirectory)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        env_path = project_root / ".env"
        
        if env_path.exists():
            load_dotenv(env_path, override=True)
        else:
            # Fallback: try current directory
            load_dotenv(override=True)
        
        # Get API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        
        # Set environment variable for LangChain
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize the LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )
    
    def get_llm(self) -> BaseChatModel:
        """
        Get the initialized LLM instance.
        
        Returns:
            BaseChatModel: The Groq chat model instance
        """
        return self.llm

