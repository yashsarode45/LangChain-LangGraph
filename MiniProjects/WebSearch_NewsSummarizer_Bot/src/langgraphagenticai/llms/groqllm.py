"""
LLM initialization module for Groq Cloud integration.

Handles model configuration and API key validation for Groq-hosted models.
"""

import os
import streamlit as st
from langchain_groq import ChatGroq
from typing import Dict, Any


class GroqLLM:
    """
    Manages Groq LLM initialization and configuration.
    
    Supports multiple Groq models with API key validation and error handling.
    """
    
    def __init__(self, user_controls_input: Dict[str, Any]):
        """
        Initialize with user-provided configuration.
        
        Args:
            user_controls_input: Dict containing 'GROQ_API_KEY' and 'selected_groq_model'
        """
        self.user_controls_input = user_controls_input

    def get_llm_model(self) -> ChatGroq:
        """
        Initialize and return configured ChatGroq model.
        
        Validates API key from user input or environment variable.
        Falls back to environment variable if user input is empty.
        
        Returns:
            ChatGroq: Configured language model instance
            
        Raises:
            ValueError: If API key is missing or model initialization fails
        """
        try:
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY", "")
            selected_groq_model = self.user_controls_input["selected_groq_model"]
            
            # Validate API key from input or environment
            if not groq_api_key and not os.environ.get("GROQ_API_KEY"):
                st.error("Please enter the Groq API key in the sidebar or set GROQ_API_KEY environment variable")
                raise ValueError("Groq API key not provided")

            # Initialize model with explicit API key
            llm = ChatGroq(
                api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
                model=selected_groq_model
            )

        except KeyError as e:
            raise ValueError(f"Missing required configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error initializing Groq model: {e}")
        
        return llm