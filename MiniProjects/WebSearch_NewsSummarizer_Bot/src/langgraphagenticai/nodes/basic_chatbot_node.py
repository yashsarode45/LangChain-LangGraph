"""
Basic chatbot node implementation for LangGraph.

Provides a simple node that processes messages through an LLM without tools.
"""

from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from src.langgraphagenticai.state.state import State


class BasicChatbotNode:
    """
    Node that processes input through an LLM and returns the response.
    
    This is the simplest form of a chatbot - just message in, LLM response out.
    No tool calling, no complex routing, just direct conversation.
    """
    
    def __init__(self, model: BaseChatModel):
        """
        Initialize node with a language model.
        
        Args:
            model: Any LangChain chat model (ChatGroq, ChatOpenAI, etc.)
        """
        self.llm = model

    def process(self, state: State) -> Dict[str, Any]:
        """
        Process the current state through the LLM.
        
        Takes the message history from state, invokes the model,
        and returns the response which will be automatically appended
        to the messages list via the add_messages reducer.
        
        Args:
            state: Current graph state containing message history
            
        Returns:
            Dict with 'messages' key containing the LLM's response.
            The response is wrapped in a list for add_messages compatibility.
        """
        response = self.llm.invoke(state['messages'])
        
        # Return as list for add_messages reducer
        # The reducer will append this to existing messages
        return {"messages": [response]}