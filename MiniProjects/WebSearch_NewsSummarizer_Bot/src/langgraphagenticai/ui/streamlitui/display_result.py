"""
Display module for rendering LangGraph execution results in Streamlit.

Handles streaming graph events and formatting them for the chat interface.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Any


class DisplayResultStreamlit:
    """
    Manages display of agent execution results in Streamlit UI.
    
    Streams graph events and renders messages in chat format.
    """
    
    def __init__(self, usecase: str, graph: Any, user_message: str):
        """
        Initialize display handler with execution context.
        
        Args:
            usecase: Agent use case identifier (e.g., "Basic Chatbot")
            graph: Compiled LangGraph instance
            user_message: User's input message to process
        """
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self) -> None:
        """
        Stream graph execution and display results in Streamlit chat interface.
        
        Processes each event from the graph stream and renders messages
        in appropriate chat message containers.
        """
        if self.usecase == "Basic Chatbot":
            # Stream events from graph execution
            # stream_mode="values" returns full state at each step
            for event in self.graph.stream(
                {'messages': [("user", self.user_message)]},
                stream_mode="values"
            ):
                # Each event contains the current state
                for value in event.values():
                    print("Value: ",value)
                  
                    if value:
                        latest_message = value[-1]
                        print("Latest Message: ",latest_message)
                        # Display user message (only once at the start)
                        if isinstance(latest_message, HumanMessage):
                            with st.chat_message("user"):
                                st.write(self.user_message)
                        
                        # Display assistant response
                        elif isinstance(latest_message, AIMessage):
                            with st.chat_message("assistant"):
                                st.write(latest_message.content)
                        
                        # Handle tool messages if needed in future
                        elif isinstance(latest_message, ToolMessage):
                            with st.chat_message("assistant"):
                                st.caption(f"Tool result: {latest_message.content}")
        elif self.usecase == "Chatbot With Web":
            message_placeholder = st.empty()
            tool_status = st.container()
            self._display_tool_chatbot(message_placeholder, tool_status)
        elif self.usecase == "AI News":
            self._display_ai_news()

    def _display_tool_chatbot(self, placeholder, tool_status) -> None:
        """
        Display results for tool-enabled chatbot with tool call visibility.
        
        Shows tool invocations as status messages and final response.
        
        Args:
            placeholder: Streamlit placeholder for final message
            tool_status: Container for tool execution status
        """
        full_response = ""
        tool_calls_made = []
        
        for event in self.graph.stream(
            {'messages': [HumanMessage(content=self.user_message)]},
            stream_mode="values"
        ):
            for value in event.values():
                
                if not value:
                    continue
                
                latest_message = value[-1]
                
                if isinstance(latest_message, HumanMessage):
                            with st.chat_message("user"):
                                st.write(self.user_message)
                # Track tool calls
                elif isinstance(latest_message, AIMessage) and latest_message.tool_calls:
                    for tool_call in latest_message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        
                        if tool_name not in tool_calls_made:
                            tool_calls_made.append(tool_name)
                            with st.chat_message("assistant"):
                                st.info(f"🔍 Searching: {tool_args.get('query', 'N/A')}")
                
                elif isinstance(latest_message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(latest_message.content)
                
                # Handle tool messages if needed in future
                elif isinstance(latest_message, ToolMessage):
                    with st.chat_message("assistant"):
                        st.caption(f"Tool result: {latest_message.content[:500] + "..." if len(latest_message.content) > 500 else latest_message.content}")
    
    def _display_ai_news(self) -> None:
        """
        Display AI News summary results.
        
        Streams graph execution and displays the final summary in markdown format.
        """
        # Stream events from graph execution
        summary = None
        error_message = None
        
        try:
            with st.spinner("Fetching and summarizing news... ⏳"):
                # Initialize state with timeframe
                initial_state = {'timeframe': self.user_message.lower()}
                
                # In stream_mode="values", each event IS the accumulated state dict
                # The final event contains the complete state after all nodes execute
                final_state = None
                
                for event in self.graph.stream(
                    initial_state,
                    stream_mode="values"
                ):
                    # Each event is the accumulated state dict (not a dict with node names)
                    if isinstance(event, dict):
                        final_state = event
                        # Debug: print state to help diagnose issues
                        print(f"State keys: {event.keys()}")
                        
                        # Check for summary in state
                        if 'summary' in event and event['summary']:
                            summary = event['summary']
                            print(f"Found summary: {summary[:100]}...")
                        
                        # Check for errors
                        if 'error' in event:
                            error_message = event['error']
                
                # Use the final state if we didn't get summary during streaming
                if not summary and final_state:
                    if 'summary' in final_state:
                        summary = final_state['summary']
                    else:
                        # Debug: print what we got
                        print(f"Final state keys: {final_state.keys()}")
                        print(f"Final state content: {final_state}")
                
                # Fallback: try invoke if streaming didn't work
                if not summary:
                    print("Trying invoke as fallback...")
                    final_state = self.graph.invoke(initial_state)
                    if isinstance(final_state, dict) and 'summary' in final_state:
                        summary = final_state['summary']
                        print(f"Got summary from invoke: {summary[:100]}...")
        
        except Exception as e:
            error_message = f"Error during execution: {str(e)}"
            print(f"Exception in _display_ai_news: {e}")
            import traceback
            traceback.print_exc()
        
        # Display the summary or error
        if error_message:
            st.error(f"Error: {error_message}")
            st.info("Please check the console for detailed error messages.")
        elif summary:
            st.markdown(summary, unsafe_allow_html=True)
        else:
            st.error("Failed to generate news summary. Please try again.")
            st.info("Debug: Check console output for state information.")