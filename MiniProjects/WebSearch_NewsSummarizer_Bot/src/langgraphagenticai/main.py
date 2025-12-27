import streamlit as st
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.llms.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """

    # Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return
    
    usecase = user_input.get("selected_usecase")
    
    # Handle AI News use case with button click
    if usecase == "AI News":
        if st.session_state.get("IsFetchButtonClicked", False):
            timeframe = st.session_state.get("timeframe", "Weekly")
            
            try:
                # Configure The LLM
                obj_llm_config = GroqLLM(user_controls_input=user_input)
                model = obj_llm_config.get_llm_model()

                if not model:
                    st.error("Error: LLM model could not be initialized")
                    return
                
                # Get Tavily API key for AI News
                tavily_api_key = user_input.get("TAVILY_API_KEY")
                
                if not tavily_api_key:
                    st.error("Error: TAVILY_API_KEY is required for AI News")
                    return
                
                # Graph Builder with API key
                graph_builder = GraphBuilder(model, tavily_api_key=tavily_api_key)
                
                try:
                    graph = graph_builder.setup_graph(usecase)
                    DisplayResultStreamlit(usecase, graph, timeframe).display_result_on_ui()
                    # Reset button state after processing
                    st.session_state.IsFetchButtonClicked = False
                except Exception as e:
                    st.error(f"Error: Graph setup failed - {e}")
                    st.session_state.IsFetchButtonClicked = False
                    return

            except Exception as e:
                st.error(f"Error: Application failed - {e}")
                st.session_state.IsFetchButtonClicked = False
                return
        return
    
    # Handle other use cases with chat input
    user_message = st.chat_input("Enter your message:")

    if user_message:
        try:
            # Configure The LLM
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: LLM model could not be initialized")
                return
            
            # Initialize and set up the graph based on use case
            if not usecase:
                st.error("Error: No use case selected.")
                return
            
            # Get Tavily API key if using web search or AI News
            tavily_api_key = user_input.get("TAVILY_API_KEY") if usecase in ["Chatbot With Web", "AI News"] else None
            
            # Graph Builder with API key
            graph_builder = GraphBuilder(model, tavily_api_key=tavily_api_key)
            
            try:
                graph = graph_builder.setup_graph(usecase)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph setup failed - {e}")
                return

        except Exception as e:
            st.error(f"Error: Application failed - {e}")
            return