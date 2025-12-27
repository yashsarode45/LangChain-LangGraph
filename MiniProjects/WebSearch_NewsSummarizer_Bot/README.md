# WebSearch NewsSummarizer Bot

A LangGraph application demonstrating stateful agentic AI workflows with three distinct use cases. Built with LangChain and LangGraph v1.x, this project showcases how to construct different graph patterns from simple linear workflows to tool enabled agents.

## Overview

This project implements three agent architectures using LangGraph's state graph patterns:

- **Basic Chatbot**: A single node workflow demonstrating the fundamental LangGraph pattern
- **Chatbot With Web**: A tool enabled agent using LangChain's `create_agent` with Tavily web search
- **AI News Summarizer**: A sequential workflow that fetches and summarizes AI news articles

## LangGraph Architecture

The application demonstrates core LangGraph v1.x concepts for building stateful agent workflows.

### State Schema

The state uses TypedDict with annotated reducers for intelligent state updates:

```python
class State(TypedDict, total=False):
    messages: Annotated[List, add_messages]
    news_data: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    timeframe: Optional[str]
```

The `add_messages` reducer automatically appends new messages to conversation history, handles deduplication by message ID, and converts various input formats to proper Message instances. This enables seamless multi turn conversations without manual message management.

### Graph Construction Patterns

**Basic Chatbot**

A linear workflow using StateGraph with explicit edges:

```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", basic_chatbot_node.process)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

This demonstrates the simplest LangGraph pattern where state flows through a single node. Each node receives the current state and returns updates that are merged into the shared state.

**Tool Enabled Chatbot**

Uses LangChain's `create_agent` which returns a compiled LangGraph. The `create_agent` function automatically builds a ReAct style reasoning loop where the model decides when to call tools, executes them, and synthesizes final answers. This pattern abstracts away the manual graph construction while still leveraging LangGraph's state management.

The agent graph structure is handled internally:

- Model node receives user query
- Model decides whether to use tools
- Tool node executes if needed
- Model synthesizes final answer from tool results

**AI News Summarizer**

A sequential workflow demonstrating state passing between nodes:

```python
graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
graph_builder.add_edge(START, "fetch_news")
graph_builder.add_edge("fetch_news", "summarize_news")
graph_builder.add_edge("summarize_news", END)
```

Each node updates the shared state. The fetch_news node adds news_data to state, and summarize_news reads that data to generate a summary. This pattern shows how state flows through multiple nodes in a pipeline.

### Node Implementation

Nodes follow LangGraph v1.x conventions:

- Accept state as input parameter
- Return dictionary with state updates
- Never mutate state directly
- Handle errors gracefully

Each node is a pure function that transforms state, making the graph predictable and testable. State updates are merged automatically based on the reducer annotations in the state schema.

### Streaming Execution

The application uses LangGraph's streaming capabilities to provide real time feedback. With `stream_mode="values"`, each event contains the accumulated state after each node execution. This allows the UI to display progress and intermediate results as the graph executes.

## Features

**Basic Chatbot**

Simple conversational interface with automatic conversation history management through LangGraph's state persistence. Messages are automatically appended and deduplicated.

**Chatbot With Web**

Real time web search integration using Tavily API. The agent automatically decides when to search based on the query. Tool calls are visible in the UI, showing the agent's reasoning process. Uses LangChain's `create_agent` which handles the ReAct loop automatically.

**AI News Summarizer**

Timeframe based news fetching with automated summarization. Users select daily, weekly, or monthly timeframes. The graph fetches articles, processes them through the LLM, and generates markdown formatted summaries with dates and source links.

## Setup

### Prerequisites

- Python 3.13 or higher
- Groq API key for LLM access
- Tavily API key for web search and news fetching

### Installation

Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Configuration

Set API keys as environment variables or in a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

API keys can also be entered directly in the Streamlit UI sidebar.

### Running

Start the Streamlit application:

```bash
streamlit run app.py
```

## Usage

**Basic Chatbot**

Select "Basic Chatbot" from the dropdown and start chatting. Conversation history is maintained automatically through LangGraph's state management.

**Chatbot With Web**

Select "Chatbot With Web" and provide your Tavily API key. Ask questions requiring current information. The agent decides when to search and incorporates results into responses.

**AI News Summarizer**

Select "AI News", provide your Tavily API key, choose a timeframe, and click "Fetch Latest AI News". The graph executes sequentially: fetching news, then summarizing it.

## Technical Details

### LangGraph Patterns Demonstrated

**StateGraph Construction**

Building graphs with explicit state schemas using TypedDict. The state schema defines what data flows through the graph and how updates are merged.

**State Reducers**

Using `add_messages` reducer for automatic message handling. Reducers define how state updates are merged, enabling accumulation patterns without manual list management.

**Node Functions**

Stateless functions that accept state and return updates. This pattern makes nodes testable and composable.

**Edge Definitions**

Explicit control flow using START and END constants. Edges define the execution path through the graph.

**Streaming**

Real time state updates during graph execution using `stream_mode="values"`. Each event contains the accumulated state, allowing progressive UI updates.

**Tool Integration**

Using `create_agent` for tool enabled workflows. This pattern abstracts ReAct loop construction while maintaining LangGraph's state management benefits.

### Project Structure

```
src/langgraphagenticai/
├── graph/
│   └── graph_builder.py      # Graph construction and compilation
├── nodes/
│   ├── basic_chatbot_node.py # Simple chatbot node
│   ├── chatbot_with_tool_node.py # Tool enabled agent wrapper
│   └── ai_news_node.py       # News fetching and summarization nodes
├── state/
│   └── state.py              # State schema with TypedDict
├── tools/
│   └── search_tool.py        # Tavily search tool wrapper
├── ui/
│   └── streamlitui/          # Streamlit UI components
└── main.py                   # Application entry point
```

## Dependencies

Key dependencies:

- `langchain` and `langgraph`: Core frameworks for building agents
- `langchain-tavily`: Tavily search integration
- `langchain-groq`: Groq LLM integration
- `streamlit`: Web UI framework
- `pydantic`: Data validation and type safety

## Learning Resources

This project demonstrates practical LangGraph v1.x patterns. For deeper understanding, refer to the LangGraph documentation on state management, graph building, and the building blocks notebook for fundamental concepts.
