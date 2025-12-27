## LangChain & LangGraph v1.x Examples & Explanations

This repo contains Notebooks that walk through modern LangChain and LangGraph patterns using the latest v1.x APIs.
Content is organized to start from fundamentals and build up on that.
I've created this as I was learning, in a way that it can help anyone new to the Langchain ecosystem to catch up easily.

### Repo Layout

- `LangChain/`: Stepwise LangChain v1.x notebooks (messages, tool calling, agents, RAG, middleware, hybrid search, query enhancement).
- `LangGraph/`: LangGraph v1.x notebooks focused on graph-based agents and orchestration patterns.
- `MiniProjects/`: Practical applications demonstrating LangGraph patterns in real-world scenarios.

### LangChain Track (recommended order)

0. `00_models_messages.ipynb` — Chat models, messages, content blocks.
1. `01_tools_and_tool_calling.ipynb` — Tool creation, structured calls.
2. `02_context_in_tools.ipynb` — Passing context into tools.
3. `03_agents_and_short_term_memory.ipynb` — Basic agents with short-term memory.
4. `04_streaming_with_agents.ipynb` — Streaming token outputs.
5. `05_structured_output_typing.ipynb` — Typed outputs and Pydantic.
6. `06_langchain_rag.ipynb` — RAG pipeline with Chroma.
7. `07_inbuilt_middlewares.ipynb` — Using built-in middleware.
8. `08_custom_middlewares.ipynb` — Authoring custom middleware.
9. `09_multi_agent.ipynb` — Multi-agent coordination.
10. `10_hybridsearch_denseparse.ipynb` — Dense hybrid search flows.
11. `11_hybridsearch_reranking.ipynb` — Hybrid search with reranking.
12. `12_queryenhancement.ipynb` — Query rewriting/enhancement.

### LangGraph Track (recommended order)

1. `01_langgraph_building_blocks.ipynb` — Core graph primitives and state.
2. `02_langgraph_advanced_patterns.ipynb` — Branching, guards, retries, middleware.
3. `03_prompt_chaining_workflow.ipynb` — Prompt chains with graph control.
4. `04_parallel_workflow.ipynb` — Parallel branches and joins.
5. `05_routing_workflow.ipynb` — Conditional routing agents.
6. `06_orchestrator_worker_workflow.ipynb` — Orchestrator/worker topology.
7. `07_evaluator_optimizer_workflow.ipynb` — Evaluator/optimizer loops.
8. `08_agentic_rag.ipynb` — Agentic RAG with tool-based retrieval, document relevance grading, and query rewriting.
9. `09_autonomous_rag.ipynb` — Autonomous RAG with query planning, chain-of-thought reasoning, iterative retrieval, and self-reflection.
10. `10_adaptive_rag.ipynb` — Adaptive RAG that dynamically adjusts retrieval strategy based on query complexity.
11. `11_corrective_rag.ipynb` — Corrective RAG with feedback loops to improve answer quality.
12. `12_multi_agent_rag.ipynb` — Multi-agent RAG with coordinated agents for complex retrieval tasks.
13. `13_kg_neo4j_rag.ipynb` — Knowledge graph RAG using Neo4j for structured knowledge retrieval.

### Requirements

- Python 3.13+
- Core deps: LangChain ≥1.1, LangGraph (via LangChain), Chroma, FAISS, sentence-transformers, Groq/Hugging Face integrations (see `pyproject.toml`).

### Setup

```bash
# using uv (recommended)
uv sync

# or pip
pip install -r <(uv pip compile pyproject.toml)
```

Start Jupyter:

```bash
uv run jupyter lab  # or: uv run jupyter notebook
```

Then open notebooks from `LangChain/` or `LangGraph/` in the listed order.

### Mini Projects

- `WebSearch_NewsSummarizer_Bot/`: A LangGraph application demonstrating stateful agentic AI workflows with three use cases: basic chatbot, web-enabled chatbot, and AI news summarizer. Features Streamlit UI, tool integration with Tavily, and sequential workflow patterns.

### Notes

- Content targets the v1.x APIs (e.g., `create_agent`, content blocks, streamlined namespaces) as outlined in the LangChain v1 release notes.

### References

- LangChain v1.x docs: https://docs.langchain.com/oss/python/langchain/overview
- LangGraph v1 docs: https://docs.langchain.com/oss/python/langgraph/overview
