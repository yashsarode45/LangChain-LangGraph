"""
FastAPI application for AI Blog Generation.

Provides REST API endpoints for:
- Basic blog generation (topic only)
- Blog generation with translation (topic + language)

Test via Postman or LangGraph Studio.
"""

import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    # Fallback: try current directory
    load_dotenv(override=True)

# Set up LangSmith for tracing (if API key is available)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Blog-Agentic-App"

app = FastAPI(
    title="AI Blog Generation API",
    description="Generate blogs from topics with optional translation support",
    version="1.0.0"
)


@app.post("/blogs")
async def create_blogs(request: Request):
    """
    Generate a blog post based on topic and optional language.
    
    Request body:
        {
            "topic": "string (required)",
            "language": "string (optional) - 'hindi' or 'french'"
        }
    
    Returns:
        {
            "data": {
                "topic": "...",
                "blog": {
                    "title": "...",
                    "content": "..."
                },
                "current_language": "..." (if provided)
            }
        }
    """
    try:
        data = await request.json()
        topic = data.get("topic", "").strip()
        language = data.get("language", "").strip().lower()
        
        if not topic:
            raise HTTPException(
                status_code=400,
                detail="Topic is required"
            )
        
        # Initialize LLM
        groq_llm = GroqLLM()
        llm = groq_llm.get_llm()
        
        # Build graph based on use case
        graph_builder = GraphBuilder(llm)
        
        if topic and language:
            # Blog generation with translation
            if language not in ["hindi", "french"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {language}. Supported: 'hindi', 'french'"
                )
            
            graph = graph_builder.setup_graph(usecase="language")
            initial_state = {
                "topic": topic,
                "current_language": language,
                "blog": None
            }
        else:
            # Basic blog generation
            graph = graph_builder.setup_graph(usecase="topic")
            initial_state = {
                "topic": topic,
                "blog": None
            }
        
        # Invoke the graph
        result_state = graph.invoke(initial_state)
        
        # Convert Blog Pydantic model to dict for JSON response
        blog_data = None
        if result_state.get("blog"):
            blog = result_state["blog"]
            blog_data = {
                "title": blog.title,
                "content": blog.content
            }
        
        response_data = {
            "topic": result_state.get("topic", topic),
            "blog": blog_data,
        }
        
        if language:
            response_data["current_language"] = result_state.get("current_language", language)
        
        return JSONResponse(content={"data": response_data})
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-blog-generation"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

