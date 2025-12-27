# AI Blog Generation

A LangGraph v1.x application for generating blog posts from topics with optional translation support.

## Features

1. **Basic Blog Generation**: Generate blog posts from a topic

   - Graph: `START → title_creation → content_generation → END`

2. **Blog Generation with Translation**: Generate and translate blogs to Hindi or French
   - Graph: `START → title_creation → content_generation → route → (hindi_translation | french_translation) → END`

## Architecture

Built using LangGraph v1.x patterns:

- **State Management**: TypedDict with Pydantic models for type safety
- **Nodes**: Modular functions that process state and return updates
- **Graphs**: StateGraph with conditional edges for routing
- **LLM**: Groq LLM (GPT-OSS-120B) for generation

## Setup

### Prerequisites

- Python 3.13+
- Groq API key
- LangSmith API key (optional, for tracing)

### Environment Variables

Create a `.env` file in the project root (`/Users/yashsarode/Downloads/Personal Projects/Python/LangChain-LangGraph/.env`) with:

```env
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key  # Optional
```

The application automatically loads `.env` from the project root, even when running from subdirectories.

### Installation

```bash
cd MiniProjects/AI_Blog_Generation
uv sync  # or pip install -e .
```

## Usage

### FastAPI Server

Start the server:

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### POST `/blogs`

Generate a blog post.

**Request Body:**

```json
{
  "topic": "Artificial Intelligence in Healthcare",
  "language": "hindi" // Optional: "hindi" or "french"
}
```

**Response:**

```json
{
  "data": {
    "topic": "Artificial Intelligence in Healthcare",
    "blog": {
      "title": "The Future of AI in Healthcare: Transforming Patient Care",
      "content": "# The Future of AI in Healthcare...\n\n[Full blog content in Markdown]"
    },
    "current_language": "hindi" // Only if language was provided
  }
}
```

#### GET `/health`

Health check endpoint.

### Testing with Postman

1. Start the FastAPI server
2. Create a POST request to `http://localhost:8000/blogs`
3. Set body to JSON:
   ```json
   {
     "topic": "Python Programming",
     "language": "french"
   }
   ```

### LangGraph Studio

The graph can be visualized and tested in LangGraph Studio:

1. Ensure `LANGSMITH_API_KEY` is set in `.env` at the project root
2. Navigate to the project directory:
   ```bash
   cd MiniProjects/AI_Blog_Generation
   ```
3. Start LangGraph Studio:
   ```bash
   langgraph dev
   ```
4. The graph is automatically exported from `src/graphs/graph_builder.py`
5. Studio will open in your browser where you can:
   - Visualize the graph workflow
   - Test with different inputs
   - Debug node execution
   - View state transitions

**Note**: The graph builder includes a pre-compiled graph instance at the module level for Studio compatibility.

## Project Structure

```
MiniProjects/AI_Blog_Generation/
├── app.py                 # FastAPI application
├── src/
│   ├── graphs/
│   │   └── graph_builder.py    # Graph construction
│   ├── llms/
│   │   └── groqllm.py          # LLM wrapper
│   ├── nodes/
│   │   └── blog_node.py        # Node functions
│   └── states/
│       └── blogstate.py        # State schema
└── README.md
```

## Implementation Details

### State Schema

```python
class BlogState(TypedDict, total=False):
    topic: str
    blog: Optional[Blog]  # Pydantic model with title and content
    current_language: Optional[str]  # For translation routing
```

### Node Functions

- `title_creation`: Generates SEO-friendly blog title
- `content_generation`: Generates full blog content in Markdown
- `translation`: Translates blog content to target language
- `route`: Prepares state for conditional routing
- `route_decision`: Determines translation path (hindi/french)

### Graph Patterns

**Basic Graph (Topic Only):**

- Linear flow with no conditional logic
- Simple state updates through nodes

**Language Graph (With Translation):**

- Uses conditional edges for language routing
- Separate translation nodes for each language
- Maintains original title in translations

## Example Usage

### Basic Blog Generation

```bash
curl -X POST http://localhost:8000/blogs \
  -H "Content-Type: application/json" \
  -d '{"topic": "Machine Learning Basics"}'
```

### Blog with Translation

```bash
curl -X POST http://localhost:8000/blogs \
  -H "Content-Type: application/json" \
  -d '{"topic": "Python Web Development", "language": "hindi"}'
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Missing or invalid input (e.g., empty topic, unsupported language)
- **500 Internal Server Error**: LLM or graph execution errors

## Notes

- The application uses LangGraph v1.x patterns (StateGraph, TypedDict, reducers)
- Blog content is generated in Markdown format
- Translations preserve Markdown formatting and structure
- State is managed through TypedDict for type safety
- Environment variables are loaded from project root automatically
- LangGraph Studio integration for visualization and debugging
- FastAPI provides automatic API documentation at `/docs` endpoint
