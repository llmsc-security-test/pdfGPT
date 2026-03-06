# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

pdfGPT is an Agentic RAG application for chatting with PDF documents.
It uses a ReAct-style agent that searches, retrieves, and reasons over document content
before generating answers with page citations.

| Service | File | Port | Start Command |
|---------|------|------|---------------|
| Gradio Frontend | `app.py` | 7860 | `python app.py` |
| FastAPI REST API | `api.py` | 8080 | `python api.py` |

### Supported LLM Providers

OpenAI, Anthropic, Google Gemini, Groq (free tier), Mistral, Cohere, Ollama (local).
All routing handled by LiteLLM. See `config.py` `ModelRegistry` for the full model list.

### Python Environment

- **Python 3.10** via deadsnakes PPA. Venv at `/workspace/.venv`.
- Activate: `source /workspace/.venv/bin/activate`

### Running Services

```bash
source /workspace/.venv/bin/activate
python app.py   # Gradio UI on :7860
python api.py   # REST API on :8080
```

API health check: `curl http://localhost:8080/healthz`

### Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI models. Also used as default when no key is entered in the UI.
- Other provider keys (optional): `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`

### Architecture

```
core/document.py     - PDF loading, text extraction, chunking (PyMuPDF)
core/embeddings.py   - Local embeddings via sentence-transformers (all-MiniLM-L6-v2)
core/vectorstore.py  - ChromaDB vector store for semantic search
core/llm.py          - LLM abstraction via LiteLLM
agents/tools.py      - Agent tools: search_document, get_page, final_answer
agents/rag_agent.py  - ReAct agent with forced initial search
config.py            - Model registry and app configuration
```

### Lint and Tests

- Lint: `flake8 config.py core/ agents/ app.py api.py --max-line-length=120`
- No automated test suite exists. Test manually via the Gradio UI or REST API.

### Key Design Notes

- The agent always performs a forced document search before LLM reasoning to prevent
  the model from answering without consulting the document.
- Embeddings are generated locally (no API calls) using sentence-transformers.
  The model downloads on first use (~80MB).
- ChromaDB runs in-memory. Document state is lost when the process restarts.
