# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

pdfGPT is a RAG (Retrieval-Augmented Generation) application with two services:

| Service | File | Port | Start Command |
|---------|------|------|---------------|
| API Backend | `api.py` | 8080 | `lc-serve deploy local api` |
| Gradio Frontend | `app.py` | 7860 | `python app.py` |

### Python Environment

- **Python 3.10** is required (installed via deadsnakes PPA). The codebase targets Python 3.8 (per Dockerfile) but 3.10 is the minimum that works with all resolved dependencies.
- Virtual environment lives at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`.

### Critical Dependency Notes

The `requirements.txt` pins versions that are mutually incompatible on modern systems. The working dependency set requires:

- `langchain-serve==0.0.61` and `jina==3.15.2` must be installed with `--no-build-isolation --no-deps` because jina's transitive dependency `opentelemetry-exporter-prometheus>=1.12.0rc1` does not exist on PyPI (only beta versions are published).
- `pydantic==1.10.x` and `fastapi==0.99.x` are required (lcserve is incompatible with pydantic v2).
- `gradio==3.50.2` is needed because `app.py` uses the deprecated `.style()` method removed in gradio 4.x.
- `tensorflow==2.15.1` is used (compatible with Python 3.10 and numpy 1.26.x).
- `litellm==0.12.5` is needed for compatibility with `openai==0.27.4`.

### Known Code Bug

`app.py` line 92 (`demo.app.server.timeout = 60000`) references a non-existent Gradio attribute. This line crashes the app on every Gradio version. To run the frontend, use this workaround (does not modify the file):

```bash
source /workspace/.venv/bin/activate
cd /workspace
python -c "
source = open('app.py').read()
source = source.replace('demo.app.server.timeout = 60000', 'pass')
exec(compile(source, 'app.py', 'exec'))
"
```

### Running Services

1. **Start API backend:** `source /workspace/.venv/bin/activate && cd /workspace && lc-serve deploy local api`
2. **Start Gradio frontend** (with workaround): see above
3. **Health check:** `curl http://localhost:8080/healthz` should return `{"status": "ok"}`
4. **API docs:** Available at `http://localhost:8080/docs`

### Lint

No linter is configured in the repo. Use `flake8 api.py app.py --max-line-length=120` for basic checks. Existing lint warnings are present in the codebase.

### External Dependencies

- **OpenAI API key** is required at runtime for the LLM completion step. Without a valid key, the PDF parsing and embedding pipeline still works but answer generation returns an API error.
- **TensorFlow Hub** downloads the Universal Sentence Encoder model on first run (~1GB). After initial download it is cached.
