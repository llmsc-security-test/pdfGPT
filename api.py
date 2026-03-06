"""PDF GPT - Agentic RAG REST API.

FastAPI backend providing programmatic access to the Agentic RAG pipeline.
Supports all LLM providers via LiteLLM.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from config import AppConfig, ModelRegistry
from core.document import PDFProcessor
from core.embeddings import EmbeddingProvider
from core.vectorstore import VectorStore
from core.llm import LLMProvider
from agents.tools import (
    ToolRegistry, SearchDocumentTool, GetPageTool, FinalAnswerTool,
)
from agents.rag_agent import AgenticRAG


class AskURLRequest(BaseModel):
    url: str = Field(..., description="URL of the PDF to process")
    question: str = Field(..., description="Question to ask about the PDF")
    model: str = Field(default="gpt-4o-mini", description="LiteLLM model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)


class AskResponse(BaseModel):
    answer: str
    steps: List[dict] = Field(default_factory=list)
    success: bool = True
    error: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"


class RAGService:
    """Manages the RAG pipeline for the API server."""

    def __init__(self):
        self.config = AppConfig()
        self.pdf_processor = PDFProcessor(
            chunk_size=self.config.default_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.embedding_provider = EmbeddingProvider(self.config.embedding_model)

    def process_query(self, pdf_path: str, question: str,
                      model: str, api_key: Optional[str],
                      temperature: float) -> AskResponse:
        vector_store = VectorStore(self.embedding_provider)
        chunks = self.pdf_processor.chunk_document(pdf_path)
        vector_store.add_chunks(chunks)

        llm = LLMProvider(
            model_id=model,
            api_key=api_key,
            temperature=temperature,
        )

        tools = ToolRegistry()
        tools.register(SearchDocumentTool(vector_store))
        tools.register(GetPageTool(self.pdf_processor, pdf_path))
        tools.register(FinalAnswerTool())

        agent = AgenticRAG(llm, tools, max_iterations=self.config.agent_max_iterations)
        response = agent.query(question)

        steps_data = [
            {
                "step": s.step_number,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:500] if s.observation else "",
            }
            for s in response.steps
        ]

        return AskResponse(
            answer=response.answer,
            steps=steps_data,
            success=response.success,
            error=response.error_message,
        )


app = FastAPI(
    title="PDF GPT - Agentic RAG API",
    description="Chat with PDF documents using AI agents and modern LLMs.",
    version="2.0.0",
)

rag_service = RAGService()


@app.get("/healthz", response_model=HealthResponse)
def health_check():
    return HealthResponse()


@app.get("/models")
def list_models():
    """List all supported LLM providers and models."""
    return ModelRegistry.PROVIDERS


@app.post("/ask_url", response_model=AskResponse)
def ask_url(request: AskURLRequest):
    """Process a question about a PDF from a URL."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    try:
        PDFProcessor.download_pdf(request.url, tmp.name)
        return rag_service.process_query(
            tmp.name, request.question, request.model,
            request.api_key, request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.post("/ask_file", response_model=AskResponse)
async def ask_file(
    file: UploadFile = File(...),
    question: str = "",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
):
    """Process a question about an uploaded PDF file."""
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        return rag_service.process_query(
            tmp.name, question, model, api_key, temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


if __name__ == "__main__":
    config = AppConfig()
    uvicorn.run(app, host="0.0.0.0", port=config.api_port)
