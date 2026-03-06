"""PDF GPT - Agentic RAG Web Application.

A modern web interface for chatting with PDF documents using AI agents
and any LLM provider (OpenAI, Anthropic, Google, Groq, Ollama, etc.).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from config import ModelRegistry, AppConfig
from core.document import PDFProcessor
from core.embeddings import EmbeddingProvider
from core.vectorstore import VectorStore
from core.llm import LLMProvider
from agents.tools import (
    ToolRegistry, SearchDocumentTool, GetPageTool, FinalAnswerTool,
)
from agents.rag_agent import AgenticRAG


class AppState:
    """Manages the shared application state across UI interactions."""

    def __init__(self):
        self.config = AppConfig()
        self.pdf_processor = PDFProcessor(
            chunk_size=self.config.default_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.embedding_provider = EmbeddingProvider(self.config.embedding_model)
        self.vector_store = VectorStore(self.embedding_provider)
        self.current_pdf_path: Optional[str] = None
        self.current_pdf_name: str = ""
        self.page_count: int = 0

    def load_pdf_from_file(self, file_path: str) -> str:
        self.current_pdf_path = file_path
        self.current_pdf_name = Path(file_path).name
        chunks = self.pdf_processor.chunk_document(file_path)
        self.page_count = self.pdf_processor.get_page_count(file_path)
        self.vector_store.add_chunks(chunks)
        return (
            f"Loaded: {self.current_pdf_name} "
            f"({self.page_count} pages, {len(chunks)} chunks indexed)"
        )

    def load_pdf_from_url(self, url: str) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        PDFProcessor.download_pdf(url, tmp.name)
        return self.load_pdf_from_file(tmp.name)

    def build_agent(self, model_id: str, api_key: str,
                    temperature: float) -> AgenticRAG:
        llm = LLMProvider(
            model_id=model_id,
            api_key=api_key if api_key else None,
            temperature=temperature,
        )
        tools = ToolRegistry()
        tools.register(SearchDocumentTool(self.vector_store))
        if self.current_pdf_path:
            tools.register(GetPageTool(self.pdf_processor, self.current_pdf_path))
        tools.register(FinalAnswerTool())
        return AgenticRAG(llm, tools, max_iterations=self.config.agent_max_iterations)


app_state = AppState()


def load_document(file, url):
    """Load a PDF from file upload or URL."""
    try:
        if file is not None and str(file).strip():
            return app_state.load_pdf_from_file(str(file))
        elif url and url.strip():
            return app_state.load_pdf_from_url(url.strip())
        else:
            return "Please upload a PDF file or enter a URL."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error loading document: {str(e)}"


def get_models_for_provider(provider_name):
    """Update model dropdown when provider changes."""
    models = ModelRegistry.get_models_for_provider(provider_name)
    if models:
        return gr.Dropdown(choices=models, value=models[0])
    return gr.Dropdown(choices=[], value=None)


def chat_with_agent(message, history, provider, model_name,
                    api_key, temperature, show_reasoning):
    """Process a user message through the Agentic RAG pipeline."""
    if not app_state.vector_store.is_loaded:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please load a PDF document first using the panel on the left."},
        ]
        return

    model_id = ModelRegistry.get_model_id(provider, model_name)
    if not model_id:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please select a valid model."},
        ]
        return

    model_info = ModelRegistry.get_model_info(provider, model_name)
    if model_info and model_info.requires_api_key and not api_key:
        env_key = os.environ.get(model_info.api_key_env_var, "")
        if not env_key:
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": (
                    f"API key required for {provider}. Enter it in the settings "
                    f"or set {model_info.api_key_env_var} environment variable."
                )},
            ]
            return
        api_key = env_key

    agent = app_state.build_agent(model_id, api_key, temperature)

    chat_history = []
    if history:
        for turn in history:
            chat_history.append(turn)

    updated_history = history + [{"role": "user", "content": message}]

    reasoning_parts = []

    def on_step(step):
        reasoning_parts.append(step.format_for_display())

    try:
        response = agent.query(message, chat_history=chat_history, on_step=on_step)

        if not response.success:
            answer = f"Agent encountered an error: {response.error_message}"
        else:
            answer = response.answer

        if show_reasoning and reasoning_parts:
            reasoning_text = "\n\n".join(reasoning_parts)
            full_response = (
                f"{answer}\n\n"
                f"---\n"
                f"**Agent Reasoning ({len(response.steps)} steps):**\n"
                f"```\n{reasoning_text}\n```"
            )
        else:
            full_response = answer

        yield updated_history + [{"role": "assistant", "content": full_response}]

    except Exception as e:
        yield updated_history + [
            {"role": "assistant", "content": f"Error: {str(e)}"},
        ]


def build_ui():
    """Construct the Gradio web interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="PDF GPT - Agentic RAG") as demo:
        gr.Markdown(
            "# PDF GPT - Agentic RAG\n"
            "Chat with your PDF documents using AI agents. "
            "Supports OpenAI, Anthropic, Google, Groq, Mistral, Cohere, and Ollama."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("Model Settings", open=True):
                    provider_dropdown = gr.Dropdown(
                        choices=ModelRegistry.get_provider_names(),
                        value="OpenAI",
                        label="Provider",
                    )
                    default_models = ModelRegistry.get_models_for_provider("OpenAI")
                    model_dropdown = gr.Dropdown(
                        choices=default_models,
                        value=default_models[0] if default_models else None,
                        label="Model",
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Enter API key (or set via environment variable)",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.5, value=0.7, step=0.1,
                        label="Temperature",
                    )

                with gr.Accordion("Load Document", open=True):
                    pdf_file = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    gr.Markdown("**-- OR --**")
                    pdf_url = gr.Textbox(
                        label="PDF URL",
                        placeholder="https://example.com/paper.pdf",
                    )
                    load_btn = gr.Button("Load Document", variant="primary")
                    doc_status = gr.Textbox(
                        label="Document Status",
                        interactive=False,
                        value="No document loaded",
                    )

                with gr.Accordion("Agent Settings", open=False):
                    show_reasoning = gr.Checkbox(
                        label="Show agent reasoning steps",
                        value=False,
                    )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=550,
                    placeholder="Load a PDF document, then ask questions about it...",
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about the loaded PDF...",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")

        provider_dropdown.change(
            fn=get_models_for_provider,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        load_btn.click(
            fn=load_document,
            inputs=[pdf_file, pdf_url],
            outputs=[doc_status],
        )

        msg_input.submit(
            fn=chat_with_agent,
            inputs=[
                msg_input, chatbot, provider_dropdown, model_dropdown,
                api_key_input, temperature_slider, show_reasoning,
            ],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        send_btn.click(
            fn=chat_with_agent,
            inputs=[
                msg_input, chatbot, provider_dropdown, model_dropdown,
                api_key_input, temperature_slider, show_reasoning,
            ],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo, theme


if __name__ == "__main__":
    demo, theme = build_ui()
    demo.launch(server_port=7860, server_name="0.0.0.0", theme=theme)
