"""Configuration constants and model registry for pdfGPT Agentic RAG."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    requires_api_key: bool = True
    api_key_env_var: str = ""


class ModelRegistry:
    """Central registry of supported LLM models grouped by provider."""

    PROVIDERS: ClassVar[dict] = {
        "OpenAI": {
            "api_key_env": "OPENAI_API_KEY",
            "models": [
                ModelInfo("OpenAI", "gpt-4o", "GPT-4o", True, "OPENAI_API_KEY"),
                ModelInfo("OpenAI", "gpt-4o-mini", "GPT-4o Mini", True, "OPENAI_API_KEY"),
                ModelInfo("OpenAI", "gpt-4-turbo", "GPT-4 Turbo", True, "OPENAI_API_KEY"),
                ModelInfo("OpenAI", "gpt-3.5-turbo", "GPT-3.5 Turbo", True, "OPENAI_API_KEY"),
            ],
        },
        "Anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "models": [
                ModelInfo("Anthropic", "claude-sonnet-4-20250514", "Claude Sonnet 4", True, "ANTHROPIC_API_KEY"),
                ModelInfo("Anthropic", "claude-3-5-haiku-20241022", "Claude 3.5 Haiku", True, "ANTHROPIC_API_KEY"),
                ModelInfo("Anthropic", "claude-3-opus-20240229", "Claude 3 Opus", True, "ANTHROPIC_API_KEY"),
            ],
        },
        "Google": {
            "api_key_env": "GEMINI_API_KEY",
            "models": [
                ModelInfo("Google", "gemini/gemini-1.5-pro", "Gemini 1.5 Pro", True, "GEMINI_API_KEY"),
                ModelInfo("Google", "gemini/gemini-1.5-flash", "Gemini 1.5 Flash", True, "GEMINI_API_KEY"),
                ModelInfo("Google", "gemini/gemini-2.0-flash", "Gemini 2.0 Flash", True, "GEMINI_API_KEY"),
            ],
        },
        "Groq (Free Tier)": {
            "api_key_env": "GROQ_API_KEY",
            "models": [
                ModelInfo("Groq (Free Tier)", "groq/llama-3.1-70b-versatile", "Llama 3.1 70B", True, "GROQ_API_KEY"),
                ModelInfo("Groq (Free Tier)", "groq/llama-3.1-8b-instant", "Llama 3.1 8B", True, "GROQ_API_KEY"),
                ModelInfo("Groq (Free Tier)", "groq/mixtral-8x7b-32768", "Mixtral 8x7B", True, "GROQ_API_KEY"),
            ],
        },
        "Mistral": {
            "api_key_env": "MISTRAL_API_KEY",
            "models": [
                ModelInfo("Mistral", "mistral/mistral-large-latest", "Mistral Large", True, "MISTRAL_API_KEY"),
                ModelInfo("Mistral", "mistral/mistral-small-latest", "Mistral Small", True, "MISTRAL_API_KEY"),
            ],
        },
        "Cohere": {
            "api_key_env": "COHERE_API_KEY",
            "models": [
                ModelInfo("Cohere", "command-r-plus", "Command R+", True, "COHERE_API_KEY"),
                ModelInfo("Cohere", "command-r", "Command R", True, "COHERE_API_KEY"),
            ],
        },
        "Ollama (Local - Free)": {
            "api_key_env": "",
            "models": [
                ModelInfo("Ollama (Local - Free)", "ollama/llama3", "Llama 3", False),
                ModelInfo("Ollama (Local - Free)", "ollama/mistral", "Mistral 7B", False),
                ModelInfo("Ollama (Local - Free)", "ollama/phi3", "Phi-3", False),
                ModelInfo("Ollama (Local - Free)", "ollama/gemma2", "Gemma 2", False),
            ],
        },
    }

    @classmethod
    def get_provider_names(cls):
        return list(cls.PROVIDERS.keys())

    @classmethod
    def get_models_for_provider(cls, provider_name):
        provider = cls.PROVIDERS.get(provider_name, {})
        return [m.display_name for m in provider.get("models", [])]

    @classmethod
    def get_model_id(cls, provider_name, display_name):
        provider = cls.PROVIDERS.get(provider_name, {})
        for model in provider.get("models", []):
            if model.display_name == display_name:
                return model.model_id
        return None

    @classmethod
    def get_model_info(cls, provider_name, display_name):
        provider = cls.PROVIDERS.get(provider_name, {})
        for model in provider.get("models", []):
            if model.display_name == display_name:
                return model
        return None

    @classmethod
    def get_api_key_env(cls, provider_name):
        return cls.PROVIDERS.get(provider_name, {}).get("api_key_env", "")


@dataclass
class AppConfig:
    """Application-wide configuration."""
    default_chunk_size: int = 200
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    max_search_results: int = 5
    agent_max_iterations: int = 5
    default_temperature: float = 0.7
    api_port: int = 8080
    ui_port: int = 7860
