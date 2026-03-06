"""LLM provider abstraction using LiteLLM for multi-provider support."""

from typing import List, Optional, Generator

from litellm import completion


class LLMProvider:
    """Unified interface for calling any LLM via LiteLLM.

    Supports OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Ollama,
    and any other provider supported by LiteLLM.
    """

    def __init__(self, model_id: str, api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 1024):
        self._model_id = model_id
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def _build_kwargs(self) -> dict:
        kwargs = {
            "model": self._model_id,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    def generate(self, messages: List[dict]) -> str:
        """Generate a response from the LLM."""
        kwargs = self._build_kwargs()
        kwargs["messages"] = messages
        try:
            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM Error] {str(e)}"

    def generate_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        """Stream a response token by token from the LLM."""
        kwargs = self._build_kwargs()
        kwargs["messages"] = messages
        kwargs["stream"] = True
        try:
            response = completion(**kwargs)
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
        except Exception as e:
            yield f"[LLM Error] {str(e)}"

    def chat(self, system_prompt: str, user_message: str,
             history: Optional[List[dict]] = None) -> str:
        """Convenience method for a single-turn or multi-turn chat."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.generate(messages)
