"""
LLM layer.

Design: BaseLLM is the contract. Swapping Claude for another provider
(or a locally hosted model for cost reasons) is a new class here --
the RAG pipeline (main.py) depends only on `generate(prompt) -> str`.
"""

from abc import ABC, abstractmethod
import os


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class ClaudeLLM(BaseLLM):
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        from anthropic import Anthropic
        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLM(BaseLLM):
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
