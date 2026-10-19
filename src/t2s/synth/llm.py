"""Minimal frontier-LLM client wrapper for synthetic data generation.

API keys come from environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY),
never from config files or code (AGENTS.md rule 3: no secrets in git).
"""

from __future__ import annotations

import os
from typing import Protocol


class FrontierClient(Protocol):
    """A chat-completion-style client that returns raw text for a prompt."""

    def generate(self, prompt: str) -> str: ...


class OpenAIClient:
    """Wraps the OpenAI chat completions API."""

    def __init__(self, model: str) -> None:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""


class AnthropicClient:
    """Wraps the Anthropic messages API."""

    def __init__(self, model: str) -> None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        return block.text if hasattr(block, "text") else ""


def get_frontier_client(model: str) -> FrontierClient:
    """Dispatch to OpenAI or Anthropic based on the model ID prefix."""
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return OpenAIClient(model)
    if model.startswith("claude-"):
        return AnthropicClient(model)
    raise ValueError(f"Don't know which provider serves model {model!r}.")
