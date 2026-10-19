"""Minimal frontier-LLM client wrapper for synthetic data generation.

API keys/credentials come from the environment or the local Azure CLI
login (never from config files or code — AGENTS.md rule 3: no secrets in git).
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


class AzureFoundryClient:
    """Wraps an Azure AI Foundry deployment via the OpenAI Responses API.

    Authenticates with Azure AD (DefaultAzureCredential) rather than a static
    API key: this picks up `az login`, a managed identity, or service-principal
    env vars (AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID), per the
    user's existing Azure setup. No secret key is ever read or stored here.
    """

    def __init__(self, model: str, endpoint: str) -> None:
        import openai
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        self._client = openai.OpenAI(base_url=endpoint, api_key=token_provider)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(model=self._model, input=prompt)
        return response.output_text or ""


def get_frontier_client(model: str, azure_endpoint: str | None = None) -> FrontierClient:
    """Dispatch to Azure Foundry, OpenAI, or Anthropic for the given model.

    `azure_endpoint` takes priority when set (from config.models.frontier_azure_endpoint)
    since Azure-deployed model names (e.g. "gpt-5-chat") can collide with public
    OpenAI model-name prefixes.
    """
    if azure_endpoint:
        return AzureFoundryClient(model, azure_endpoint)
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return OpenAIClient(model)
    if model.startswith("claude-"):
        return AnthropicClient(model)
    raise ValueError(f"Don't know which provider serves model {model!r}.")
