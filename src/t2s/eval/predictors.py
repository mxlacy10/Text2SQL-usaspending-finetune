"""Predictor implementations for each AGENTS.md eval baseline:

- base model, zero-shot (mlx-lm, no adapter, minimal system prompt)
- base model, few-shot (k=5) + schema-RAG (mlx-lm, no adapter, schema card +
  few-shot examples in the prompt)
- frontier model, few-shot (Azure AI Foundry, schema card + few-shot examples)
- fine-tuned model, zero-shot (mlx-lm, LoRA adapter, minimal system prompt)
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from t2s.eval.postprocess import clean_sql_output
from t2s.synth.llm import FrontierClient


class Predictor(Protocol):
    """Generates a single SQL prediction for one natural-language question."""

    def predict(self, question: str) -> tuple[str, float]:
        """Returns (predicted_sql, latency_ms)."""
        ...


class MlxZeroShotPredictor:
    """Base or fine-tuned model via mlx-lm, zero-shot (no schema card, no
    few-shot examples) -- matches the system prompt used for training."""

    def __init__(
        self,
        mlx_model: str,
        system_prompt: str,
        max_tokens: int,
        adapter_path: str | None = None,
    ) -> None:
        from mlx_lm import load

        # load() returns a 2- or 3-tuple depending on return_config (default False
        # here, so always 2); mypy can't narrow the Union without return_config
        # being a literal.
        self._model, self._tokenizer = load(mlx_model, adapter_path=adapter_path)  # type: ignore[misc]
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    def _generate(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        from mlx_lm import generate

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        start = time.perf_counter()
        output = generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=self._max_tokens, verbose=False
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return clean_sql_output(output), latency_ms

    def predict(self, question: str) -> tuple[str, float]:
        return self._generate(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": question},
            ]
        )

    def repair(
        self, question: str, previous_sql: str, error: str, template: str
    ) -> tuple[str, float]:
        return self._generate(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": previous_sql},
                {"role": "user", "content": template.format(sql=previous_sql, error=error)},
            ]
        )


class MlxFewShotPredictor:
    """Base model via mlx-lm, with the schema card and few-shot examples
    rendered directly into the prompt (schema-RAG baseline)."""

    def __init__(
        self,
        mlx_model: str,
        prompt_template: str,
        schema_card: str,
        examples_block: str,
        max_tokens: int,
    ) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(mlx_model)  # type: ignore[misc]
        self._prompt_template = prompt_template
        self._schema_card = schema_card
        self._examples_block = examples_block
        self._max_tokens = max_tokens

    def predict(self, question: str) -> tuple[str, float]:
        from mlx_lm import generate

        user_content = self._prompt_template.format(
            schema_card=self._schema_card, examples=self._examples_block, question=question
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        start = time.perf_counter()
        output = generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=self._max_tokens, verbose=False
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return clean_sql_output(output), latency_ms


class FrontierFewShotPredictor:
    """Frontier model (Azure AI Foundry), with the schema card and few-shot
    examples rendered directly into the prompt."""

    def __init__(
        self,
        client: FrontierClient,
        prompt_template: str,
        schema_card: str,
        examples_block: str,
    ) -> None:
        self._client = client
        self._prompt_template = prompt_template
        self._schema_card = schema_card
        self._examples_block = examples_block
        self.last_usage: tuple[int, int] | None = None

    def predict(self, question: str) -> tuple[str, float]:
        prompt = self._prompt_template.format(
            schema_card=self._schema_card, examples=self._examples_block, question=question
        )
        start = time.perf_counter()
        output = self._client.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        self.last_usage = getattr(self._client, "last_usage", None)
        return clean_sql_output(output), latency_ms

    def repair(
        self, question: str, previous_sql: str, error: str, template: str
    ) -> tuple[str, float]:
        prompt = (
            self._prompt_template.format(
                schema_card=self._schema_card, examples=self._examples_block, question=question
            )
            + "\n\n"
            + template.format(sql=previous_sql, error=error)
        )
        start = time.perf_counter()
        output = self._client.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        self.last_usage = getattr(self._client, "last_usage", None)
        return clean_sql_output(output), latency_ms


_RATE_LIMIT_ERRORS = {"TooManyRequests", "ResourceExhausted"}


def extract_genie_sql(message: Any) -> str:
    """Pull the generated SQL out of a completed Genie message.

    Returns "" (an invalid prediction, never a skip) for an errored message or a
    text-only answer with no query attachment."""
    if getattr(message, "error", None) is not None:
        return ""
    for attachment in getattr(message, "attachments", None) or []:
        query = getattr(attachment, "query", None)
        sql = getattr(query, "query", None) if query is not None else None
        if sql:
            return str(sql).strip()
    return ""


class GeniePredictor:
    """Databricks Genie space. Starts a fresh conversation per question; the SQL
    Genie generated is returned and executed by our harness (not Genie's own rows).
    Latency runs from the start call to the completed message, including retries."""

    def __init__(
        self,
        genie_api: Any,
        space_id: str,
        max_retries: int = 5,
        base_backoff_s: float = 2.0,
    ) -> None:
        self._genie = genie_api
        self._space_id = space_id
        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s

    def predict(self, question: str) -> tuple[str, float]:
        start = time.perf_counter()
        sql = ""
        for attempt in range(self._max_retries + 1):
            try:
                message = self._genie.start_conversation_and_wait(self._space_id, question)
            except Exception as e:  # noqa: BLE001 - SDK error types are imported lazily
                if type(e).__name__ in _RATE_LIMIT_ERRORS and attempt < self._max_retries:
                    time.sleep(self._base_backoff_s * 2**attempt)
                    continue
                break
            sql = extract_genie_sql(message)
            break
        return sql, (time.perf_counter() - start) * 1000
