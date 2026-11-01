"""DSPy baseline: a Text2SQL program whose few-shot demos are selected by a
DSPy optimizer (BootstrapFewShot) using execution match as the metric.

Compared with the hand-built few-shot baseline, the only things that change
are the prompt framework and how the demos are chosen. The optimizer sees
only the train split (never dev/test), and the metric compares the model's
SQL result with the train gold SQL result.

`dspy` is an optional dependency (extra: `dspy`), imported lazily.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from t2s.eval.postprocess import clean_sql_output
from t2s.synth.models import Example

# Returns True when SQL executes and matches the gold result for the question.
ExecMatch = Callable[[str, str], bool]


def make_signature(instructions: str, static_schema: str | None = None) -> Any:
    """With `static_schema`, the schema card lives once in the instructions instead of
    being an input field. DSPy renders every demo's inputs, so a per-example schema
    field is repeated in each few-shot demo (about 5x the tokens)."""
    import dspy

    if static_schema is not None:
        instructions = f"{instructions}\n\nSchema card:\n\n{static_schema}"

        class CompactTextToSql(dspy.Signature):  # type: ignore[misc]
            question: str = dspy.InputField()
            sql: str = dspy.OutputField(desc="One Spark SQL query, no markdown fences")

        return CompactTextToSql.with_instructions(instructions)

    class TextToSql(dspy.Signature):  # type: ignore[misc]
        schema_card: str = dspy.InputField(desc="Tables and columns available")
        question: str = dspy.InputField()
        sql: str = dspy.OutputField(desc="One Spark SQL query, no markdown fences")

    return TextToSql.with_instructions(instructions)


def build_program(instructions: str, static_schema: str | None = None) -> Any:
    import dspy

    return dspy.Predict(make_signature(instructions, static_schema))


def make_lm(model: str, endpoint: str) -> Any:
    """Azure AI Foundry deployment via DefaultAzureCredential (no static key).
    The token is fetched once, so a run must finish within its ~1h lifetime."""
    import dspy
    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token("https://ai.azure.com/.default").token
    # cache=False: honest latency, no replayed completions. Temperature is left at the
    # deployment default because gpt-5-chat rejects temperature=0.
    return dspy.LM(f"openai/{model}", api_base=endpoint, api_key=token, cache=False)


def optimize(
    program: Any,
    train: list[Example],
    schema_card: str,
    exec_match: ExecMatch,
    max_demos: int,
    static_schema: bool = False,
) -> Any:
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    input_keys = ("question",) if static_schema else ("schema_card", "question")
    trainset = [
        dspy.Example(schema_card=schema_card, question=e.question, gold_sql=e.sql).with_inputs(
            *input_keys
        )
        for e in train
    ]

    def metric(example: Any, pred: Any, trace: Any = None) -> bool:
        return exec_match(clean_sql_output(pred.sql or ""), example.gold_sql)

    optimizer = BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=max_demos, max_labeled_demos=0
    )
    return optimizer.compile(program, trainset=trainset)


class DspyPredictor:
    """Predictor adapter so the DSPy program runs through the shared eval harness."""

    def __init__(
        self, program: Any, lm: Any, schema_card: str, static_schema: bool = False
    ) -> None:
        self._program = program
        self._lm = lm
        self._schema_card = schema_card
        self._static_schema = static_schema
        self.last_usage: tuple[int, int] | None = None

    def predict(self, question: str) -> tuple[str, float]:
        import dspy

        start = time.perf_counter()
        with dspy.context(lm=self._lm):
            if self._static_schema:
                pred = self._program(question=question)
            else:
                pred = self._program(schema_card=self._schema_card, question=question)
        latency_ms = (time.perf_counter() - start) * 1000
        usage = (self._lm.history[-1].get("usage") or {}) if self._lm.history else {}
        self.last_usage = (
            (int(usage["prompt_tokens"]), int(usage["completion_tokens"])) if usage else None
        )
        return clean_sql_output(pred.sql or ""), latency_ms


def save_program(program: Any, path: Path) -> None:
    program.save(str(path))


def load_program(instructions: str, path: Path, static_schema: str | None = None) -> Any:
    program = build_program(instructions, static_schema)
    program.load(str(path))
    return program
