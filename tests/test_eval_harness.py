from __future__ import annotations

from typing import Any

import pytest

from t2s.eval import harness as harness_module
from t2s.eval.config import EvalConfig
from t2s.eval.execute import ExecutionResult
from t2s.synth.models import Example


class FakePredictor:
    """Returns a fixed SQL/latency pair per question, or a per-question map."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def predict(self, question: str) -> tuple[str, float]:
        return self._answers[question], 10.0


def make_config() -> EvalConfig:
    return EvalConfig(
        seed=42,
        test_split="data/splits/test.jsonl",
        train_split="data/splits/train.jsonl",
        schema_card="data/schema_card.md",
        results_dir="results",
        k_shot=5,
        max_tokens=256,
        zero_shot_prompt="t2s_sql_system_v1.md",
        few_shot_prompt="t2s_sql_fewshot_v1.md",
    )


@pytest.fixture(autouse=True)
def fake_execute(monkeypatch: pytest.MonkeyPatch) -> dict[str, ExecutionResult]:
    """Stub out real Spark execution with a lookup table keyed by SQL text,
    so the harness can be tested without a SparkSession."""
    table: dict[str, ExecutionResult] = {}

    def _execute(spark: Any, sql: str) -> ExecutionResult:
        return table.get(sql, ExecutionResult(valid=False, rows=[], error="not registered"))

    monkeypatch.setattr(harness_module, "execute_sql", _execute)
    return table


def test_run_eval_execution_match(fake_execute: dict[str, ExecutionResult]) -> None:
    fake_execute["SELECT a FROM t"] = ExecutionResult(valid=True, rows=[(1,), (2,)])
    fake_execute["SELECT a FROM t2"] = ExecutionResult(valid=True, rows=[(2,), (1,)])

    examples = [Example(question="q1", sql="SELECT a FROM t", tags=["single_table"], split="test")]
    predictor = FakePredictor({"q1": "SELECT a FROM t2"})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    assert summary.n_examples == 1
    assert summary.overall.execution_accuracy == 1.0
    assert summary.overall.valid_sql_rate == 1.0
    assert summary.records[0].execution_match is True


def test_run_eval_execution_mismatch_on_wrong_rows(
    fake_execute: dict[str, ExecutionResult],
) -> None:
    fake_execute["SELECT a FROM t"] = ExecutionResult(valid=True, rows=[(1,)])
    fake_execute["SELECT b FROM t"] = ExecutionResult(valid=True, rows=[(2,)])

    examples = [Example(question="q1", sql="SELECT a FROM t", tags=["single_table"], split="test")]
    predictor = FakePredictor({"q1": "SELECT b FROM t"})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    assert summary.overall.execution_accuracy == 0.0
    assert summary.records[0].execution_match is False


def test_run_eval_invalid_predicted_sql_never_matches(
    fake_execute: dict[str, ExecutionResult],
) -> None:
    fake_execute["SELECT a FROM t"] = ExecutionResult(valid=True, rows=[(1,)])

    examples = [Example(question="q1", sql="SELECT a FROM t", tags=["single_table"], split="test")]
    predictor = FakePredictor({"q1": "NOT VALID SQL AT ALL ((("})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    assert summary.overall.execution_accuracy == 0.0
    assert summary.overall.valid_sql_rate == 0.0


def test_run_eval_aggregates_by_tag(fake_execute: dict[str, ExecutionResult]) -> None:
    fake_execute["SELECT a"] = ExecutionResult(valid=True, rows=[(1,)])
    fake_execute["SELECT b"] = ExecutionResult(valid=True, rows=[(2,)])

    examples = [
        Example(question="q1", sql="SELECT a", tags=["single_table"], split="test"),
        Example(question="q2", sql="SELECT b", tags=["join"], split="test"),
    ]
    predictor = FakePredictor({"q1": "SELECT a", "q2": "SELECT a"})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    by_tag = {t.tag: t for t in summary.by_tag}
    assert by_tag["single_table"].execution_accuracy == 1.0
    assert by_tag["join"].execution_accuracy == 0.0


def test_run_eval_latency_percentiles_reported(fake_execute: dict[str, ExecutionResult]) -> None:
    fake_execute["SELECT a"] = ExecutionResult(valid=True, rows=[(1,)])
    examples = [Example(question="q1", sql="SELECT a", tags=["single_table"], split="test")]
    predictor = FakePredictor({"q1": "SELECT a"})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    assert summary.latency_p50_ms == 10.0
    assert summary.latency_p95_ms == 10.0


def test_run_eval_cost_is_none_when_pricing_unconfigured(
    fake_execute: dict[str, ExecutionResult],
) -> None:
    fake_execute["SELECT a"] = ExecutionResult(valid=True, rows=[(1,)])
    examples = [Example(question="q1", sql="SELECT a", tags=["single_table"], split="test")]
    predictor = FakePredictor({"q1": "SELECT a"})

    summary = harness_module.run_eval(
        model_name="fake", examples=examples, predictor=predictor, spark=None, config=make_config()
    )

    assert summary.cost_per_1k_queries_usd is None
