"""Eval harness: runs a Predictor over the frozen test set, executes gold and
predicted SQL against the silver Delta tables, computes per-example and
aggregate metrics (overall + per difficulty tag), and returns an EvalSummary.
"""

from __future__ import annotations

import statistics

from pyspark.sql import SparkSession

from t2s.eval.config import EvalConfig
from t2s.eval.execute import execute_sql
from t2s.eval.metrics import exact_match, rows_match, sql_has_order_by
from t2s.eval.models import EvalSummary, PredictionRecord, TagMetrics
from t2s.eval.predictors import Predictor
from t2s.synth.models import Example


def _tag_metrics(tag: str, records: list[PredictionRecord]) -> TagMetrics:
    n = len(records)
    if n == 0:
        return TagMetrics(
            tag=tag, n=0, execution_accuracy=0.0, exact_match_rate=0.0, valid_sql_rate=0.0
        )
    return TagMetrics(
        tag=tag,
        n=n,
        execution_accuracy=sum(r.execution_match for r in records) / n,
        exact_match_rate=sum(r.exact_match for r in records) / n,
        valid_sql_rate=sum(r.predicted_valid for r in records) / n,
    )


def _estimate_cost_per_1k(config: EvalConfig, records: list[PredictionRecord]) -> float | None:
    """Cost per 1K queries, only if both published pricing is configured and
    real token usage was captured -- never a fabricated placeholder number
    (AGENTS.md rule 1)."""
    if config.frontier_input_price_per_1k_usd is None:
        return None
    if config.frontier_output_price_per_1k_usd is None:
        return None
    usage_records = [
        r for r in records if r.prompt_tokens is not None and r.completion_tokens is not None
    ]
    if not usage_records:
        return None
    avg_prompt = statistics.mean(
        r.prompt_tokens for r in usage_records if r.prompt_tokens is not None
    )
    avg_completion = statistics.mean(
        r.completion_tokens for r in usage_records if r.completion_tokens is not None
    )
    cost_per_query = (
        avg_prompt / 1000 * config.frontier_input_price_per_1k_usd
        + avg_completion / 1000 * config.frontier_output_price_per_1k_usd
    )
    return cost_per_query * 1000


def evaluate_example(
    spark: SparkSession,
    example: Example,
    predictor: Predictor,
) -> PredictionRecord:
    """Predict SQL for one example, execute both gold and predicted SQL, and
    compare them via execution accuracy + exact match."""
    predicted_sql, latency_ms = predictor.predict(example.question)

    gold_result = execute_sql(spark, example.sql)
    pred_result = execute_sql(spark, predicted_sql)

    execution_match = (
        gold_result.valid
        and pred_result.valid
        and rows_match(gold_result.rows, pred_result.rows, ordered=sql_has_order_by(example.sql))
    )

    return PredictionRecord(
        question=example.question,
        tags=example.tags,
        gold_sql=example.sql,
        predicted_sql=predicted_sql,
        gold_valid=gold_result.valid,
        gold_error=gold_result.error,
        predicted_valid=pred_result.valid,
        predicted_error=pred_result.error,
        exact_match=exact_match(example.sql, predicted_sql),
        execution_match=execution_match,
        latency_ms=latency_ms,
    )


def run_eval(
    model_name: str,
    examples: list[Example],
    predictor: Predictor,
    spark: SparkSession,
    config: EvalConfig,
    git_commit: str | None = None,
    dataset_sha256: str | None = None,
    prompt_sha256: str | None = None,
) -> EvalSummary:
    """Run `predictor` over every example, execute + compare, and aggregate."""
    records = [evaluate_example(spark, example, predictor) for example in examples]

    by_tag_records: dict[str, list[PredictionRecord]] = {}
    for record in records:
        primary_tag = record.tags[0] if record.tags else "single_table"
        by_tag_records.setdefault(primary_tag, []).append(record)

    overall = _tag_metrics("overall", records)
    by_tag = [_tag_metrics(tag, tag_records) for tag, tag_records in sorted(by_tag_records.items())]

    latencies = sorted(r.latency_ms for r in records)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0

    return EvalSummary(
        model_name=model_name,
        n_examples=len(records),
        overall=overall,
        by_tag=by_tag,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        cost_per_1k_queries_usd=_estimate_cost_per_1k(config, records),
        git_commit=git_commit,
        dataset_sha256=dataset_sha256,
        prompt_sha256=prompt_sha256,
        records=records,
    )
