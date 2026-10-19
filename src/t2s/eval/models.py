"""Pydantic models for eval harness inputs/outputs.

See AGENTS.md eval section: per-example prediction records, aggregate
EX/exact-match/valid-SQL metrics (overall and broken down by difficulty tag),
and latency/cost reporting.
"""

from __future__ import annotations

from pydantic import BaseModel

from t2s.synth.models import DifficultyTag


class PredictionRecord(BaseModel):
    """One test example's gold vs. predicted SQL, with comparison outcomes."""

    question: str
    tags: list[DifficultyTag]
    gold_sql: str
    predicted_sql: str
    gold_valid: bool
    gold_error: str | None = None
    predicted_valid: bool
    predicted_error: str | None = None
    exact_match: bool
    execution_match: bool
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class TagMetrics(BaseModel):
    """Aggregate metrics for one difficulty tag (or "overall")."""

    tag: str
    n: int
    execution_accuracy: float
    exact_match_rate: float
    valid_sql_rate: float


class EvalSummary(BaseModel):
    """Full eval run summary: overall + per-tag metrics, latency, cost."""

    model_name: str
    n_examples: int
    overall: TagMetrics
    by_tag: list[TagMetrics]
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_queries_usd: float | None = None
    git_commit: str | None = None
    dataset_sha256: str | None = None
    prompt_sha256: str | None = None
    records: list[PredictionRecord]
