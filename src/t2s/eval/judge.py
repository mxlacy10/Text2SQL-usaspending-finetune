"""LLM-judge helpers: a secondary SQL-equivalence metric reported next to execution
accuracy (EX stays primary and is never replaced by the judge).

Pure functions only; the judge call and Spark previews live in scripts/judge.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from t2s.eval.metrics import Row


class JudgeRecord(BaseModel):
    question: str
    tags: list[str]
    execution_match: bool
    judge_equivalent: bool | None  # None: the judge returned no usable verdict
    judge_rationale: str | None = None
    predicted_sql: str
    gold_sql: str


class Disagreement(JudgeRecord):
    run: str
    # Filled in by the human during calibration; never by an agent.
    human_label: bool | None = None
    human_note: str | None = None


def format_preview(rows: list[Row], error: str | None, limit: int, max_cell: int = 80) -> str:
    """Compact text preview of a result set: row count plus the first `limit` rows."""
    if error is not None:
        return f"ERROR: {error[:300]}"

    def cell(value: Any) -> str:
        text = repr(value)
        return text if len(text) <= max_cell else text[: max_cell - 3] + "..."

    shown = "\n".join("(" + ", ".join(cell(v) for v in row) + ")" for row in rows[:limit])
    return f"{len(rows)} rows total; first {min(limit, len(rows))}:\n{shown}"


def parse_verdict(value: Any) -> bool | None:
    """Map a judge feedback value to a bool; anything that isn't a clear yes/no is None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "true"):
            return True
        if v in ("no", "false"):
            return False
    return None


def agreement_stats(records: list[JudgeRecord]) -> dict[str, float | int]:
    """Judge-vs-EX agreement over records that have a usable judge verdict."""
    judged = [r for r in records if r.judge_equivalent is not None]
    n = len(judged)
    agree = sum(r.judge_equivalent == r.execution_match for r in judged)
    return {
        "n": len(records),
        "n_judged": n,
        "ex_correct": sum(r.execution_match for r in records),
        "judge_equivalent": sum(bool(r.judge_equivalent) for r in judged),
        "agreement": agree / n if n else 0.0,
        "disagreements": n - agree,
    }


def find_disagreements(run: str, records: list[JudgeRecord]) -> list[Disagreement]:
    """Records where the judge and EX differ, or the judge gave no verdict (needs a human)."""
    return [
        Disagreement(run=run, **r.model_dump())
        for r in records
        if r.judge_equivalent is None or r.judge_equivalent != r.execution_match
    ]
