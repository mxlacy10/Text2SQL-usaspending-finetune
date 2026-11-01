from __future__ import annotations

from t2s.eval.judge import (
    JudgeRecord,
    agreement_stats,
    find_disagreements,
    format_preview,
    parse_verdict,
)


def _rec(ex: bool, judge: bool | None) -> JudgeRecord:
    return JudgeRecord(
        question="q",
        tags=["join"],
        execution_match=ex,
        judge_equivalent=judge,
        predicted_sql="SELECT 1",
        gold_sql="SELECT 1",
    )


def test_parse_verdict() -> None:
    assert parse_verdict("yes") is True
    assert parse_verdict(" No ") is False
    assert parse_verdict(True) is True
    assert parse_verdict("maybe") is None
    assert parse_verdict(None) is None


def test_format_preview_truncates_rows_and_cells() -> None:
    rows = [(i, "x" * 200) for i in range(10)]
    text = format_preview(rows, None, limit=3)
    assert text.startswith("10 rows total; first 3:")
    assert text.count("\n") == 3
    assert "..." in text


def test_format_preview_error() -> None:
    assert format_preview([], "boom", limit=3) == "ERROR: boom"


def test_agreement_and_disagreements() -> None:
    records = [_rec(True, True), _rec(False, False), _rec(True, False), _rec(False, None)]
    stats = agreement_stats(records)
    assert stats["n_judged"] == 3 and stats["agreement"] == 2 / 3
    assert stats["ex_correct"] == 2 and stats["judge_equivalent"] == 1
    diffs = find_disagreements("run", records)
    assert [(d.execution_match, d.judge_equivalent) for d in diffs] == [
        (True, False),
        (False, None),
    ]
    assert all(d.human_label is None for d in diffs)
