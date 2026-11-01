from __future__ import annotations

import json
from pathlib import Path

from t2s.eval.report_tables import build_results_block, load_runs


def _run(
    model: str, backend: str | None, matches: list[bool], **extra: object
) -> dict[str, object]:
    n = len(matches)
    return {
        "model_name": model,
        "execution_backend": backend,
        "n_examples": n,
        "overall": {
            "tag": "overall",
            "n": n,
            "execution_accuracy": sum(matches) / n,
            "exact_match_rate": 0.0,
            "valid_sql_rate": 1.0,
        },
        "by_tag": [
            {
                "tag": "join",
                "n": n,
                "execution_accuracy": sum(matches) / n,
                "exact_match_rate": 0.0,
                "valid_sql_rate": 1.0,
            }
        ],
        "latency_p50_ms": 1000.0,
        "latency_p95_ms": 2000.0,
        "git_commit": "abcdef1234",
        "records": [{"execution_match": m} for m in matches],
        **extra,
    }


def _write(results: Path, name: str, data: object) -> None:
    (results / name).write_text(json.dumps(data))


def test_runs_without_backend_are_local_and_non_runs_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _run("finetuned", None, [True, False]))
    _write(tmp_path, "judge_calibration.json", {"human_signoff": None})
    assert list(load_runs(tmp_path)) == [("finetuned", "local_spark")]


def test_block_shows_counts_and_judge_not_yet_run_without_signoff(tmp_path: Path) -> None:
    _write(tmp_path, "f.json", _run("finetuned", "databricks_uc", [True, False, True, False]))
    _write(tmp_path, "g.json", _run("genie", "databricks_uc", [True, True, True, False]))
    block = build_results_block(tmp_path)
    assert "2/4 (50.0%)" in block and "3/4 (75.0%)" in block
    assert "not yet run" in block
    assert "Fine-tune vs Genie vs frontier" in block
    assert "not statistically significant" in block


def test_judge_column_appears_only_after_human_signoff(tmp_path: Path) -> None:
    _write(tmp_path, "f.json", _run("finetuned", "databricks_uc", [True, False]))
    _write(tmp_path, "g.json", _run("genie", "databricks_uc", [True, True]))
    judge = {
        "judge_model": "m",
        "judge_date": "2026-11-01",
        "human_signoff": "me",
        "disagreements": [{}],
        "runs": {
            "finetuned": {"stats": {"judge_equivalent": 2}},
            "genie": {"stats": {"judge_equivalent": 2}},
        },
    }
    _write(tmp_path, "judge_calibration.json", judge)
    block = build_results_block(tmp_path)
    assert "2/2 (100.0%)" in block and "not yet run" not in block
    assert "never replaces execution accuracy" in block


def test_empty_results() -> None:
    assert build_results_block(Path("/nonexistent-dir-for-test")) == "_Not yet run._"
