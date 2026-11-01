"""Build the README results block from results/*.json only (never hand-edited).

Pure functions: scripts/report.py just writes the returned markdown into README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ORDER = [
    "base_zero_shot",
    "base_few_shot",
    "frontier",
    "dspy_frontier",
    "dspy_frontier_compact",
    "finetuned",
    "finetuned_selfcorrect",
    "genie",
]
LABELS = {
    "base_zero_shot": "Base model, zero-shot",
    "base_few_shot": "Base model, few-shot (schema-RAG)",
    "frontier": "Frontier model, few-shot",
    "dspy_frontier": "Frontier model via DSPy (BootstrapFewShot)",
    "dspy_frontier_compact": "Frontier model via DSPy, schema once (compact)",
    "finetuned": "Fine-tuned (QLoRA), zero-shot",
    "finetuned_selfcorrect": "Fine-tuned + execute-and-repair (<=2 retries)",
    "genie": "Databricks Genie space",
}
COMPARE = ["finetuned", "genie", "frontier"]
UC = "databricks_uc"
LOCAL = "local_spark"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _count(correct: int, n: int) -> str:
    return f"{correct}/{n} ({_pct(correct / n)})" if n else "-"


def _label(name: str) -> str:
    return LABELS.get(name, name)


def load_runs(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Eval-run files keyed by (model_name, execution_backend). Files without a model_name
    (e.g. judge_calibration.json) are skipped. Results written before the execution_backend
    field existed were all local runs."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "model_name" not in data:
            continue
        runs[(data["model_name"], data.get("execution_backend") or LOCAL)] = data
    return runs


def load_judge(results_dir: Path) -> dict[str, Any] | None:
    """Judge calibration, only if a human has signed off (otherwise it is not reported)."""
    path = results_dir / "judge_calibration.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data if data.get("human_signoff") else None


def _ex_correct(run: dict[str, Any]) -> int:
    return sum(1 for r in run["records"] if r["execution_match"])


def _judge_cell(judge: dict[str, Any] | None, model: str, backend: str, n: int) -> str:
    if judge is None or backend != UC or model not in judge["runs"]:
        return "not yet run"
    return _count(int(judge["runs"][model]["stats"]["judge_equivalent"]), n)


def _ordered(runs: dict[tuple[str, str], dict[str, Any]], backend: str) -> list[str]:
    names = {m for (m, b) in runs if b == backend}
    return [n for n in ORDER if n in names] + sorted(names - set(ORDER))


def _main_table(
    runs: dict[tuple[str, str], dict[str, Any]], judge: dict[str, Any] | None, backend: str
) -> list[str]:
    lines = [
        "| Run | Execution acc. (primary) | Judge-equivalent (secondary) | Exact match "
        "| Valid SQL | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in _ordered(runs, backend):
        r = runs[(name, backend)]
        o = r["overall"]
        n = r["n_examples"]
        lines.append(
            f"| {_label(name)} | {_count(_ex_correct(r), n)} "
            f"| {_judge_cell(judge, name, backend, n)} | {_pct(o['exact_match_rate'])} "
            f"| {_pct(o['valid_sql_rate'])} | {r['latency_p50_ms'] / 1000:.1f}s "
            f"| {r['latency_p95_ms'] / 1000:.1f}s |"
        )
    return lines


def _tag_table(runs: dict[tuple[str, str], dict[str, Any]], backend: str) -> list[str]:
    names = _ordered(runs, backend)
    tags = sorted({t["tag"] for n in names for t in runs[(n, backend)]["by_tag"]})
    lines = [
        "| Tag | " + " | ".join(_label(n) for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for tag in tags:
        cells = []
        for n in names:
            match = next((t for t in runs[(n, backend)]["by_tag"] if t["tag"] == tag), None)
            cells.append(f"{_pct(match['execution_accuracy'])} (n={match['n']})" if match else "-")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    return lines


def _comparison(
    runs: dict[tuple[str, str], dict[str, Any]], judge: dict[str, Any] | None
) -> list[str]:
    present = [m for m in COMPARE if (m, UC) in runs]
    if len(present) < 2:
        return ["_Not yet run._"]
    n = runs[(present[0], UC)]["n_examples"]
    lines = [
        "| Run | Correct by execution accuracy | Judge-equivalent |",
        "|---|---|---|",
    ]
    for m in present:
        lines.append(
            f"| {_label(m)} | {_count(_ex_correct(runs[(m, UC)]), n)} "
            f"| {_judge_cell(judge, m, UC, n)} |"
        )
    counts = {m: _ex_correct(runs[(m, UC)]) for m in present}
    spread = max(counts.values()) - min(counts.values())
    lines += [
        "",
        f"With n = {n}, one question is {100 / n:.1f} points. The gap between the highest and "
        f"lowest of these runs is {spread} questions by execution accuracy; a 2-question "
        "difference between two runs is not statistically significant.",
    ]
    genie = runs.get(("genie", UC))
    tables = (genie or {}).get("run_metadata", {}) or {}
    if tables.get("genie_tables"):
        lines += [
            "",
            f"Genie's space was configured over: {tables['genie_tables']}; "
            "the space had no custom instructions.",
        ]
    if judge is not None:
        lines += [
            "",
            f"Judge: `{judge['judge_model']}` (run {judge['judge_date']}), a secondary metric that "
            "never replaces execution accuracy. A human reviewed the "
            f"{len(judge['disagreements'])} records where the judge and execution accuracy "
            "disagreed and agreed with the judge on all of them (execution accuracy was a false "
            "negative: extra or missing columns, aliases, `DISTINCT`, redundant filters). "
            "Records where both agreed were not hand-checked.",
        ]
    return lines


def build_results_block(results_dir: Path) -> str:
    runs = load_runs(results_dir)
    if not runs:
        return "_Not yet run._"
    judge = load_judge(results_dir)
    n_examples = sorted({r["n_examples"] for r in runs.values()})

    lines = [
        f"Frozen test set, n = {', '.join(str(n) for n in n_examples)} questions. "
        "Each run records its `execution_backend`: `databricks_uc` (Unity Catalog on a Databricks "
        "Free Edition workspace) or `local_spark` (local Spark over the same silver tables).",
        "",
        "### Databricks Unity Catalog runs",
        "",
        *_main_table(runs, judge, UC),
        "",
        "Execution accuracy by difficulty tag:",
        "",
        *_tag_table(runs, UC),
        "",
        "### Fine-tune vs Genie vs frontier",
        "",
        *_comparison(runs, judge),
    ]
    if any(b == LOCAL for (_, b) in runs):
        lines += [
            "",
            "### Earlier local Spark runs",
            "",
            "Kept for parity checks. `base_zero_shot` here was scored before the SQL "
            "post-processing fix (markdown-fence stripping), so its 0% is a formatting artifact.",
            "",
            *_main_table(runs, None, LOCAL),
        ]
    commits = sorted({str(r["git_commit"])[:7] for r in runs.values() if r.get("git_commit")})
    lines += [
        "",
        f"Generated from `results/*.json` (git commits: {', '.join(commits)}). "
        "Cost per 1K queries: not yet measured.",
    ]
    return "\n".join(lines)
