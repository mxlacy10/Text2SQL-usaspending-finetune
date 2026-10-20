"""Regenerate the README results table from results/*.json (never edit it by hand)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
ORDER = ["base_zero_shot", "base_few_shot", "frontier", "finetuned"]
LABELS = {
    "base_zero_shot": "Base model, zero-shot",
    "base_few_shot": "Base model, few-shot (schema-RAG)",
    "frontier": "Frontier model, few-shot",
    "finetuned": "Fine-tuned (QLoRA), zero-shot",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_table(results_dir: Path) -> str:
    runs = {}
    for path in results_dir.glob("*.json"):
        data = json.loads(path.read_text())
        runs[data["model_name"]] = data
    if not runs:
        return "_Not yet run._"

    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
    n_examples = {runs[n]["n_examples"] for n in names}
    lines = [
        f"Frozen test set, n = {', '.join(str(n) for n in sorted(n_examples))} questions.",
        "",
        "| Run | Execution acc. | Exact match | Valid SQL | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        r = runs[name]
        o = r["overall"]
        lines.append(
            f"| {LABELS.get(name, name)} | {_pct(o['execution_accuracy'])} "
            f"| {_pct(o['exact_match_rate'])} | {_pct(o['valid_sql_rate'])} "
            f"| {r['latency_p50_ms'] / 1000:.1f}s | {r['latency_p95_ms'] / 1000:.1f}s |"
        )

    tags = sorted({t["tag"] for n in names for t in runs[n]["by_tag"]})
    lines += [
        "",
        "Execution accuracy by difficulty tag:",
        "",
        "| Tag | " + " | ".join(LABELS.get(n, n) for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for tag in tags:
        cells = []
        for n in names:
            match = next((t for t in runs[n]["by_tag"] if t["tag"] == tag), None)
            cells.append(f"{_pct(match['execution_accuracy'])} (n={match['n']})" if match else "-")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    commits = sorted({runs[n]["git_commit"][:7] for n in names})
    lines += [
        "",
        f"Generated from `results/*.json` (git commits: {', '.join(commits)}). "
        "Cost per 1K queries: not yet measured.",
    ]
    return "\n".join(lines)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = f"{START}\n{build_table(ROOT / 'results')}\n{END}"
    pattern = re.compile(f"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing {START} / {END} markers")
    readme.write_text(pattern.sub(lambda _: block, text))


if __name__ == "__main__":
    main()
