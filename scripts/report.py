"""Regenerate the README results block from results/*.json (never edit it by hand)."""

from __future__ import annotations

import re
from pathlib import Path

from t2s.eval.report_tables import build_results_block

ROOT = Path(__file__).resolve().parent.parent
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = f"{START}\n{build_results_block(ROOT / 'results')}\n{END}"
    pattern = re.compile(f"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README.md is missing {START} / {END} markers")
    readme.write_text(pattern.sub(lambda _: block, text))


if __name__ == "__main__":
    main()
