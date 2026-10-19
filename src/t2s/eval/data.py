"""Load data/splits/*.jsonl files as Example records for the eval harness."""

from __future__ import annotations

from pathlib import Path

from t2s.synth.models import Example


def load_examples(path: str | Path) -> list[Example]:
    """Read one data/splits/<split>.jsonl file into a list of Example records."""
    examples: list[Example] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(Example.model_validate_json(line))
    return examples
