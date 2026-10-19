from __future__ import annotations

from pathlib import Path

from t2s.eval.data import load_examples
from t2s.synth.models import Example


def test_load_examples_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    examples = [
        Example(question="q1", sql="SELECT 1", tags=["single_table"], split="test"),
        Example(question="q2", sql="SELECT 2", tags=["join"], split="test"),
    ]
    with path.open("w") as f:
        for e in examples:
            f.write(e.model_dump_json() + "\n")

    loaded = load_examples(path)
    assert loaded == examples


def test_load_examples_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    example = Example(question="q1", sql="SELECT 1", tags=["single_table"], split="test")
    path.write_text(f"\n{example.model_dump_json()}\n\n")

    loaded = load_examples(path)
    assert loaded == [example]
