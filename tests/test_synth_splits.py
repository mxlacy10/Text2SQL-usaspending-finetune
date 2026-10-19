import json

import pytest

from t2s.synth.models import ExecutedExample
from t2s.synth.splits import stratified_split, write_splits


def make_examples(tag: str, count: int) -> list[ExecutedExample]:
    return [
        ExecutedExample(
            question=f"question {tag} {i}",
            sql=f"SELECT {i}",
            tags=[tag],
            valid=True,
            row_count=1,
        )
        for i in range(count)
    ]


def test_stratified_split_proportions_per_tag() -> None:
    examples = make_examples("single_table", 10) + make_examples("join", 10)
    splits = stratified_split(examples, seed=42)
    assert len(splits["train"]) + len(splits["dev"]) + len(splits["test"]) == 20
    # Each tag stratum of 10 should produce 8 train / 1 dev / 1 test.
    train_tags = [e.tags[0] for e in splits["train"]]
    assert train_tags.count("single_table") == 8
    assert train_tags.count("join") == 8


def test_stratified_split_is_deterministic_given_seed() -> None:
    examples = make_examples("single_table", 10)
    splits_a = stratified_split(examples, seed=42)
    splits_b = stratified_split(examples, seed=42)
    assert [e.question for e in splits_a["train"]] == [e.question for e in splits_b["train"]]


def test_write_splits_refuses_to_overwrite_frozen_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    examples = make_examples("single_table", 10)
    splits = stratified_split(examples, seed=42)
    write_splits(splits, tmp_path)

    with pytest.raises(RuntimeError, match="frozen"):
        write_splits(splits, tmp_path)

    # force_test=True is allowed to overwrite.
    write_splits(splits, tmp_path, force_test=True)


def test_write_splits_writes_valid_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    examples = make_examples("single_table", 10)
    splits = stratified_split(examples, seed=42)
    paths = write_splits(splits, tmp_path)

    train_lines = paths["train"].read_text().splitlines()
    assert len(train_lines) == 8
    record = json.loads(train_lines[0])
    assert record["split"] == "train"
    assert record["tags"] == ["single_table"]
