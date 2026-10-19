"""Stratified train/dev/test splits by difficulty tag.

See AGENTS.md: "Split 80/10/10 stratified by difficulty tag" and
"The test set is frozen... Any change to the test set must be a deliberate,
documented commit."
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from t2s.synth.models import Example, ExecutedExample, Split

DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_DEV_FRACTION = 0.1


def stratified_split(
    examples: list[ExecutedExample],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    seed: int = 42,
) -> dict[Split, list[Example]]:
    """Split examples 80/10/10 (by default), stratified by each example's
    first (primary) difficulty tag, so every tag is proportionally represented
    in every split."""
    rng = random.Random(seed)
    buckets: dict[str, list[ExecutedExample]] = defaultdict(list)
    for example in examples:
        primary_tag = example.tags[0] if example.tags else "single_table"
        buckets[primary_tag].append(example)

    splits: dict[Split, list[Example]] = {"train": [], "dev": [], "test": []}
    for items in buckets.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_fraction)
        n_dev = round(n * dev_fraction)
        grouped: list[tuple[Split, list[ExecutedExample]]] = [
            ("train", shuffled[:n_train]),
            ("dev", shuffled[n_train : n_train + n_dev]),
            ("test", shuffled[n_train + n_dev :]),
        ]
        for split_name, split_items in grouped:
            splits[split_name].extend(
                Example(question=e.question, sql=e.sql, tags=e.tags, split=split_name)
                for e in split_items
            )
    return splits


def write_splits(
    splits: dict[Split, list[Example]], out_dir: Path, force_test: bool = False
) -> dict[Split, Path]:
    """Write each split to <out_dir>/<split>.jsonl.

    Refuses to overwrite an existing test.jsonl unless force_test=True, since
    the test set is frozen once created (AGENTS.md rule 2).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.jsonl"
    if test_path.exists() and not force_test:
        raise RuntimeError(
            f"{test_path} already exists and the test set is frozen (AGENTS.md rule 2). "
            "Pass force_test=True only for a deliberate, documented change."
        )

    paths: dict[Split, Path] = {}
    for split_name, examples in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w") as f:
            for example in examples:
                f.write(example.model_dump_json() + "\n")
        paths[split_name] = path
    return paths
