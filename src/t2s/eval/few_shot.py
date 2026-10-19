"""Deterministic few-shot example selection for the schema-RAG baseline.

Per AGENTS.md: "base model, few-shot (k=5) plus schema-RAG" draws its
examples from the train split (never from dev/test, which must stay unseen
by any baseline) -- stratified across difficulty tags and seeded for
reproducibility.
"""

from __future__ import annotations

import random
from collections import defaultdict

from t2s.synth.models import Example


def select_few_shot_examples(train_examples: list[Example], k: int, seed: int) -> list[Example]:
    """Pick k examples from train_examples, round-robin across each example's
    primary (first) difficulty tag so the few-shot pool covers as many
    difficulty tags as k allows, with a fixed seed for reproducibility."""
    if k <= 0:
        return []

    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = defaultdict(list)
    for example in train_examples:
        primary_tag = example.tags[0] if example.tags else "single_table"
        buckets[primary_tag].append(example)
    for items in buckets.values():
        rng.shuffle(items)

    tags = sorted(buckets)
    selected: list[Example] = []
    i = 0
    while len(selected) < k and tags:
        tag = tags[i % len(tags)]
        if buckets[tag]:
            selected.append(buckets[tag].pop())
        else:
            tags.remove(tag)
            continue
        i += 1
        if not tags:
            break
    return selected[:k]


def render_examples(examples: list[Example]) -> str:
    """Render few-shot examples as "Question: ...\\nSQL: ..." blocks."""
    blocks = [f"Question: {e.question}\nSQL: {e.sql}" for e in examples]
    return "\n\n".join(blocks)
