from __future__ import annotations

from t2s.eval.few_shot import render_examples, select_few_shot_examples
from t2s.synth.models import DifficultyTag, Example


def make_example(tag: DifficultyTag, i: int) -> Example:
    return Example(question=f"question {tag} {i}", sql=f"SELECT {i}", tags=[tag], split="train")


def test_select_few_shot_examples_respects_k() -> None:
    examples = [make_example("single_table", i) for i in range(5)] + [
        make_example("join", i) for i in range(5)
    ]
    selected = select_few_shot_examples(examples, k=3, seed=42)
    assert len(selected) == 3


def test_select_few_shot_examples_covers_multiple_tags_when_possible() -> None:
    examples = [make_example("single_table", i) for i in range(5)] + [
        make_example("join", i) for i in range(5)
    ]
    selected = select_few_shot_examples(examples, k=2, seed=42)
    tags = {e.tags[0] for e in selected}
    assert tags == {"single_table", "join"}


def test_select_few_shot_examples_deterministic_for_fixed_seed() -> None:
    examples = [make_example("single_table", i) for i in range(10)]
    first = select_few_shot_examples(examples, k=4, seed=42)
    second = select_few_shot_examples(examples, k=4, seed=42)
    assert [e.question for e in first] == [e.question for e in second]


def test_select_few_shot_examples_zero_k_returns_empty() -> None:
    examples = [make_example("single_table", 0)]
    assert select_few_shot_examples(examples, k=0, seed=42) == []


def test_select_few_shot_examples_caps_at_available_examples() -> None:
    examples = [make_example("single_table", 0)]
    selected = select_few_shot_examples(examples, k=5, seed=42)
    assert len(selected) == 1


def test_render_examples_formats_question_and_sql() -> None:
    examples = [make_example("single_table", 0), make_example("join", 1)]
    rendered = render_examples(examples)
    assert "Question: question single_table 0" in rendered
    assert "SQL: SELECT 0" in rendered
    assert "Question: question join 1" in rendered


def test_render_examples_empty_list() -> None:
    assert render_examples([]) == ""
