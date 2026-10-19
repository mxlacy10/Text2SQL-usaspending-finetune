import pytest

from t2s.synth.generate import build_generation_prompt, parse_candidates
from t2s.synth.models import CandidateExample


def test_build_generation_prompt_fills_template() -> None:
    prompt = build_generation_prompt("SCHEMA CARD TEXT", "aggregation", 5)
    assert "SCHEMA CARD TEXT" in prompt
    assert "aggregation" in prompt
    assert "5 distinct" in prompt


def test_parse_candidates_plain_json() -> None:
    raw = (
        '[{"question": "How many awards?", "sql": "SELECT COUNT(*) FROM award_transactions", '
        '"tags": ["aggregation"], "expects_empty_result": false}]'
    )
    candidates = parse_candidates(raw)
    assert candidates == [
        CandidateExample(
            question="How many awards?",
            sql="SELECT COUNT(*) FROM award_transactions",
            tags=["aggregation"],
            expects_empty_result=False,
        )
    ]


def test_parse_candidates_strips_markdown_fence() -> None:
    raw = '```json\n[{"question": "q", "sql": "SELECT 1", "tags": ["single_table"]}]\n```'
    candidates = parse_candidates(raw)
    assert len(candidates) == 1
    assert candidates[0].question == "q"


def test_parse_candidates_rejects_unknown_tag() -> None:
    raw = '[{"question": "q", "sql": "SELECT 1", "tags": ["not_a_real_tag"]}]'
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        parse_candidates(raw)
