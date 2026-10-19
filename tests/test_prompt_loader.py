import pytest

from t2s.common.prompt_loader import load_prompt


def test_load_prompt_returns_text_and_stable_hash() -> None:
    prompt = load_prompt("synth_question_sql_v1.md")
    assert "{schema_card}" in prompt.text
    assert "{tag}" in prompt.text
    assert len(prompt.sha256) == 64

    # Loading again should produce the exact same hash (stable/reproducible).
    prompt_again = load_prompt("synth_question_sql_v1.md")
    assert prompt.sha256 == prompt_again.sha256


def test_load_prompt_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.md")
