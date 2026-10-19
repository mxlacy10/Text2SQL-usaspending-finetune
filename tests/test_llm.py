import pytest

from t2s.synth.llm import get_frontier_client


def test_get_frontier_client_dispatches_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = get_frontier_client("gpt-4o")
    assert type(client).__name__ == "OpenAIClient"


def test_get_frontier_client_dispatches_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = get_frontier_client("claude-3-5-sonnet-20241022")
    assert type(client).__name__ == "AnthropicClient"


def test_get_frontier_client_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_frontier_client("gpt-4o")


def test_get_frontier_client_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Don't know which provider"):
        get_frontier_client("mystery-model-9000")
