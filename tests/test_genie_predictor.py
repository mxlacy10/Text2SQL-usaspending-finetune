from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from t2s.eval.predictors import GeniePredictor, extract_genie_sql


def _msg(attachments: list[Any] | None = None, error: Any = None) -> SimpleNamespace:
    return SimpleNamespace(attachments=attachments, error=error)


def _query_attachment(sql: str) -> SimpleNamespace:
    return SimpleNamespace(query=SimpleNamespace(query=sql), text=None)


def test_extract_sql_from_query_attachment() -> None:
    assert extract_genie_sql(_msg([_query_attachment("  SELECT 1  ")])) == "SELECT 1"


def test_extract_sql_skips_text_only_attachment() -> None:
    text_only = SimpleNamespace(query=None, text=SimpleNamespace(content="I can't answer that"))
    assert extract_genie_sql(_msg([text_only])) == ""
    assert extract_genie_sql(_msg([text_only, _query_attachment("SELECT 2")])) == "SELECT 2"


def test_extract_sql_error_message_is_empty() -> None:
    assert extract_genie_sql(_msg([_query_attachment("SELECT 1")], error=object())) == ""
    assert extract_genie_sql(_msg(None)) == ""


class TooManyRequests(Exception):
    pass


class FakeGenie:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str]] = []

    def start_conversation_and_wait(self, space_id: str, content: str) -> Any:
        self.calls.append((space_id, content))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_predict_returns_sql_and_latency() -> None:
    genie = FakeGenie([_msg([_query_attachment("SELECT 1")])])
    sql, latency = GeniePredictor(genie, "space").predict("q")
    assert sql == "SELECT 1" and latency >= 0
    assert genie.calls == [("space", "q")]


def test_predict_retries_rate_limit_then_succeeds() -> None:
    genie = FakeGenie([TooManyRequests(), _msg([_query_attachment("SELECT 1")])])
    sql, _ = GeniePredictor(genie, "space", base_backoff_s=0).predict("q")
    assert sql == "SELECT 1" and len(genie.calls) == 2


def test_predict_gives_up_after_max_retries_as_empty_sql() -> None:
    genie = FakeGenie([TooManyRequests() for _ in range(3)])
    sql, _ = GeniePredictor(genie, "space", max_retries=2, base_backoff_s=0).predict("q")
    assert sql == "" and len(genie.calls) == 3


def test_predict_other_errors_are_invalid_not_skipped() -> None:
    genie = FakeGenie([RuntimeError("boom")])
    sql, _ = GeniePredictor(genie, "space", base_backoff_s=0).predict("q")
    assert sql == "" and len(genie.calls) == 1
