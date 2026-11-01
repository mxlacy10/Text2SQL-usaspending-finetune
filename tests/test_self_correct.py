from __future__ import annotations

from t2s.eval.self_correct import SelfCorrectingPredictor

TEMPLATE = "SQL: {sql}\nError: {error}"


class FakeBase:
    def __init__(self, first: str, repairs: list[str]) -> None:
        self._first = first
        self._repairs = list(repairs)
        self.repair_calls: list[tuple[str, str]] = []
        self.last_usage: tuple[int, int] | None = (10, 5)

    def predict(self, question: str) -> tuple[str, float]:
        return self._first, 100.0

    def repair(
        self, question: str, previous_sql: str, error: str, template: str
    ) -> tuple[str, float]:
        self.repair_calls.append((previous_sql, error))
        return self._repairs.pop(0), 50.0


def test_valid_first_attempt_is_not_repaired() -> None:
    base = FakeBase("SELECT 1", [])
    p = SelfCorrectingPredictor(base, lambda sql: None, TEMPLATE)

    assert p.predict("q") == ("SELECT 1", 100.0)
    assert base.repair_calls == []
    assert p.last_attempts == 1
    assert p.last_usage == (10, 5)


def test_repairs_until_valid_and_sums_latency_and_tokens() -> None:
    base = FakeBase("BAD1", ["BAD2", "GOOD"])
    errors = {"BAD1": "e1", "BAD2": "e2"}
    p = SelfCorrectingPredictor(base, lambda sql: errors.get(sql), TEMPLATE, max_repairs=2)

    sql, ms = p.predict("q")

    assert sql == "GOOD"
    assert ms == 200.0
    assert base.repair_calls == [("BAD1", "e1"), ("BAD2", "e2")]
    assert p.last_attempts == 3
    assert p.last_usage == (30, 15)


def test_stops_after_max_repairs_and_returns_last_attempt() -> None:
    base = FakeBase("BAD1", ["BAD2", "BAD3"])
    p = SelfCorrectingPredictor(base, lambda sql: "err", TEMPLATE, max_repairs=2)

    assert p.predict("q")[0] == "BAD3"
    assert len(base.repair_calls) == 2


def test_long_errors_are_truncated() -> None:
    base = FakeBase("BAD", ["GOOD"])
    errors = {"BAD": "x" * 5000}
    p = SelfCorrectingPredictor(base, lambda sql: errors.get(sql), TEMPLATE, max_repairs=1)

    p.predict("q")

    assert len(base.repair_calls[0][1]) == 500
