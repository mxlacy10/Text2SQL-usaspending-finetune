"""Execute-and-repair wrapper: if the predicted SQL fails to parse or run,
feed the error back to the model and retry (up to `max_repairs` times).

Only the model's own SQL and the engine's error message are used -- the gold
SQL and gold result are never visible to the loop, so this is not test-set
leakage. Latency and token usage include every attempt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

MAX_ERROR_CHARS = 500


class RepairablePredictor(Protocol):
    def predict(self, question: str) -> tuple[str, float]: ...

    def repair(
        self, question: str, previous_sql: str, error: str, template: str
    ) -> tuple[str, float]:
        """Returns (corrected_sql, latency_ms) given the failed attempt."""
        ...


# Returns an error message when the SQL is not executable, else None.
Validator = Callable[[str], str | None]


class SelfCorrectingPredictor:
    def __init__(
        self,
        base: RepairablePredictor,
        validate: Validator,
        repair_template: str,
        max_repairs: int = 2,
    ) -> None:
        self._base = base
        self._validate = validate
        self._repair_template = repair_template
        self._max_repairs = max_repairs
        self.last_usage: tuple[int, int] | None = None
        self.last_attempts = 0

    def _add_usage(self) -> None:
        usage: tuple[int, int] | None = getattr(self._base, "last_usage", None)
        if usage is None:
            return
        prev = self.last_usage or (0, 0)
        self.last_usage = (prev[0] + usage[0], prev[1] + usage[1])

    def predict(self, question: str) -> tuple[str, float]:
        self.last_usage = None
        sql, total_ms = self._base.predict(question)
        self._add_usage()
        self.last_attempts = 1
        for _ in range(self._max_repairs):
            error = self._validate(sql)
            if error is None:
                break
            sql, ms = self._base.repair(
                question, sql, error[:MAX_ERROR_CHARS], self._repair_template
            )
            total_ms += ms
            self.last_attempts += 1
            self._add_usage()
        return sql, total_ms
