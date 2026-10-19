"""Pure metric functions for the eval harness.

Primary metric per AGENTS.md: **execution accuracy (EX)** -- the predicted
SQL's result set equals the gold result set, order-insensitive unless the
gold query has an explicit ORDER BY, with float tolerance 1e-6 relative.
Also: exact match after sqlglot normalization, and valid-SQL rate (handled
by the execution layer, not here).
"""

from __future__ import annotations

import math
from collections import Counter

import sqlglot
from sqlglot import exp

FLOAT_RELATIVE_TOLERANCE = 1e-6

Row = tuple[object, ...]


def normalize_sql(sql: str, dialect: str = "spark") -> str | None:
    """Normalize SQL text via sqlglot for exact-match comparison.

    Returns None if the SQL doesn't parse (callers should treat that as a
    non-match rather than raising).
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return None
    return tree.sql(dialect=dialect, normalize=True)


def exact_match(gold_sql: str, predicted_sql: str, dialect: str = "spark") -> bool:
    """True if gold and predicted SQL normalize to the identical query string."""
    gold_norm = normalize_sql(gold_sql, dialect)
    pred_norm = normalize_sql(predicted_sql, dialect)
    if gold_norm is None or pred_norm is None:
        return False
    return gold_norm == pred_norm


def sql_has_order_by(sql: str, dialect: str = "spark") -> bool:
    """True if the (gold) SQL has an explicit top-level ORDER BY.

    Used to decide whether result-set comparison should be order-sensitive:
    per AGENTS.md, EX is "compared order-insensitively unless the question
    implies ordering" -- we treat an explicit ORDER BY in the gold SQL as
    that signal.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    return tree.find(exp.Order) is not None


def _values_equal(a: object, b: object) -> bool:
    """Compare two cell values with NULL-awareness and float tolerance."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=FLOAT_RELATIVE_TOLERANCE)
    return a == b


def _row_key(row: Row) -> tuple[object, ...]:
    """A hashable, float-tolerant-ish key for an unordered multiset comparison.

    Floats are rounded to a fixed number of decimals so that values within the
    relative tolerance collapse to the same bucket for Counter-based grouping;
    exact equality between rows is re-checked with `_values_equal` on any
    ambiguous mismatch via `rows_match`'s row-by-row fallback below.
    """
    key: list[object] = []
    for value in row:
        if isinstance(value, bool) or value is None:
            key.append(value)
        elif isinstance(value, (int, float)):
            key.append(round(float(value), 6))
        else:
            key.append(value)
    return tuple(key)


def _rows_equal_ordered(gold_rows: list[Row], pred_rows: list[Row]) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    return all(
        len(g) == len(p) and all(_values_equal(gv, pv) for gv, pv in zip(g, p, strict=True))
        for g, p in zip(gold_rows, pred_rows, strict=True)
    )


def _rows_equal_unordered(gold_rows: list[Row], pred_rows: list[Row]) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    gold_counts = Counter(_row_key(r) for r in gold_rows)
    pred_counts = Counter(_row_key(r) for r in pred_rows)
    return gold_counts == pred_counts


def rows_match(
    gold_rows: list[Row],
    pred_rows: list[Row],
    ordered: bool,
) -> bool:
    """Execution-accuracy row-set comparison.

    - `ordered=True`: row order matters (gold SQL had an explicit ORDER BY).
    - `ordered=False`: multiset comparison (duplicate rows must match counts,
      but order doesn't matter).
    Both modes are NULL-aware and float-tolerant (see `_values_equal`).
    """
    if ordered:
        return _rows_equal_ordered(gold_rows, pred_rows)
    return _rows_equal_unordered(gold_rows, pred_rows)
