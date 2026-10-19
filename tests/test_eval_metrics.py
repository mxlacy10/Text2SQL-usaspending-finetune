"""Tests for pure execution-accuracy metric functions. Per AGENTS.md: "EX
comparison logic must have tests for ordering, NULLs, float tolerance, and
duplicate rows."""

from __future__ import annotations

from t2s.eval.metrics import exact_match, normalize_sql, rows_match, sql_has_order_by


def test_normalize_sql_ignores_whitespace_and_case_of_keywords() -> None:
    a = normalize_sql("select a, b from t where x = 1")
    b = normalize_sql("SELECT   a,b\nFROM t\nWHERE x=1")
    assert a is not None
    assert a == b


def test_normalize_sql_returns_none_for_invalid_sql() -> None:
    assert normalize_sql("SELECT FROM WHERE") is None


def test_normalize_sql_returns_none_for_unterminated_quote() -> None:
    # sqlglot raises TokenError (not ParseError) for malformed tokens like an
    # unterminated backtick-quoted identifier -- both must be treated as "did
    # not parse" rather than propagating.
    assert normalize_sql("SELECT `unterminated FROM t") is None


def test_exact_match_true_for_equivalent_formatting() -> None:
    assert exact_match("SELECT a FROM t WHERE x = 1", "select   a from t where x=1")


def test_exact_match_false_for_different_queries() -> None:
    assert not exact_match("SELECT a FROM t", "SELECT b FROM t")


def test_exact_match_false_when_either_side_invalid() -> None:
    assert not exact_match("SELECT FROM WHERE", "SELECT a FROM t")


def test_sql_has_order_by_true() -> None:
    assert sql_has_order_by("SELECT a FROM t ORDER BY a")


def test_sql_has_order_by_false() -> None:
    assert not sql_has_order_by("SELECT a FROM t")


def test_sql_has_order_by_false_for_invalid_sql() -> None:
    assert not sql_has_order_by("SELECT FROM WHERE")


# --- rows_match: ordering ---------------------------------------------------


def test_rows_match_unordered_ignores_row_order() -> None:
    gold = [(1, "a"), (2, "b")]
    pred = [(2, "b"), (1, "a")]
    assert rows_match(gold, pred, ordered=False)


def test_rows_match_ordered_requires_same_row_order() -> None:
    gold = [(1, "a"), (2, "b")]
    pred = [(2, "b"), (1, "a")]
    assert not rows_match(gold, pred, ordered=True)
    assert rows_match(gold, gold, ordered=True)


# --- rows_match: NULLs -------------------------------------------------------


def test_rows_match_treats_none_as_equal_to_none() -> None:
    gold = [(1, None), (2, "b")]
    pred = [(1, None), (2, "b")]
    assert rows_match(gold, pred, ordered=False)


def test_rows_match_none_does_not_equal_other_values() -> None:
    gold = [(1, None)]
    pred = [(1, 0)]
    assert not rows_match(gold, pred, ordered=False)


# --- rows_match: float tolerance --------------------------------------------


def test_rows_match_float_within_relative_tolerance() -> None:
    gold = [(1, 100.0000001)]
    pred = [(1, 100.0000002)]
    assert rows_match(gold, pred, ordered=False)


def test_rows_match_float_outside_relative_tolerance() -> None:
    gold = [(1, 100.0)]
    pred = [(1, 100.01)]
    assert not rows_match(gold, pred, ordered=False)


def test_rows_match_int_and_float_equal_values() -> None:
    gold = [(1, 100)]
    pred = [(1, 100.0)]
    assert rows_match(gold, pred, ordered=False)


# --- rows_match: duplicate rows ---------------------------------------------


def test_rows_match_unordered_respects_duplicate_counts() -> None:
    gold = [(1,), (1,), (2,)]
    pred = [(1,), (2,), (2,)]
    assert not rows_match(gold, pred, ordered=False)


def test_rows_match_unordered_matches_when_duplicate_counts_equal() -> None:
    gold = [(1,), (1,), (2,)]
    pred = [(1,), (2,), (1,)]
    assert rows_match(gold, pred, ordered=False)


def test_rows_match_different_lengths_never_match() -> None:
    assert not rows_match([(1,)], [(1,), (1,)], ordered=False)
