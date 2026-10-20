from __future__ import annotations

from t2s.eval.postprocess import clean_sql_output


def test_clean_sql_output_passes_through_bare_sql() -> None:
    assert clean_sql_output("SELECT 1") == "SELECT 1"


def test_clean_sql_output_strips_sql_label_prefix() -> None:
    assert clean_sql_output("SQL: SELECT 1") == "SELECT 1"


def test_clean_sql_output_strips_label_case_insensitive() -> None:
    assert clean_sql_output("sql:   SELECT 1") == "SELECT 1"
    assert clean_sql_output("Answer: SELECT 1") == "SELECT 1"
    assert clean_sql_output("Query: SELECT 1") == "SELECT 1"


def test_clean_sql_output_strips_markdown_fence() -> None:
    assert clean_sql_output("```sql\nSELECT 1\n```") == "SELECT 1"
    assert clean_sql_output("```\nSELECT 1\n```") == "SELECT 1"


def test_clean_sql_output_strips_fence_then_label() -> None:
    assert clean_sql_output("```sql\nSQL: SELECT 1\n```") == "SELECT 1"


def test_clean_sql_output_strips_surrounding_whitespace() -> None:
    assert clean_sql_output("  SELECT 1  \n") == "SELECT 1"
