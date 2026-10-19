"""Execution filter: parse (sqlglot, Spark dialect) and execute candidate SQL
against the silver Delta tables. Only examples that parse and execute --
returning a non-empty result, or an empty one the question explicitly
expects -- are kept. See AGENTS.md rule 5.
"""

from __future__ import annotations

from pathlib import Path

import sqlglot
from pyspark.sql import SparkSession

from t2s.common.config import BaseConfig
from t2s.ingest.silver import (
    SILVER_AGENCIES_TABLE,
    SILVER_FACT_TABLE,
    SILVER_LOCATIONS_TABLE,
    SILVER_RECIPIENTS_TABLE,
)
from t2s.synth.models import CandidateExample, ExecutedExample

SILVER_TABLES = [
    SILVER_FACT_TABLE,
    SILVER_AGENCIES_TABLE,
    SILVER_RECIPIENTS_TABLE,
    SILVER_LOCATIONS_TABLE,
]


def register_silver_tables(spark: SparkSession, config: BaseConfig) -> None:
    """Register each silver Delta table as a temp view under its table name,
    so generated SQL like `SELECT ... FROM award_transactions` just works."""
    warehouse_dir = Path(config.spark.warehouse_dir)
    for table in SILVER_TABLES:
        df = spark.read.format("delta").load(str(warehouse_dir / table))
        df.createOrReplaceTempView(table)


def validate_sql_syntax(sql: str) -> str | None:
    """Parse sql with sqlglot's Spark dialect. Returns an error message, or None if valid."""
    try:
        sqlglot.parse_one(sql, dialect="spark")
    except sqlglot.errors.ParseError as e:
        return str(e)
    return None


def execute_candidate(spark: SparkSession, candidate: CandidateExample) -> ExecutedExample:
    """Parse and execute one candidate; return it annotated with validity/row count/error."""
    syntax_error = validate_sql_syntax(candidate.sql)
    if syntax_error is not None:
        return ExecutedExample(**candidate.model_dump(), valid=False, error=syntax_error)

    try:
        result = spark.sql(candidate.sql)
        row_count = result.count()
    except Exception as e:  # noqa: BLE001 - any Spark/SQL execution error disqualifies the candidate
        return ExecutedExample(**candidate.model_dump(), valid=False, error=str(e))

    if row_count == 0 and not candidate.expects_empty_result:
        return ExecutedExample(
            **candidate.model_dump(),
            valid=False,
            row_count=row_count,
            error="empty result, but expects_empty_result=False",
        )

    return ExecutedExample(**candidate.model_dump(), valid=True, row_count=row_count)


def filter_candidates(
    spark: SparkSession, candidates: list[CandidateExample]
) -> list[ExecutedExample]:
    """Execute every candidate and return only the valid ones."""
    executed = [execute_candidate(spark, c) for c in candidates]
    return [e for e in executed if e.valid]
