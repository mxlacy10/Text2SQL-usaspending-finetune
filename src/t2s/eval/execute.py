"""SQL execution helpers for the eval harness.

Parses with sqlglot (Spark dialect) and executes against the registered
silver Delta tables, same tables used by synth (see
`t2s.synth.execute.register_silver_tables`). Unlike the synth execution
filter (which only needs a row count), eval needs the actual rows to compare
against gold.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import SparkSession

from t2s.eval.metrics import Row, normalize_sql


@dataclass(frozen=True)
class ExecutionResult:
    valid: bool
    rows: list[Row]
    error: str | None = None


def execute_sql(spark: SparkSession, sql: str, dialect: str = "spark") -> ExecutionResult:
    """Parse then execute one SQL statement, returning its result rows.

    A parse failure or execution error both produce `valid=False` with the
    error message captured (and are therefore always treated as a non-match
    by the eval harness, never as a false positive).
    """
    if normalize_sql(sql, dialect) is None:
        return ExecutionResult(valid=False, rows=[], error="SQL failed to parse")

    try:
        result = spark.sql(sql)
        rows: list[Row] = [tuple(row) for row in result.collect()]
    except Exception as e:  # noqa: BLE001 - any Spark/SQL execution error disqualifies the result
        return ExecutionResult(valid=False, rows=[], error=str(e))

    return ExecutionResult(valid=True, rows=rows)
