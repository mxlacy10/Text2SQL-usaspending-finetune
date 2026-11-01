"""Spark session factory.

`spark.mode: local` builds a local SparkSession with Delta Lake support.
`spark.mode: databricks` connects to Unity Catalog via Databricks Connect and sets
the current catalog/schema, so unqualified table names in the gold SQL resolve
unchanged. The mode is never chosen implicitly: a failure to connect raises.
"""

from __future__ import annotations

from typing import Any

from t2s.common.config import BaseConfig


def _quote(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def get_spark_session(config: BaseConfig) -> Any:
    """Return a SparkSession for config.spark.mode (a Spark Connect session in databricks mode)."""
    if config.spark.mode == "databricks":
        return _get_databricks_session(config)
    return _get_local_session(config)


def _get_databricks_session(config: BaseConfig) -> Any:
    # databricks-connect replaces the pyspark package, so it lives in its own
    # environment (see docs/databricks_plan.md Run log) and is imported lazily.
    from databricks.connect import DatabricksSession

    if not config.catalog:  # also enforced by BaseConfig validation
        raise ValueError("spark.mode=databricks requires `catalog` to be set")
    spark = DatabricksSession.builder.getOrCreate()
    spark.sql(f"USE CATALOG {_quote(config.catalog)}")
    spark.sql(f"USE SCHEMA {_quote(config.schema_name)}")
    return spark


def _get_local_session(config: BaseConfig) -> Any:
    """Build a local SparkSession configured for Delta Lake.

    Idempotent: calling this repeatedly returns the same underlying JVM
    session (Spark reuses an active session within a process).
    """
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(config.spark.app_name)
        .config("spark.sql.warehouse.dir", config.spark.warehouse_dir)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.master", "local[*]")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
