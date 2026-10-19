"""Spark session factory with Delta Lake support.

Local mode only. On Databricks, Unity Catalog tables are used instead of this
session (see AGENTS.md: config/base.yaml: catalog).
"""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from t2s.common.config import BaseConfig


def get_spark_session(config: BaseConfig) -> SparkSession:
    """Build a local SparkSession configured for Delta Lake.

    Idempotent: calling this repeatedly returns the same underlying JVM
    session (Spark reuses an active session within a process).
    """
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
