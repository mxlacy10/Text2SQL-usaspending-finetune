"""CLI entry point for the ingest pipeline: download -> bronze -> silver.

Usage:
  uv run python scripts/ingest.py --config config/base.yaml
  uv run python scripts/ingest.py --config config/base.yaml --steps bronze,silver  # skip download
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from t2s.common.config import BaseConfig, load_base_config
from t2s.common.spark import get_spark_session
from t2s.ingest.bronze import BRONZE_TABLE, build_bronze
from t2s.ingest.download import download_award_data
from t2s.ingest.silver import build_silver

logger = logging.getLogger(__name__)

ALL_STEPS = ["download", "bronze", "silver"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to a base config YAML file."
    )
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated subset of steps to run: {ALL_STEPS}.",
    )
    return parser.parse_args(argv)


def bronze_table_path(config: BaseConfig) -> Path:
    return Path(config.spark.warehouse_dir) / BRONZE_TABLE


def run(config: BaseConfig, steps: list[str]) -> None:
    logging.basicConfig(level=logging.INFO)

    if "download" in steps:
        logger.info(
            "Downloading USAspending award data for fiscal years %s", config.data.fiscal_years
        )
        download_award_data(config)

    if "bronze" not in steps and "silver" not in steps:
        return

    spark = get_spark_session(config)
    try:
        table_path = bronze_table_path(config)
        if "bronze" in steps:
            logger.info("Building bronze table from %s", config.data.raw_dir)
            table_path = build_bronze(spark, config)
            logger.info("Wrote bronze table to %s", table_path)

        if "silver" in steps:
            logger.info("Building silver tables from bronze table %s", table_path)
            silver_paths = build_silver(spark, config, table_path)
            for name, path in silver_paths.items():
                logger.info("Wrote silver table %s to %s", name, path)
    finally:
        spark.stop()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_base_config(args.config)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    run(config, steps)


if __name__ == "__main__":
    main()
