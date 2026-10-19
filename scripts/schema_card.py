"""CLI entry point to (re)generate data/schema_card.md from the silver Delta tables.

Usage:
  uv run python scripts/schema_card.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from t2s.common.config import load_base_config
from t2s.common.spark import get_spark_session
from t2s.schema.card import DEFAULT_OUTPUT_PATH, build_schema_card

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to a base config YAML file."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the schema card markdown.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    config = load_base_config(args.config)

    spark = get_spark_session(config)
    try:
        path = build_schema_card(spark, config, output_path=Path(args.output))
        logger.info("Wrote schema card to %s", path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
