"""CLI entry point for synthetic question/SQL generation: generate -> execute
filter -> dedupe -> stratified split -> write data/splits/*.jsonl.

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment, matching
whichever frontier model is set in config.models.frontier.

Usage:
  uv run python scripts/synth.py --config config/base.yaml --n-per-tag 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from t2s.common.config import load_base_config
from t2s.common.prompt_loader import load_prompt
from t2s.common.spark import get_spark_session
from t2s.synth.dedupe import dedupe_candidates
from t2s.synth.execute import filter_candidates, register_silver_tables
from t2s.synth.generate import PROMPT_NAME, generate_candidates
from t2s.synth.llm import get_frontier_client
from t2s.synth.models import DifficultyTag
from t2s.synth.splits import stratified_split, write_splits

logger = logging.getLogger(__name__)

ALL_TAGS: list[DifficultyTag] = [
    "single_table",
    "aggregation",
    "join",
    "filter_date_fy",
    "top_n",
    "nested",
    "window",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/base.yaml", help="Path to a base config YAML file."
    )
    parser.add_argument(
        "--schema-card", default="data/schema_card.md", help="Path to the generated schema card."
    )
    parser.add_argument(
        "--out-dir", default="data/splits", help="Directory to write train/dev/test.jsonl to."
    )
    parser.add_argument("--n-per-tag", type=int, default=20, help="Candidates to generate per tag.")
    parser.add_argument(
        "--tags", default=",".join(ALL_TAGS), help="Comma-separated difficulty tags to generate."
    )
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="Allow overwriting an existing frozen test.jsonl (AGENTS.md rule 2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    config = load_base_config(args.config)

    if config.models.frontier is None:
        raise RuntimeError(
            "config.models.frontier is not set. Record the exact frontier model ID "
            "used for synthetic generation (AGENTS.md: 'record exact model ID and date')."
        )

    prompt = load_prompt(PROMPT_NAME)
    logger.info("Using prompt %s (sha256=%s)", prompt.name, prompt.sha256)

    schema_card = Path(args.schema_card).read_text()
    tags: list[DifficultyTag] = [t.strip() for t in args.tags.split(",") if t.strip()]

    client = get_frontier_client(config.models.frontier)
    logger.info("Generating %d candidates per tag for tags=%s", args.n_per_tag, tags)
    candidates = generate_candidates(client, schema_card, tags, args.n_per_tag)
    logger.info("Generated %d raw candidates", len(candidates))

    spark = get_spark_session(config)
    try:
        register_silver_tables(spark, config)
        executed = filter_candidates(spark, candidates)
        logger.info("%d/%d candidates executed successfully", len(executed), len(candidates))

        deduped = dedupe_candidates(executed)
        logger.info("%d candidates remain after dedup", len(deduped))

        splits = stratified_split(deduped, seed=config.seed)
        paths = write_splits(splits, Path(args.out_dir), force_test=args.force_test)
        for split_name, path in paths.items():
            logger.info("Wrote %d %s examples to %s", len(splits[split_name]), split_name, path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
