"""Optimize the DSPy Text2SQL program on the train split and save it.

Run in the Databricks-Connect environment so the metric executes SQL on Unity Catalog:

  set -a && . ./.env && set +a
  .venv-dbx/bin/python scripts/dspy_optimize.py --spark-mode databricks

The test split is never read here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from t2s.common.config import load_base_config
from t2s.common.prompt_loader import load_prompt
from t2s.common.spark import get_spark_session
from t2s.eval.config import load_eval_config
from t2s.eval.data import load_examples
from t2s.eval.dspy_baseline import build_program, make_lm, optimize, save_program
from t2s.eval.execute import execute_sql
from t2s.eval.few_shot import select_few_shot_examples
from t2s.eval.metrics import rows_match, sql_has_order_by
from t2s.synth.execute import register_silver_tables

logger = logging.getLogger(__name__)

INSTRUCTIONS = "dspy_sql_instructions_v1.md"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/eval.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--spark-mode", choices=["local", "databricks"], default=None)
    parser.add_argument("--n-train", type=int, default=40, help="train examples for bootstrapping")
    parser.add_argument("--max-demos", type=int, default=5)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Put the schema card once in the instructions instead of in every demo.",
    )
    args = parser.parse_args()
    program_path = Path(
        "results/dspy_frontier_compact_program.json"
        if args.compact
        else "results/dspy_frontier_program.json"
    )

    eval_config = load_eval_config(args.config)
    base_config = load_base_config(args.base_config)
    if args.spark_mode is not None:
        base_config.spark.mode = args.spark_mode
    if base_config.models.frontier is None or base_config.models.frontier_azure_endpoint is None:
        raise RuntimeError("config.models.frontier and frontier_azure_endpoint must be set.")

    train = select_few_shot_examples(
        load_examples(eval_config.train_split), args.n_train, eval_config.seed
    )
    schema_card = Path(eval_config.schema_card).read_text()
    lm = make_lm(base_config.models.frontier, base_config.models.frontier_azure_endpoint)

    spark = get_spark_session(base_config)
    try:
        register_silver_tables(spark, base_config)

        def exec_match(pred_sql: str, gold_sql: str) -> bool:
            pred = execute_sql(spark, pred_sql)
            gold = execute_sql(spark, gold_sql)
            return (
                pred.valid
                and gold.valid
                and rows_match(gold.rows, pred.rows, ordered=sql_has_order_by(gold_sql))
            )

        import dspy

        program = build_program(
            load_prompt(INSTRUCTIONS).text, schema_card if args.compact else None
        )
        with dspy.context(lm=lm):
            optimized = optimize(
                program, train, schema_card, exec_match, args.max_demos, args.compact
            )
    finally:
        spark.stop()

    save_program(optimized, program_path)
    logger.info("saved %d demos -> %s", len(optimized.demos), program_path)


if __name__ == "__main__":
    main()
