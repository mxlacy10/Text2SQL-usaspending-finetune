"""CLI entry point for the eval harness: runs one of the four AGENTS.md
baselines against the frozen test set and writes results/<model>.json
(+ logs to MLflow).

Baselines (--model):
  base_zero_shot   base mlx model, zero-shot, no adapter
  base_few_shot    base mlx model, few-shot (k=5) + schema-RAG, no adapter
  frontier         frontier model (Azure AI Foundry), few-shot + schema-RAG
  finetuned        base mlx model + LoRA adapter, zero-shot

Usage:
  uv run python scripts/eval.py --model finetuned --config config/eval.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Literal

import mlflow

from t2s.common.config import load_base_config
from t2s.common.mlflow_utils import init_mlflow
from t2s.common.prompt_loader import load_prompt
from t2s.common.spark import get_spark_session
from t2s.eval.config import load_eval_config
from t2s.eval.data import load_examples
from t2s.eval.few_shot import render_examples, select_few_shot_examples
from t2s.eval.harness import run_eval
from t2s.eval.predictors import (
    FrontierFewShotPredictor,
    MlxFewShotPredictor,
    MlxZeroShotPredictor,
    Predictor,
)
from t2s.synth.execute import register_silver_tables
from t2s.synth.llm import get_frontier_client
from t2s.train.config import load_train_config

logger = logging.getLogger(__name__)

ModelChoice = Literal["base_zero_shot", "base_few_shot", "frontier", "finetuned"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["base_zero_shot", "base_few_shot", "frontier", "finetuned"],
    )
    parser.add_argument("--config", default="config/eval.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    model_choice: ModelChoice = args.model

    eval_config = load_eval_config(args.config)
    base_config = load_base_config(args.base_config)
    train_config = load_train_config(args.train_config)

    test_examples = load_examples(eval_config.test_split)
    logger.info("Loaded %d test examples", len(test_examples))

    schema_card = Path(eval_config.schema_card).read_text()

    zero_shot_prompt = load_prompt(eval_config.zero_shot_prompt)
    few_shot_prompt = load_prompt(eval_config.few_shot_prompt)

    train_examples = load_examples(eval_config.train_split)
    few_shot_examples = select_few_shot_examples(
        train_examples, eval_config.k_shot, eval_config.seed
    )
    examples_block = render_examples(few_shot_examples)

    zero_shot_models = ("base_zero_shot", "finetuned")
    prompt_used = zero_shot_prompt if model_choice in zero_shot_models else few_shot_prompt

    predictor: Predictor
    if model_choice == "base_zero_shot":
        predictor = MlxZeroShotPredictor(
            train_config.mlx_model, zero_shot_prompt.text, eval_config.max_tokens
        )
    elif model_choice == "finetuned":
        predictor = MlxZeroShotPredictor(
            train_config.mlx_model,
            zero_shot_prompt.text,
            eval_config.max_tokens,
            adapter_path=train_config.adapter_path,
        )
    elif model_choice == "base_few_shot":
        predictor = MlxFewShotPredictor(
            train_config.mlx_model,
            few_shot_prompt.text,
            schema_card,
            examples_block,
            eval_config.max_tokens,
        )
    else:  # frontier
        if base_config.models.frontier is None:
            raise RuntimeError(
                "config.models.frontier is not set; cannot run the frontier baseline."
            )
        client = get_frontier_client(
            base_config.models.frontier, base_config.models.frontier_azure_endpoint
        )
        predictor = FrontierFewShotPredictor(
            client, few_shot_prompt.text, schema_card, examples_block
        )

    spark = get_spark_session(base_config)
    try:
        register_silver_tables(spark, base_config)
        summary = run_eval(
            model_name=model_choice,
            examples=test_examples,
            predictor=predictor,
            spark=spark,
            config=eval_config,
            git_commit=_git_commit(),
            dataset_sha256=_file_sha256(Path(eval_config.test_split)),
            prompt_sha256=prompt_used.sha256,
        )
    finally:
        spark.stop()

    results_dir = Path(eval_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{model_choice}.json"
    results_path.write_text(summary.model_dump_json(indent=2))
    logger.info(
        "model=%s execution_accuracy=%.3f exact_match=%.3f valid_sql=%.3f -> %s",
        model_choice,
        summary.overall.execution_accuracy,
        summary.overall.exact_match_rate,
        summary.overall.valid_sql_rate,
        results_path,
    )

    init_mlflow(base_config.mlflow)
    with mlflow.start_run(run_name=f"eval-{model_choice}"):
        mlflow.log_params(
            {
                "model_name": model_choice,
                "git_commit": summary.git_commit,
                "dataset_sha256": summary.dataset_sha256,
                "prompt_sha256": summary.prompt_sha256,
                "k_shot": eval_config.k_shot,
                "max_tokens": eval_config.max_tokens,
            }
        )
        mlflow.log_metrics(
            {
                "execution_accuracy": summary.overall.execution_accuracy,
                "exact_match_rate": summary.overall.exact_match_rate,
                "valid_sql_rate": summary.overall.valid_sql_rate,
                "latency_p50_ms": summary.latency_p50_ms,
                "latency_p95_ms": summary.latency_p95_ms,
            }
        )
        mlflow.log_artifact(str(results_path))


if __name__ == "__main__":
    main()
