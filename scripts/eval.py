"""CLI entry point for the eval harness: runs one of the four AGENTS.md
baselines against the frozen test set and writes results/<model>.json
(+ logs to MLflow).

Baselines (--model):
  base_zero_shot   base mlx model, zero-shot, no adapter
  base_few_shot    base mlx model, few-shot (k=5) + schema-RAG, no adapter
  frontier         frontier model (Azure AI Foundry), few-shot + schema-RAG
  dspy_frontier    same frontier model via DSPy with BootstrapFewShot-selected demos
                   (run scripts/dspy_optimize.py first)
  finetuned        base mlx model + LoRA adapter, zero-shot
  genie            Databricks Genie space (GENIE_SPACE_ID); requires --spark-mode databricks

Usage:
  uv run python scripts/eval.py --model finetuned --config config/eval.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
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
from t2s.eval.models import ExecutionBackend
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

ModelChoice = Literal[
    "base_zero_shot",
    "base_few_shot",
    "frontier",
    "dspy_frontier",
    "dspy_frontier_compact",
    "finetuned",
    "genie",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_genie() -> tuple[Predictor, dict[str, str | None], str | None]:
    """Genie predictor plus provenance: space/warehouse IDs, date, instructions hash."""
    from databricks.sdk import WorkspaceClient

    from t2s.eval.predictors import GeniePredictor

    space_id = os.environ.get("GENIE_SPACE_ID")
    if not space_id:
        raise RuntimeError("GENIE_SPACE_ID is not set (environment or .env).")
    w = WorkspaceClient()
    space = w.genie.get_space(space_id, include_serialized_space=True)
    serialized = json.loads(space.serialized_space) if space.serialized_space else {}
    instructions = serialized.get("instructions")
    metadata: dict[str, str | None] = {
        "genie_space_id": space_id,
        "warehouse_id": space.warehouse_id,
        "run_date": datetime.date.today().isoformat(),
        "genie_tables": ", ".join(
            t["identifier"] for t in serialized.get("data_sources", {}).get("tables", [])
        ),
    }
    if instructions:
        digest = hashlib.sha256(json.dumps(instructions, sort_keys=True).encode()).hexdigest()
    else:
        digest = None
        metadata["prompt_sha256_note"] = (
            "null: the exported Genie space has no instructions to hash"
        )
    return GeniePredictor(w.genie, space_id), metadata, digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "base_zero_shot",
            "base_few_shot",
            "frontier",
            "dspy_frontier",
            "dspy_frontier_compact",
            "finetuned",
            "genie",
        ],
    )
    parser.add_argument("--config", default="config/eval.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument("--train-config", default="config/train.yaml")
    parser.add_argument(
        "--spark-mode",
        choices=["local", "databricks"],
        default=None,
        help="Override spark.mode from the base config (never chosen implicitly).",
    )
    parser.add_argument(
        "--self-correct",
        type=int,
        default=0,
        metavar="N",
        help="Wrap the predictor in an execute-and-repair loop with up to N retries "
        "(finetuned/frontier only). Results are saved as <model>_selfcorrect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    model_choice: ModelChoice = args.model

    eval_config = load_eval_config(args.config)
    base_config = load_base_config(args.base_config)
    if args.spark_mode is not None:
        base_config.spark.mode = args.spark_mode
    backend: ExecutionBackend = (
        "databricks_uc" if base_config.spark.mode == "databricks" else "local_spark"
    )
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

    run_metadata: dict[str, str | None] | None = None
    prompt_sha256: str | None = None
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
    elif model_choice == "genie":
        if backend != "databricks_uc":
            raise RuntimeError("The genie baseline requires --spark-mode databricks.")
        predictor, run_metadata, prompt_sha256 = _build_genie()
    elif model_choice in ("dspy_frontier", "dspy_frontier_compact"):
        compact = model_choice == "dspy_frontier_compact"
        if (
            base_config.models.frontier is None
            or base_config.models.frontier_azure_endpoint is None
        ):
            raise RuntimeError("config.models.frontier and frontier_azure_endpoint must be set.")
        import dspy

        from t2s.eval.dspy_baseline import DspyPredictor, load_program, make_lm

        program_path = Path(f"results/{model_choice}_program.json")
        if not program_path.exists():
            raise RuntimeError("Run scripts/dspy_optimize.py first.")
        instructions = load_prompt("dspy_sql_instructions_v1.md")
        program = load_program(instructions.text, program_path, schema_card if compact else None)
        predictor = DspyPredictor(
            program,
            make_lm(base_config.models.frontier, base_config.models.frontier_azure_endpoint),
            schema_card,
            static_schema=compact,
        )
        program_sha = _file_sha256(program_path)
        prompt_sha256 = program_sha
        run_metadata = {
            "framework": f"dspy {dspy.__version__}",
            "optimizer": "BootstrapFewShot",
            "schema_in_demos": str(not compact),
            "program": str(program_path),
            "instructions_sha256": instructions.sha256,
            "model": base_config.models.frontier,
            "run_date": datetime.date.today().isoformat(),
        }
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

    run_name: str = model_choice
    if args.self_correct:
        if model_choice not in ("finetuned", "frontier"):
            raise RuntimeError("--self-correct supports only finetuned and frontier.")
        run_name = f"{model_choice}_selfcorrect"
    repair_prompt = load_prompt("t2s_sql_repair_v1.md")

    spark = get_spark_session(base_config)
    try:
        register_silver_tables(spark, base_config)
        if args.self_correct:
            from t2s.eval.execute import execute_sql
            from t2s.eval.self_correct import RepairablePredictor, SelfCorrectingPredictor

            def _validate(sql: str) -> str | None:
                return execute_sql(spark, sql).error

            assert isinstance(predictor, (MlxZeroShotPredictor, FrontierFewShotPredictor))
            base: RepairablePredictor = predictor
            predictor = SelfCorrectingPredictor(
                base, _validate, repair_prompt.text, max_repairs=args.self_correct
            )
            run_metadata = {
                **(run_metadata or {}),
                "self_correct_max_repairs": str(args.self_correct),
                "repair_prompt_sha256": repair_prompt.sha256,
            }
        summary = run_eval(
            model_name=run_name,
            examples=test_examples,
            predictor=predictor,
            spark=spark,
            config=eval_config,
            git_commit=_git_commit(),
            dataset_sha256=_file_sha256(Path(eval_config.test_split)),
            prompt_sha256=prompt_sha256
            if model_choice in ("genie", "dspy_frontier", "dspy_frontier_compact")
            else prompt_used.sha256,
            run_metadata=run_metadata,
            execution_backend=backend,
        )
    finally:
        spark.stop()

    results_dir = Path(eval_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # UC runs get a suffix so they don't overwrite the local results.
    suffix = "_databricks_uc" if backend == "databricks_uc" and model_choice != "genie" else ""
    results_path = results_dir / f"{run_name}{suffix}.json"
    results_path.write_text(summary.model_dump_json(indent=2))
    logger.info(
        "model=%s execution_accuracy=%.3f exact_match=%.3f valid_sql=%.3f -> %s",
        run_name,
        summary.overall.execution_accuracy,
        summary.overall.exact_match_rate,
        summary.overall.valid_sql_rate,
        results_path,
    )

    init_mlflow(base_config.mlflow)
    with mlflow.start_run(run_name=f"eval-{run_name}"):
        mlflow.log_params(
            {
                "model_name": run_name,
                "execution_backend": backend,
                **{k: v for k, v in (run_metadata or {}).items() if v is not None},
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
