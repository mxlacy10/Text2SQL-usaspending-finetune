"""CLI entry point for QLoRA fine-tuning via mlx-lm on Apple Silicon.

See src/t2s/train/config.py for why mlx-lm (CUDA-only Unsloth/bitsandbytes
is not available on this machine) and config/train.yaml for hyperparameters.

Usage:
  uv run python scripts/train.py --config config/train.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import types
from pathlib import Path

import mlflow
import mlx_lm.lora as mlx_lora
from mlx_lm.tuner.callbacks import TrainingCallback

from t2s.common.config import load_base_config
from t2s.common.mlflow_utils import init_mlflow
from t2s.train.config import TrainConfig, load_train_config
from t2s.train.format_data import convert_splits_for_mlx

logger = logging.getLogger(__name__)


class MLflowTrainingCallback(TrainingCallback):
    """Forwards mlx-lm's train/val loss reports to the active MLflow run."""

    def on_train_loss_report(self, train_info: dict[str, object]) -> None:
        step = train_info.get("iteration")
        metrics = {
            f"train_{k}": float(v)
            for k, v in train_info.items()
            if k != "iteration" and isinstance(v, int | float)
        }
        mlflow.log_metrics(metrics, step=step if isinstance(step, int) else None)

    def on_val_loss_report(self, val_info: dict[str, object]) -> None:
        step = val_info.get("iteration")
        metrics = {
            f"val_{k}": float(v)
            for k, v in val_info.items()
            if k != "iteration" and isinstance(v, int | float)
        }
        mlflow.log_metrics(metrics, step=step if isinstance(step, int) else None)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_mlx_lora_args(config: TrainConfig) -> types.SimpleNamespace:
    """Build the argparse-style namespace mlx_lm.lora.run() expects, merging
    our config.train.yaml on top of mlx-lm's own defaults."""
    args = dict(mlx_lora.CONFIG_DEFAULTS)
    args.update(
        model=config.mlx_model,
        train=True,
        fine_tune_type=config.fine_tune_type,
        data=config.mlx_data_dir,
        seed=config.seed,
        num_layers=config.lora_layers,
        batch_size=config.batch_size,
        iters=config.iters,
        learning_rate=config.learning_rate,
        adapter_path=config.adapter_path,
        mask_prompt=config.mask_prompt,
        test=True,
    )
    return types.SimpleNamespace(**args)


def run_lora_training(args: types.SimpleNamespace, training_callback: TrainingCallback) -> None:
    """Load the model/data and train, mirroring mlx_lm.lora.run() but using our
    own training_callback directly (run() unconditionally overwrites whatever
    callback it's given via get_reporting_callbacks(args.report_to, ...)).

    Also logs final test-set loss/perplexity to MLflow, which
    mlx_lora.evaluate_model() only prints rather than returns."""
    import math

    import numpy as np

    np.random.seed(args.seed)
    # mlx_lm.lora re-exports these from other modules without `__all__`, so
    # mypy doesn't see them as part of its public surface; the calls are
    # exactly what mlx_lm.lora.run() itself does internally.
    model, tokenizer = mlx_lora.load(  # type: ignore[attr-defined,misc]
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )
    train_set, valid_set, test_set = mlx_lora.load_dataset(args, tokenizer)  # type: ignore[attr-defined,arg-type]
    mlx_lora.train_model(args, model, train_set, valid_set, training_callback)
    if args.test:
        test_loss = mlx_lora.evaluate(  # type: ignore[attr-defined]
            model=model,
            dataset=mlx_lora.CacheDataset(test_set),  # type: ignore[attr-defined]
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
        )
        test_ppl = math.exp(test_loss)
        logger.info("Test loss %.3f, test ppl %.3f", test_loss, test_ppl)
        mlflow.log_metrics({"test_loss": float(test_loss), "test_ppl": float(test_ppl)})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/train.yaml", help="Path to a train config YAML file."
    )
    parser.add_argument(
        "--base-config", default="config/base.yaml", help="Path to the base config YAML file."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    train_config = load_train_config(args.config)
    base_config = load_base_config(args.base_config)

    prompt, counts = convert_splits_for_mlx(
        Path(train_config.splits_dir), Path(train_config.mlx_data_dir)
    )
    logger.info("Converted splits to mlx-lm chat format: %s", counts)

    init_mlflow(base_config.mlflow)

    train_jsonl = Path(train_config.splits_dir) / "train.jsonl"
    with mlflow.start_run(run_name="train-qlora-mlx"):
        mlflow.log_params(
            {
                "mlx_model": train_config.mlx_model,
                "upstream_base_model": train_config.upstream_base_model,
                "fine_tune_type": train_config.fine_tune_type,
                "iters": train_config.iters,
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
                "lora_layers": train_config.lora_layers,
                "mask_prompt": train_config.mask_prompt,
                "seed": train_config.seed,
                "system_prompt_name": prompt.name,
                "system_prompt_sha256": prompt.sha256,
                "dataset_sha256": _file_sha256(train_jsonl),
                "git_commit": _git_commit(),
            }
        )
        mlx_args = build_mlx_lora_args(train_config)
        run_lora_training(mlx_args, training_callback=MLflowTrainingCallback())
        mlflow.log_artifact(train_config.adapter_path)
    logger.info("Adapter saved to %s", train_config.adapter_path)


if __name__ == "__main__":
    main()
