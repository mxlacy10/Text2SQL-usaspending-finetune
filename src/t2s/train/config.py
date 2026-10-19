"""Pydantic config for config/train.yaml: QLoRA fine-tuning via mlx-lm on
Apple Silicon.

See AGENTS.md stack section for why mlx-lm (not Unsloth/bitsandbytes, which
are CUDA-only): training runs locally on an Apple M1 (16GB RAM), which has no
CUDA GPU. mlx-lm supports LoRA/QLoRA fine-tuning natively on Apple Silicon via
Metal. TODO(max): if/when a CUDA GPU (e.g. Colab/Kaggle T4) becomes available,
the original Unsloth+bitsandbytes path documented in earlier AGENTS.md
revisions could be reinstated as an alternate path for the full 7B base model.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class TrainConfig(BaseModel):
    """Top-level config, loaded from config/train.yaml."""

    seed: int
    # MLX-format model repo (pre-quantized 4-bit for QLoRA). See
    # https://huggingface.co/mlx-community for the mlx-format mirror of a
    # given Hugging Face base model.
    mlx_model: str
    # Human-readable record of the original (non-MLX) upstream base model,
    # for documentation/reproducibility (AGENTS.md rule 6).
    upstream_base_model: str
    splits_dir: str
    mlx_data_dir: str
    adapter_path: str
    iters: int
    batch_size: int
    learning_rate: float
    lora_layers: int
    mask_prompt: bool = True
    fine_tune_type: str = "lora"


def load_train_config(path: str | Path = "config/train.yaml") -> TrainConfig:
    """Load and validate config/train.yaml (or an equivalent file)."""
    raw = yaml.safe_load(Path(path).read_text())
    return TrainConfig.model_validate(raw)
