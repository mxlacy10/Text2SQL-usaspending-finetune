"""Pydantic config for config/eval.yaml: the eval harness that runs all four
AGENTS.md baselines (base zero-shot, base few-shot/schema-RAG, frontier
few-shot, fine-tuned zero-shot) and writes results/<run_name>.json.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class EvalConfig(BaseModel):
    """Top-level config, loaded from config/eval.yaml."""

    seed: int
    test_split: str
    train_split: str
    schema_card: str
    results_dir: str
    k_shot: int
    max_tokens: int
    zero_shot_prompt: str
    few_shot_prompt: str
    frontier_input_price_per_1k_usd: float | None = None
    frontier_output_price_per_1k_usd: float | None = None


def load_eval_config(path: str | Path = "config/eval.yaml") -> EvalConfig:
    """Load and validate config/eval.yaml (or an equivalent file)."""
    raw = yaml.safe_load(Path(path).read_text())
    return EvalConfig.model_validate(raw)
