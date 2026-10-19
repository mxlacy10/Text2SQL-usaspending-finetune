from __future__ import annotations

from t2s.eval.config import load_eval_config


def test_load_eval_config_parses_defaults() -> None:
    config = load_eval_config("config/eval.yaml")
    assert config.seed == 42
    assert config.test_split == "data/splits/test.jsonl"
    assert config.k_shot == 5
    assert config.frontier_input_price_per_1k_usd is None
    assert config.frontier_output_price_per_1k_usd is None
