from pathlib import Path

from t2s.train.config import load_train_config


def test_load_train_config(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
seed: 42
mlx_model: mlx-community/Qwen2.5-3B-Instruct-4bit
upstream_base_model: Qwen/Qwen2.5-3B-Instruct
splits_dir: data/splits
mlx_data_dir: data/mlx_train
adapter_path: adapters/test
iters: 10
batch_size: 2
learning_rate: 1.0e-5
lora_layers: 8
"""
    )
    config = load_train_config(config_path)
    assert config.seed == 42
    assert config.mlx_model == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert config.mask_prompt is True
    assert config.fine_tune_type == "lora"
