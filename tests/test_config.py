from t2s.common.config import load_base_config


def test_load_base_config() -> None:
    config = load_base_config("config/base.yaml")
    assert config.seed == 42
    assert config.schema_name == "usaspending"
    assert config.data.fiscal_years == [2022, 2023, 2024]
    assert config.data.award_types == ["contracts"]
    assert config.models.base == "Qwen/Qwen2.5-7B-Instruct"
    assert config.catalog is None
