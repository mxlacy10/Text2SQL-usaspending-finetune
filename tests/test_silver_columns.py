from t2s.ingest.silver import CORE_COLUMNS


def test_core_columns_unique_names() -> None:
    raw_names = [spec.raw_name for spec in CORE_COLUMNS]
    silver_names = [spec.silver_name for spec in CORE_COLUMNS]
    assert len(raw_names) == len(set(raw_names)), "duplicate raw column name"
    assert len(silver_names) == len(set(silver_names)), "duplicate silver column name"


def test_core_columns_valid_spark_types() -> None:
    allowed_types = {"string", "int", "double", "date", "timestamp", "boolean"}
    for spec in CORE_COLUMNS:
        assert spec.spark_type in allowed_types, f"unexpected spark type: {spec.spark_type}"
