import sys
import types
from typing import Any

import pytest
from pydantic import ValidationError

from t2s.common import spark as spark_mod
from t2s.common.config import BaseConfig, load_base_config
from t2s.synth.execute import register_silver_tables


def _config(mode: str = "local", catalog: str | None = None) -> BaseConfig:
    cfg = load_base_config("config/base.yaml").model_dump(by_alias=False)
    cfg["spark"]["mode"] = mode
    cfg["catalog"] = catalog
    return BaseConfig.model_validate(cfg)


def test_default_mode_is_local() -> None:
    assert load_base_config("config/base.yaml").spark.mode == "local"


def test_databricks_mode_requires_catalog() -> None:
    with pytest.raises(ValidationError, match="requires `catalog`"):
        _config("databricks", None)


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        _config("spark-connect")


class _FakeSession:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def sql(self, statement: str) -> None:
        self.statements.append(statement)


def _install_fake_connect(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    class _Builder:
        def getOrCreate(self) -> _FakeSession:  # noqa: N802 - mirrors Databricks API
            return session

    class _DatabricksSession:
        builder = _Builder()

    connect = types.ModuleType("databricks.connect")
    connect.DatabricksSession = _DatabricksSession  # type: ignore[attr-defined]
    pkg = types.ModuleType("databricks")
    pkg.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databricks", pkg)
    monkeypatch.setitem(sys.modules, "databricks.connect", connect)


def test_databricks_mode_sets_catalog_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    _install_fake_connect(monkeypatch, session)
    result = spark_mod.get_spark_session(_config("databricks", "workspace"))
    assert result is session
    assert session.statements == ["USE CATALOG `workspace`", "USE SCHEMA `usaspending`"]


def test_local_mode_does_not_touch_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(spark_mod, "_get_local_session", lambda cfg: sentinel)
    monkeypatch.setattr(
        spark_mod, "_get_databricks_session", lambda cfg: pytest.fail("used databricks")
    )
    assert spark_mod.get_spark_session(_config("local")) is sentinel


def test_register_silver_tables_noop_in_databricks_mode() -> None:
    class _Spark:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"spark.{name} must not be used in databricks mode")

    register_silver_tables(_Spark(), _config("databricks", "workspace"))  # type: ignore[arg-type]
