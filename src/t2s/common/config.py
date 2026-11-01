"""Pydantic models for loading config/*.yaml files.

See AGENTS.md: "pydantic models for config and for dataset records."
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

AwardType = Literal["contracts"]


class SparkSettings(BaseModel):
    app_name: str
    warehouse_dir: str
    # "local": Delta tables under warehouse_dir. "databricks": Unity Catalog via
    # Databricks Connect; requires `catalog` to be set.
    mode: Literal["local", "databricks"] = "local"


class DataSettings(BaseModel):
    raw_dir: str
    fiscal_years: list[int]
    award_types: list[AwardType]
    # Toptier awarding agency names to bound the slice. Empty = no filter
    # (all of government), which is not recommended for a local Spark setup.
    agencies: list[str] = []


class ModelsSettings(BaseModel):
    base: str
    alternate: str
    frontier: str | None = None
    # Set only when the frontier model is served via an Azure AI Foundry
    # deployment (OpenAI Responses API + Azure AD auth) rather than the
    # public OpenAI/Anthropic APIs. See src/t2s/synth/llm.py AzureFoundryClient.
    frontier_azure_endpoint: str | None = None


class MLflowSettings(BaseModel):
    tracking_uri: str
    experiment: str


class BaseConfig(BaseModel):
    """Top-level config, loaded from config/base.yaml."""

    seed: int
    spark: SparkSettings
    catalog: str | None = None
    schema_name: str
    data: DataSettings
    models: ModelsSettings
    mlflow: MLflowSettings

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _databricks_requires_catalog(self) -> BaseConfig:
        if self.spark.mode == "databricks" and not self.catalog:
            raise ValueError("spark.mode=databricks requires `catalog` to be set")
        return self


def load_base_config(path: str | Path = "config/base.yaml") -> BaseConfig:
    """Load and validate config/base.yaml (or an equivalent file)."""
    raw = yaml.safe_load(Path(path).read_text())
    # YAML key is `schema`, which shadows BaseModel.schema(); store as schema_name.
    if "schema" in raw:
        raw["schema_name"] = raw.pop("schema")
    return BaseConfig.model_validate(raw)
