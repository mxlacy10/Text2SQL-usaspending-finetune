"""Shared MLflow setup for run scripts (train, eval).

Newer MLflow server versions put the local filesystem tracking backend (e.g.
"./mlruns", which AGENTS.md documents as the default local tracking_uri) into
maintenance mode and require an explicit opt-out. This keeps the documented
local-by-default workflow working without changing tracking_uri semantics.
"""

from __future__ import annotations

import os

import mlflow

from t2s.common.config import MLflowSettings


def init_mlflow(settings: MLflowSettings) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment)
