"""Resolve and download a registered model. Shared by the API and the indexer."""

import glob
import os
import tempfile

import mlflow
from mlflow import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "stylelens-clip")
CACHE = os.path.join(tempfile.gettempdir(), "stylelens_models")


def resolve(alias: str | None = None, version: str | None = None) -> tuple[str, str]:
    """Return (local .onnx path, version). An explicit version wins over the alias."""
    if not (alias or version):
        raise ValueError("need an alias or a version")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    mv = (
        client.get_model_version(MODEL_NAME, version)
        if version
        else client.get_model_version_by_alias(MODEL_NAME, alias)
    )
    dst = os.path.join(CACHE, f"v{mv.version}")
    os.makedirs(dst, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=dst)
    return glob.glob(os.path.join(local, "*.onnx"))[0], str(mv.version)
