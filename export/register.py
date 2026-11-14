import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

p = argparse.ArgumentParser()
p.add_argument("onnx")
p.add_argument("--name", default="stylelens-clip")
p.add_argument("--arch", required=True)
p.add_argument("--p5", type=float, required=True)
p.add_argument("--embed-p99-ms", type=float, required=True)
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("stylelens")
client = MlflowClient()

try:
    client.get_registered_model(a.name)
except MlflowException:
    client.create_registered_model(a.name)

with mlflow.start_run(run_name=f"export-{a.arch}") as run:
    mlflow.log_params({"arch": a.arch, "embedding_dim": 512, "runtime": "onnxruntime-cpu"})
    mlflow.log_metrics({"precision_at_5": a.p5, "embed_p99_ms": a.embed_p99_ms})
    mlflow.log_artifact(a.onnx, artifact_path="model")
    mv = client.create_model_version(
        a.name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )

client.set_registered_model_alias(a.name, "candidate", mv.version)
print(f"registered {a.name} v{mv.version} -> alias 'candidate'")