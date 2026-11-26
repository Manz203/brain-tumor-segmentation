"""Optional local SQLite-backed MLflow experiment tracking and model registry."""
from pathlib import Path
import json
import pandas as pd
from .model import load_checkpoint

def configure(out, uri=None):
    import mlflow
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    uri = uri or "sqlite:///" + (out/"mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name("NeuroSeg V2") is None:
        mlflow.create_experiment("NeuroSeg V2",artifact_location=(out/"mlflow_artifacts").as_uri())
    mlflow.set_experiment("NeuroSeg V2")
    return mlflow

def log_candidate(mlflow, directory, params):
    directory = Path(directory)
    history = pd.read_csv(directory/"training_history.csv")
    with mlflow.start_run(run_name=directory.name) as run:
        mlflow.log_params(params)
        for row in history.to_dict("records"):
            mlflow.log_metrics({k: float(v) for k,v in row.items() if k != "epoch"}, step=int(row["epoch"]))
        for name in ("training_history.csv","training_curves.png"):
            mlflow.log_artifact(str(directory/name))
        mlflow.log_artifact(str(directory/"best_unet.pth"),artifact_path="checkpoint")
        return run.info.run_id

def register_champion(mlflow, checkpoint, out, summary):
    import mlflow.pytorch
    from mlflow import MlflowClient
    model, ck = load_checkpoint(checkpoint, "cpu")
    with mlflow.start_run(run_name="selected-model"):
        mlflow.log_params(ck["config"])
        mlflow.log_metrics({"test_patient_"+k: float(v) for k,v in summary["patient_macro"].items()})
        mlflow.log_artifact(str(Path(out)/"test_metrics.json"))
        mlflow.log_artifact(str(checkpoint),artifact_path="checkpoint")
        import torch
        # An explicit example avoids guessing model input dimensions.
        import numpy as np
        example = np.zeros((1,1,ck["config"]["size"],ck["config"]["size"]),dtype=np.float32)
        info = mlflow.pytorch.log_model(model, name="segmentation_model", input_example=example,
                                        code_paths=[str(Path(__file__).parent)],
                                        pip_requirements=["torch","numpy","mlflow>=3,<4"])
        version = mlflow.register_model(info.model_uri,"NeuroSeg").version
        MlflowClient().set_registered_model_alias("NeuroSeg","champion",str(version))
        result={"model":"NeuroSeg","version":str(version),"alias":"champion",
                "architecture":ck["config"].get("architecture","unet"),"run_id":info.run_id}
    (Path(out)/"registry.json").write_text(json.dumps(result,indent=2))
    return result
