from __future__ import annotations
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME="ModelDeployment-Lab"
MODEL_NAME="IrisClassifier"

def main():
    client=MlflowClient()
    exp=client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError("Experiment introuvable. Lancez Lab 01.")
    runs=client.search_runs([exp.experiment_id], order_by=["metrics.accuracy DESC"], max_results=1)
    if not runs:
        raise RuntimeError("Aucun run trouvé. Lancez Lab 01.")
    best=runs[0]; run_id=best.info.run_id
    print(f"🏆 Best run: {run_id} (accuracy={best.data.metrics.get('accuracy')})")
    model_uri=f"runs:/{run_id}/model"
    res=mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"✅ Registered: {MODEL_NAME} v{res.version}")
    print("➡️ Ouvrez MLflow UI → Models pour promouvoir en Staging/Production.")
if __name__=="__main__":
    main()
