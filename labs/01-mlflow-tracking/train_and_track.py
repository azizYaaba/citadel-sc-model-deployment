from __future__ import annotations
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from shared.data import make_split, FEATURE_COLUMNS

EXPERIMENT_NAME="ModelDeployment-Lab"
ARTIFACT_DIR=Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

def run_logreg(C: float, seed: int):
    ds=make_split(random_state=seed)
    m=LogisticRegression(max_iter=500, C=C)
    t0=time.time(); m.fit(ds.X_train, ds.y_train); train_ms=(time.time()-t0)*1000
    preds=m.predict(ds.X_test)
    metrics={"accuracy":accuracy_score(ds.y_test,preds),"f1_macro":f1_score(ds.y_test,preds,average="macro"),"train_ms":train_ms}
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(ds.y_test,preds,ax=ax)
    ax.set_title("Confusion Matrix — LogisticRegression")
    cm=ARTIFACT_DIR/f"cm_logreg_C{C}.png"; fig.tight_layout(); fig.savefig(cm,dpi=150); plt.close(fig)
    mp=ARTIFACT_DIR/f"logreg_C{C}.joblib"; joblib.dump(m, mp)
    return m, metrics, cm, mp

def run_rf(n_estimators: int, max_depth, seed: int):
    ds=make_split(random_state=seed)
    m=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=seed,n_jobs=-1)
    t0=time.time(); m.fit(ds.X_train, ds.y_train); train_ms=(time.time()-t0)*1000
    preds=m.predict(ds.X_test)
    metrics={"accuracy":accuracy_score(ds.y_test,preds),"f1_macro":f1_score(ds.y_test,preds,average="macro"),"train_ms":train_ms}
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(ds.y_test,preds,ax=ax)
    ax.set_title("Confusion Matrix — RandomForest")
    cm=ARTIFACT_DIR/f"cm_rf_{n_estimators}_{max_depth}.png"; fig.tight_layout(); fig.savefig(cm,dpi=150); plt.close(fig)
    mp=ARTIFACT_DIR/f"rf_{n_estimators}_{max_depth}.joblib"; joblib.dump(m, mp)
    return m, metrics, cm, mp

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs=[("logreg",{"C":0.5,"seed":42}),("logreg",{"C":1.0,"seed":42}),("rf",{"n_estimators":200,"max_depth":None,"seed":42}),("rf",{"n_estimators":200,"max_depth":5,"seed":42})]
    for algo,params in runs:
        with mlflow.start_run(run_name=f"{algo}-{params}"):
            mlflow.log_params(params)
            mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
            if algo=="logreg":
                model,metrics,cm,mp = run_logreg(C=params["C"], seed=params["seed"])
                mlflow.log_param("algo","LogisticRegression")
            else:
                model,metrics,cm,mp = run_rf(n_estimators=params["n_estimators"], max_depth=params["max_depth"], seed=params["seed"])
                mlflow.log_param("algo","RandomForestClassifier")
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm), artifact_path="plots")
            mlflow.log_artifact(str(mp), artifact_path="models_raw")
            mlflow.sklearn.log_model(model, artifact_path="model")
    print("✅ Terminé. Ouvrez MLflow UI pour comparer.")
if __name__=="__main__":
    main()
