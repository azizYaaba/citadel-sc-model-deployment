from __future__ import annotations
import os, time
from typing import Optional, Any
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from shared.data import FEATURE_COLUMNS
from shared.metrics import latency_stats

MODEL_NAME=os.getenv("MODEL_NAME","IrisClassifier")
MODEL_STAGE=os.getenv("MODEL_STAGE","Production")

mlflow.set_tracking_uri("sqlite:////Users/abdoul.bonkoungou/Downloads/citadel-sc-model-deployment/mlflow.db")
mlflow.set_registry_uri("sqlite:////Users/abdoul.bonkoungou/Downloads/citadel-sc-model-deployment/mlflow.db")

app=FastAPI(title="Iris Model API", version="1.0.0")
_model=None
_latencies_ms: list[float]=[]

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10)
    sepal_width: float = Field(..., ge=0, le=10)
    petal_length: float = Field(..., ge=0, le=10)
    petal_width: float = Field(..., ge=0, le=10)

class Feedback(BaseModel):
    features: IrisFeatures
    prediction: int
    true_label: Optional[int]=None

def load_model() -> Any:
    uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(uri)

@app.on_event("startup")
def _startup():
    global _model
    _model=load_model()

@app.get("/health")
def health():
    info={"status":"ok","model_name":MODEL_NAME,"stage":MODEL_STAGE}
    if _latencies_ms:
        s=latency_stats(_latencies_ms)
        info.update({"latency_ms_p50":s.p50_ms,"latency_ms_p95":s.p95_ms,"latency_ms_max":s.max_ms})
    return info

@app.post("/predict")
def predict(payload: IrisFeatures):
    global _latencies_ms
    t0=time.time()
    df=pd.DataFrame([[getattr(payload,c) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    pred=_model.predict(df)
    latency_ms=(time.time()-t0)*1000
    _latencies_ms=(_latencies_ms+[latency_ms])[-200:]
    return {"prediction":int(pred[0]),"latency_ms":latency_ms,"model":{"name":MODEL_NAME,"stage":MODEL_STAGE}}

@app.post("/feedback")
def feedback(item: Feedback):
    os.makedirs("feedback", exist_ok=True)
    row={**item.features.model_dump(),"prediction":item.prediction,"true_label":item.true_label,"ts":time.time()}
    with open(os.path.join("feedback","events.jsonl"),"a",encoding="utf-8") as f:
        f.write(pd.Series(row).to_json()+"\n")
    return {"saved":True}
