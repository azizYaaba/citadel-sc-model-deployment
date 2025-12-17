from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    max_ms: float

def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    arr = np.array(latencies_ms, dtype=float)
    return LatencyStats(
        p50_ms=float(np.percentile(arr,50)),
        p95_ms=float(np.percentile(arr,95)),
        max_ms=float(arr.max()),
    )

def mean_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str,float]:
    eps=1e-9
    out={}
    for col in reference.columns:
        mu_ref=reference[col].mean()
        std_ref=reference[col].std()+eps
        mu_cur=current[col].mean()
        out[col]=float(abs(mu_cur-mu_ref)/std_ref)
    return out
