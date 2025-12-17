from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from shared.data import make_split, FEATURE_COLUMNS
from shared.metrics import mean_drift

def read_events(p: Path) -> pd.DataFrame:
    rows=[]
    with p.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def main():
    ref=make_split().X_train.copy()
    p=Path("labs/03-deploy-fastapi/feedback/events.jsonl").resolve()
    if not p.exists():
        raise FileNotFoundError(f"{p} introuvable. Lancez l'API + generate_traffic.py.")
    cur=read_events(p)[FEATURE_COLUMNS].copy()
    drift=mean_drift(ref,cur)
    print("=== Drift score (simple) ===")
    for k,v in drift.items():
        print(f"- {k:12s}: {v:.3f} {'⚠️' if v>0.5 else ''}")
if __name__=="__main__":
    main()
