from __future__ import annotations
import requests
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from shared.data import example_payload
BASE="http://127.0.0.1:8000"
def main():
    payload=example_payload()
    out=requests.post(f"{BASE}/predict", json=payload, timeout=10).json()
    print("✅ predict:", out)
    requests.post(f"{BASE}/feedback", json={"features":payload,"prediction":out["prediction"],"true_label":None}, timeout=10)
    print("ℹ️ health:", requests.get(f"{BASE}/health", timeout=10).json())
if __name__=="__main__":
    main()
