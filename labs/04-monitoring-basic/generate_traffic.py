from __future__ import annotations
import random, time, requests
BASE="http://127.0.0.1:8000"

def sample(step:int):
    x={
        "sepal_length": random.uniform(4.3, 7.9),
        "sepal_width": random.uniform(2.0, 4.4),
        "petal_length": random.uniform(1.0, 6.9),
        "petal_width": random.uniform(0.1, 2.5),
    }
    drift=min(step/200,1.0)
    x["petal_length"] += drift*0.8
    x["petal_width"]  += drift*0.3
    return x

def main(n:int=250):
    for i in range(n):
        payload=sample(i)
        out=requests.post(f"{BASE}/predict", json=payload, timeout=10).json()
        requests.post(f"{BASE}/feedback", json={"features":payload,"prediction":out["prediction"],"true_label":None}, timeout=10)
        if i%25==0:
            h=requests.get(f"{BASE}/health", timeout=10).json()
            print(f"[{i}] latency p50/p95:", h.get("latency_ms_p50"), h.get("latency_ms_p95"))
        time.sleep(0.02)
    print("✅ Trafic généré.")
if __name__=="__main__":
    main()
