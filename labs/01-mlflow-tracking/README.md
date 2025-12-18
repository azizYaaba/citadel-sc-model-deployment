# Lab 01 — MLflow Tracking : entraîner, logger, comparer

## Lancer si vous ne l'avez pas déja fait 
1) Terminal 1:
```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

2) Terminal 2:
```bash
python labs/01-mlflow-tracking/train_and_track.py
```

Puis ouvrir http://127.0.0.1:5000 et comparer les runs.
