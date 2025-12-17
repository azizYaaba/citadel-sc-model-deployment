# Lab 01 — MLflow Tracking : entraîner, logger, comparer

## Lancer
1) Terminal 1:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns_artifacts
```

2) Terminal 2:
```bash
cd labs/01-mlflow-tracking
python train_and_track.py
```

Puis ouvrir http://127.0.0.1:5000 et comparer les runs.
