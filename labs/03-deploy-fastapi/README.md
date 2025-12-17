# Lab 03 — Déploiement API (FastAPI)

## Pré-requis
Le modèle `IrisClassifier` doit être en **Production** (Lab 02).

## Lancer l’API
```bash
cd labs/03-deploy-fastapi
uvicorn app:app --reload --port 8000
```

## Tester
```bash
python client_test.py
```

Swagger: http://127.0.0.1:8000/docs
