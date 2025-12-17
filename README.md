# Labs — Déploiement & intégration de modèles ML (MLflow + API)

Ces labs accompagnent la présentation **Déploiement et intégration de modèles de Machine Learning** (slide *Labs / Travaux pratiques*). 

Objectifs:
- Utiliser **MLflow** pour suivre une expérience
- Enregistrer et comparer plusieurs modèles
- Déployer un modèle via **API (FastAPI)**
- Observer des métriques en temps réel (monitoring basique)

## Installation
```bash
cd mlops-model-deployment-labs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Démarrer MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns_artifacts
# UI: http://127.0.0.1:5000
```

## Parcours
1. `labs/01-mlflow-tracking`
2. `labs/02-model-registry`
3. `labs/03-deploy-fastapi`
4. `labs/04-monitoring-basic`

Bon labs 🚀
