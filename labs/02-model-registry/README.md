# Lab 02 — Model Registry : enregistrer & promouvoir

## Pré-requis
Avoir des runs (Lab 01).

## Enregistrer le meilleur run (accuracy max)
```bash
python labs/02-model-registry/register_best_model.py
```

Ensuite dans MLflow UI → **Models**: promouvoir en **Staging** puis **Production**.

## Promouvoir en CLI
```bash
python labs/02-model-registry/promote_model.py --version 1 --stage Production
```
