# Lab 02 — Model Registry : enregistrer & promouvoir

## Pré-requis
Avoir des runs (Lab 01).

## Enregistrer le meilleur run (accuracy max)
```bash
cd labs/02-model-registry
python register_best_model.py
```

Ensuite dans MLflow UI → **Models**: promouvoir en **Staging** puis **Production**.

## Bonus: promouvoir en CLI
```bash
python promote_model.py --version 1 --stage Production
```
