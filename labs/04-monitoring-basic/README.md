# Lab 04 — Monitoring basique (latence + drift simple)

## Générer du trafic (et des feedback events)
```bash
cd labs/04-monitoring-basic
python generate_traffic.py
```

## Calculer un drift simple (moyennes)
```bash
python compute_drift.py
```

> En prod: utiliser des outils dédiés (Prometheus/Grafana, Evidently, WhyLabs, etc.).
