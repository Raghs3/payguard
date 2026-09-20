# PayGuard — Real-Time Payment Fraud Detection

MLOps course project (AI3206C) — sponsor-assigned problem statement, delivered end-to-end from architecture through a deployable fraud-detection pipeline.

> Status: architecture finalized, PayGuard selected as the team's problem statement. See `PROGRESS.md` for current status and next steps.

## Team

4 members.

## What this is

A real-time transaction fraud-scoring system: transactions stream in, get scored against a trained model within a low-latency budget, and flagged/prioritized cases feed back into monitoring and retraining. Full problem framing lives in [`docs/PRD.md`](docs/PRD.md); the system design lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repo layout

```
.
├── docs/                 # PRD, architecture
├── data/
│   ├── raw/              # untouched source data (gitignored — never committed)
│   └── processed/        # cleaned/feature-engineered data (gitignored)
├── notebooks/            # exploratory analysis, not production code
├── src/
│   ├── ingestion/        # streaming/batch data ingestion (Kafka producers/consumers)
│   ├── features/         # feature engineering + feature store integration (Feast)
│   ├── models/           # training scripts, model definitions (XGBoost/LightGBM/GNN/Isolation Forest)
│   ├── serving/          # inference service (Triton/Seldon, ONNX export)
│   └── monitoring/       # drift/performance monitoring (Evidently/WhyLabs)
├── pipelines/            # Airflow DAGs (training + retraining orchestration)
├── configs/              # config files (yaml/json) per environment
├── tests/                # unit/integration tests
├── infra/                # Docker, docker-compose, k8s/deployment manifests
└── .github/workflows/    # CI (lint, test, on push/PR)
```

## Getting started

1. Clone the repo and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd MLOPs_CP
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in local values (never commit `.env`).
3. Pull/download the dataset per [`data/README.md`](data/README.md) — raw data is not committed to the repo.
4. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit together and where your task's code should live.
5. See [`PROGRESS.md`](PROGRESS.md) for current status and what to work on next.

## Branching & workflow

- `main` — always demo-able / presentable state.
- `dev` — integration branch; feature branches merge here first.
- `feature/<short-name>` — one branch per task (e.g. `feature/kafka-ingestion`, `feature/xgboost-baseline`).
- Open a PR into `dev`; at least one teammate reviews before merge. Merge `dev` → `main` before reviews/demos.

## Docs index

| Doc | Purpose |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Problem definition, scope, success metrics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, data flow, component choices |
| [`PROGRESS.md`](PROGRESS.md) | Current status, open questions, next steps |
| [`data/README.md`](data/README.md) | Dataset source, download steps, schema notes |
| [`CLAUDE.md`](CLAUDE.md) | Context/rules for AI-assisted work in this repo |
