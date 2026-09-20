# Architecture — PayGuard

High-level system design. Update this as decisions solidify; log the *why* behind each choice in `docs/DECISIONS.md`.

## 1. Data flow (end to end)

```
                 ┌────────────┐
 Transactions →  │   Kafka    │  (streaming ingestion)
                 └─────┬──────┘
                       │
              ┌────────▼─────────┐
              │  Feature builder  │  (real-time + batch features)
              └────────┬─────────┘
                       │
             ┌─────────▼──────────┐
             │  Feature store      │  (Feast + Redis, online/offline split)
             └─────────┬──────────┘
                       │
        ┌──────────────▼───────────────┐
        │   Inference service            │  (Triton / Seldon, ONNX/TFLite export)
        │   Model: XGBoost / LightGBM     │
        │   + Isolation Forest (anomaly)  │
        │   + GNN (stretch — relational)  │
        └──────────────┬───────────────┘
                       │
          ┌────────────▼─────────────┐
          │  Decision: allow / flag /  │
          │  block + explanation       │
          └────────────┬─────────────┘
                       │
        ┌──────────────▼───────────────┐
        │  Monitoring (Evidently/WhyLabs) │ → drift/perf alerts
        └──────────────┬───────────────┘
                       │
          ┌────────────▼─────────────┐
          │  Airflow: retraining DAG  │ → new model version → MLflow registry
          └───────────────────────────┘
```

## 2. Components

| Layer | Choice | Why |
|---|---|---|
| Streaming/orchestration | Kafka, Airflow | Kafka for real-time transaction events; Airflow to orchestrate scheduled/triggered retraining. |
| Feature store | Feast + Redis | Feast for feature definitions/versioning; Redis for low-latency online serving. |
| Experiment tracking | MLflow | Every training run logged — params, metrics, artifacts, model registry. |
| Model serving | Triton / Seldon, ONNX / TFLite | Framework-agnostic, low-latency inference; ONNX export decouples training framework from serving. |
| Models | XGBoost, LightGBM, Isolation Forest, GNN (stretch), BERT embeddings (if text/metadata features used) | Gradient-boosted trees for tabular fraud signal; Isolation Forest for unsupervised anomaly detection; GNN if transaction-graph relationships are modeled; SMOTE to address class imbalance. |
| Monitoring | Evidently, WhyLabs | Data/prediction drift detection, triggers retraining. |
| Dataset | IEEE-CIS Fraud | Public benchmark, real fraud patterns, avoids AgroScan-style data-acquisition risk. |

**Model status (2026-09-20):** time-series CV benchmark of 7 models ranks LightGBM first (PR-AUC 0.562) ahead of XGBoost (0.498); untuned, so provisional. See `docs/BENCHMARK.md`.

## 3. Key design questions (open)

- **Real-time feature store design** — what features need online (sub-100ms) computation vs. can be precomputed offline? How is feature freshness guaranteed at inference time? *Draft answer in `docs/FEATURE_STORE_DESIGN.md`.*
- **Retraining pipeline** — trigger conditions (scheduled vs. drift-triggered), how new labels (confirmed fraud/not-fraud) flow back in, rollback strategy if a new model regresses.

## 4. Metrics that matter

- **Model:** precision-recall AUC (fraud is rare — accuracy is meaningless), precision@k for the review queue.
- **System:** inference latency (p50/p99), throughput under load.
- **MLOps:** time from drift detection → retrained model in production; experiment reproducibility via MLflow.

## 5. Mapping to repo structure

| Component | Code lives in |
|---|---|
| Kafka producers/consumers | `src/ingestion/` |
| Feature engineering + Feast integration | `src/features/` |
| Model training (XGBoost/LightGBM/Isolation Forest/GNN) | `src/models/` |
| Inference service, ONNX export, Triton/Seldon config | `src/serving/` |
| Drift/performance monitoring | `src/monitoring/` |
| Airflow DAGs (training, retraining) | `pipelines/` |
| Docker/compose, deployment manifests | `infra/` |

## 6. Deployment target: AWS, developed local-first

Build and test locally, run in Docker, then deploy to AWS. Local and AWS equivalents:

| Component | Local (now) | AWS (later) |
|---|---|---|
| Data / artifacts | `data/`, local folders | S3 |
| MLflow backend | `mlflow.db` (SQLite) | RDS (metadata) + S3 (artifacts) |
| Streaming | Kafka in docker-compose | MSK or Kinesis |
| Feature store online | Redis in docker-compose | ElastiCache |
| Orchestration | Airflow in docker-compose | MWAA |
| Serving | container via docker-compose | ECS/EKS (or SageMaker) |
| CI/CD | GitHub Actions (tests) | GitHub Actions deploying with IAM roles |

Portability habits (follow from day one so the move is small, not a rewrite):

1. **Config over hard-coding** — paths, hostnames, and settings live in `configs/` or environment variables, never inside code. Local folder vs `s3://` bucket becomes a config change.
2. **MLflow location is configurable** — tracking URI and artifact location come from config/env, so `mlflow.db` can later become RDS + S3.
3. **Containerize** — everything must be runnable in Docker (`infra/`), so the same image runs locally and on AWS (ECS/EKS).
4. **Secrets never in code or git** — credentials go in environment variables / `.env` (gitignored); on AWS use IAM roles.
5. **Keep cloud-specific calls behind small wrappers** — e.g. one storage helper, so swapping local disk for S3 touches one file.
