# PRD — PayGuard: Real-Time Payment Fraud Detection

**Status:** Draft — fill in bracketed sections as the team finalizes scope.
**Course:** AI3206C (MLOps elective) | **Team size:** 4

## 1. Problem statement

Sponsor-assigned problem: detect fraudulent digital payment transactions in real time, minimizing false negatives (missed fraud) while keeping false positives low enough to avoid blocking legitimate customers.

## 2. Why this, over the alternatives

The team evaluated three sponsor-assigned problem statements before selecting PayGuard:

| Option | Verdict |
|---|---|
| **PayGuard** — real-time fraud detection | **Selected.** Accessible public data (IEEE-CIS Fraud), structurally similar to ClaimShield (transferable lessons), clear real-time latency angle that showcases MLOps skills (streaming, feature store, low-latency serving). |
| AgroScan — crop disease detection | Rejected for now — needs real field images beyond PlantVillage to hold up to scrutiny; higher data-acquisition risk. |
| ClaimShield — insurance claim fraud screening | Rejected — similar structure to PayGuard but less real-time/streaming emphasis. |

## 3. Users & use case

- **Primary user:** a payments platform's fraud/risk team.
- **Use case:** each transaction is scored the moment it happens; high-risk transactions are flagged/blocked or queued for review; the system explains *why* a transaction was flagged.

## 4. Scope

**In scope (v1):**
- [ ] Batch + streaming ingestion of transaction data (Kafka)
- [ ] Feature engineering + online feature store (Feast/Redis)
- [ ] Baseline + advanced fraud model (XGBoost/LightGBM, Isolation Forest for anomaly signal, GNN as stretch goal)
- [ ] Low-latency inference service (Triton/Seldon, ONNX export)
- [ ] Drift/performance monitoring (Evidently/WhyLabs)
- [ ] Retraining pipeline (Airflow-orchestrated)
- [ ] Cloud deployment on AWS (developed local-first, see section 5b)

**Out of scope (v1):**
- [ ] Production-grade auth/user management
- [ ] Multi-region deployment
- [ ] [Add anything the team explicitly decides to cut]

## 5. Data

- **Dataset:** IEEE-CIS Fraud Detection (public benchmark) — see `data/README.md` for access.
- **Class imbalance handling:** SMOTE (or equivalent) during training.
- **Data risk:** low — public benchmark, unlike AgroScan's field-image dependency.

## 5b. Deployment target

- **Goal:** actually deploy on **AWS** (cloud MLOps), not only run locally.
- **Approach:** develop local-first (venv, then Docker/docker-compose), then move to AWS. Moving early would slow the modeling work; moving later stays cheap if the portability habits in `docs/ARCHITECTURE.md` section 6 are followed from day one.
- **Cost guardrail:** use free tier/credits, set a billing alert before creating anything, shut resources down when idle.
- **Status:** decided 2026-09-20. Which AWS services (e.g. ECS vs EKS vs SageMaker) is still open.

## 6. Success metrics

- **Model quality:** precision-recall AUC, precision@k (top-k flagged transactions).
- **Latency:** [target — team workflow suggests < 200ms end-to-end; PRD example was p99 < 100ms; decide] — to be set once the serving stack is benchmarked.
- **Operational:** monitored drift with automated retraining trigger; MLflow-tracked experiment lineage for every model in production.

## 7. Milestones

| Milestone | Target date | Owner |
|---|---|---|
| Architecture pitch (all 3 PS) presented to sponsor board | Done | Team |
| PayGuard finalized as chosen problem statement | Done | Team |
| Data pipeline + baseline model | [date] | [name] |
| Feature store + retraining pipeline design | [date] | [name] |
| Midsem review presentation | [date] | Team |
| Serving + monitoring live end-to-end | [date] | [name] |
| Final demo/delivery | [date] | Team |

## 8. Open questions

- Real-time feature store design for PayGuard — flagged for deeper exploration.
- Retraining pipeline design for PayGuard (in addition to ClaimShield's) — flagged for deeper exploration.
- AWS service choices (serving, orchestration, Kafka vs Kinesis) — decide when the local Docker setup works.
- [Add as they come up — move resolved ones to `docs/DECISIONS.md`]
