# Feature store design (draft)

Answers the open question in `docs/ARCHITECTURE.md` section 3: which features must be
computed online (sub-100ms) and which can be precomputed offline, and how freshness is
guaranteed. **Draft — several points below are assumptions to verify in EDA.**

## 1. Latency budget

Target end-to-end scoring: under 200 ms (team workflow) and ideally under 100 ms (PRD example). Rough split:

| Step | Budget |
|---|---|
| Request parsing + validation | ~5 ms |
| Feature lookup (Redis) | ~10 ms |
| Model inference | ~20 ms |
| Decision rules + response | ~10 ms |
| Headroom (network, queueing) | rest |

Rule: at scoring time we only do cheap key lookups. Anything expensive is computed before the request arrives.

## 2. Three kinds of features

| Kind | What it is | Where it comes from | Examples |
|---|---|---|---|
| **Request-time** | Values in the transaction itself | The incoming event, no lookup | amount, ProductCD, card type, address, email domain, device info |
| **Online (streaming)** | Recent behaviour that changes every few seconds | Computed from the Kafka stream, kept in Redis | transactions per card in last 1h / 24h, amount sum in last 1h, distinct addresses/devices in last 24h, seconds since last transaction |
| **Offline (batch)** | Long-term history that changes slowly | Recomputed on a schedule (e.g. daily), loaded into Redis | 30-day average/std of amount per card, typical hour of day, historical share of fraud per email domain |

Why the split: velocity features are the strongest fraud signals but go stale in minutes, so they need the stream. Long-term aggregates are expensive to compute and stay valid for hours or days, so a nightly batch is enough.

**Dataset notes to verify in EDA:**
- IEEE-CIS has no clean card/customer ID. We likely need a derived entity key (e.g. built from `card1`, `addr1`, and a day-offset column). Its quality decides how useful velocity features are.
- The `C*` (counts), `D*` (time deltas) and `V*` columns are already engineered by the data provider. We should check how much our own velocity features add on top.
- Fraud-rate style features (target encoding) can leak the label. They must be computed only from data before each transaction's time.

## 3. Freshness

- Scoring order per event: **read features → score → then write the update.** A transaction must not see itself in its own velocity features.
- Every stored feature carries a timestamp. The serving code checks age and falls back to a default when a value is missing or older than its limit (e.g. online features > 5 min old).
- Redis keys get a TTL a bit longer than the feature window, so dead cards don't pile up.
- Use event time (when the transaction happened), not processing time, so a delayed Kafka message doesn't corrupt windows.
- New/unknown entities (cold start) get defaults plus a flag feature, `is_new_entity`, so the model can learn to treat them differently.

## 4. Training/serving consistency

The biggest production risk is train/serve skew: features computed one way for training and another way live.

- Define every feature once in Feast (`src/features/`) and use the same definition offline and online.
- Build training sets with Feast point-in-time joins (`get_historical_features`), so each row only sees feature values that existed at its transaction time.
- The time-based train/test split already used in `src/models/` matches this.
- Add a test that computes the same feature via the batch path and the streaming path on a sample and checks they match.

## 5. Local now, AWS later

| Piece | Local | AWS |
|---|---|---|
| Online store | Redis in docker-compose | ElastiCache |
| Offline store | Parquet files in `data/processed/` | S3 (Parquet) |
| Stream | Kafka in docker-compose | MSK / Kinesis |
| Feast registry | local file | S3 or RDS |

Store locations come from `configs/` (portability habits, `docs/ARCHITECTURE.md` section 6).

## 6. Suggested build order

1. EDA: confirm the entity key and the time column semantics (`TransactionDT` is seconds from a reference, not a date).
2. Batch-only: compute velocity and aggregate features offline, retrain, and measure PR-AUC gain. If features don't help, the streaming layer isn't worth building.
3. Define the same features in Feast; materialise to local Redis.
4. Replay the dataset through a Kafka producer to update online features; measure lookup latency.
5. Add freshness checks and the batch-vs-stream consistency test.

Step 2 first is deliberate: it proves the features matter before we build streaming infrastructure.

## 7. Open decisions

- Entity key definition (needs EDA).
- Window sizes (1h/24h/7d?) — tune with the batch experiment.
- Staleness limits per feature.
- Feast vs a plain Redis wrapper if Feast setup proves too heavy for the timeline.
