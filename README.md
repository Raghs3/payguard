# PayGuard — Real-Time Digital Payment Fraud Detection

PayGuard is an MLOps course project (AI3206C) for fraud scoring on digital payment transactions. The implemented Build & Deploy path trains and packages a LightGBM model, serves predictions through FastAPI, and runs locally in Docker. The repository also contains architecture and design plans for streaming, feature-store, monitoring, and orchestration work; those components are not currently deployed.

## Project layout

```text
.
├── configs/                    # Final model and experiment settings
├── data/                       # Local raw/processed data; not committed
├── docs/                       # PRD, architecture, benchmark, design notes
├── infra/
│   ├── Dockerfile               # Inference image
│   └── requirements-serving.txt # Small, pinned serving runtime
├── models/                     # Local model packages; not committed
├── reports/                    # Tuning and benchmark results
├── src/
│   ├── ingestion/               # Planned ingestion placeholder
│   ├── features/                # Planned feature engineering placeholder
│   ├── models/                  # Training and model package code
│   ├── serving/                 # FastAPI application
│   └── monitoring/              # Planned monitoring placeholder
├── pipelines/                   # Planned orchestration placeholder
├── tests/                       # Synthetic model and API tests
├── .github/workflows/ci.yml     # Tests and Docker build check
├── .dockerignore                # Excludes data and model artifacts from images
└── docker-compose.yml           # Local API with a read-only model mount
```

See [`docs/PRD.md`](docs/PRD.md) for the problem and scope, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system design, [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for model comparisons, and [`docs/FEATURE_STORE_DESIGN.md`](docs/FEATURE_STORE_DESIGN.md) for the feature-store proposal.

## Deployment overview

The implemented deployment is local: a Python FastAPI service or a Docker container loads a prebuilt model package and scores one transaction per request. Docker exposes port `8000`. The container does not contain training data or the model package; Compose mounts the local model package read-only when the container starts.

There is no cloud, Kubernetes, or production deployment configured. The Kafka ingestion, Feast feature store, monitoring, and Airflow pipeline directories are plans/placeholders, not running parts of the serving path.

## Model packaging and MLflow

`src/models/train_final.py` reads configuration from [`configs/final.json`](configs/final.json), chooses the highest-PR-AUC LightGBM trial from `reports/tuning_results.csv`, trains on the older time period, and evaluates on the newest 20% holdout. It packages the fitted classifier with its preprocessing state, categorical mappings, and ordered feature names.

Prerequisites:

- Download the labeled IEEE-CIS files into `data/raw/`: `train_transaction.csv` and `train_identity.csv`. See [`data/README.md`](data/README.md).
- Keep the existing `reports/tuning_results.csv` available.
- Install the model and MLflow dependencies described in the Python setup section.

From the repository root, run:

```powershell
python -m src.models.train_final
```

By default, this writes `models/payguard-lightgbm.joblib`, logs the package and readable preprocessing JSON to MLflow, and registers the LightGBM model as `payguard-lightgbm`. The package writer refuses to overwrite an existing file. To generate another package, set a new `model_package_path` in `configs/final.json` first.

The current tuning results file predates the locked-holdout field. The final training command warns about this and continues; the resulting holdout metrics may be optimistic. Rerun tuning with the current holdout rule before treating those metrics as a clean final evaluation.

Local MLflow records use `mlflow.db`. To view them, open a second terminal and run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` in a browser.

## FastAPI prediction API

The application is [`src/serving/app.py`](src/serving/app.py). Start it from the repository root after the model package exists:

```powershell
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

### `GET /health`

Loads the configured package and reports whether it is available:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```

Successful response:

```json
{"status":"ok","model_loaded":true}
```

If the package cannot be loaded, the endpoint returns HTTP 503.

### `POST /predict`

Send a flat JSON object containing all **431 required model features**, using the exact feature names stored in the package. Missing required features or values of the wrong type return HTTP 422. The response contains a boolean fraud decision and a probability when the model supports probabilities:

```json
{"fraud_prediction":false,"fraud_probability":0.0653}
```

This example response is illustrative; the result depends on the submitted transaction and model package. The API also accepts known extra transaction fields such as `TransactionID`, `TransactionDT`, and `isFraud`; preprocessing drops them before scoring.

To submit a real row from the local training files in Windows PowerShell:

```powershell
$payload = python -c "import pandas as pd; tx=pd.read_csv('data/raw/train_transaction.csv',nrows=1); ident=pd.read_csv('data/raw/train_identity.csv'); print(tx.merge(ident,on='TransactionID',how='left').to_json(orient='records')[1:-1])"
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $payload
```

This request-generation example reads the local dataset on the host; the API container itself does not need those files.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PAYGUARD_MODEL_PATH` | `models/payguard-lightgbm.joblib` locally; `/app/models/payguard-lightgbm.joblib` in the image | Path to the packaged model and preprocessing state |
| `PAYGUARD_FRAUD_THRESHOLD` | `0.5` | Probability cutoff for returning `fraud_prediction: true` |

Windows PowerShell example for local serving:

```powershell
$env:PAYGUARD_MODEL_PATH = (Resolve-Path .\models\payguard-lightgbm.joblib).Path
$env:PAYGUARD_FRAUD_THRESHOLD = "0.5"
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

Compose passes these values into the container. Set a different threshold before recreating it, for example:

```powershell
$env:PAYGUARD_FRAUD_THRESHOLD = "0.35"
docker compose up -d --force-recreate
```

## Local Python setup and execution

Use a Python installation compatible with the pinned serving stack (Python 3.13 is used by the Docker image). From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r infra/requirements-serving.txt
python -m pip install xgboost mlflow pytest
```

With a model package present, start the API using the Uvicorn command above. Run the complete synthetic test suite with:

```powershell
python -m pytest tests -q
```

The tests do not need the private dataset or a trained model artifact. They use synthetic data and test-only model doubles.

## Docker image

Build from the repository root so Docker uses the root `.dockerignore` and can read the files under `infra/`:

```powershell
docker build --file infra/Dockerfile --tag payguard-api:local .
```

The image installs only inference dependencies from `infra/requirements-serving.txt` and starts:

```text
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Port `8000` is exposed. The model is not copied into the image, so the image can be built in CI without a local artifact.

## Docker Compose

Before starting Compose, place a trusted package at `models/payguard-lightgbm.joblib`. Then build and run:

```powershell
docker compose up -d --build
docker compose ps
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
```

The Compose service publishes `8000:8000` and mounts the host `models/` directory at `/app/models` as read-only. Follow the valid `POST /predict` example above to score a real row. View logs and stop the service with:

```powershell
docker compose logs -f payguard-api
docker compose down
```

### Model storage and data handling

The `.joblib` artifact is ignored by Git (`*.joblib` in `.gitignore`) and excluded from Docker build contexts by `.dockerignore`, along with `data/` and CSV files. Keep the package in the local `models/` directory or a trusted artifact store, and mount/copy it to the host before starting Compose. The bind mount is read-only. Do not commit the model package, raw training files, or private data.

## GitHub Actions CI

The workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pushes to `dev` and `main` and on pull requests targeting those branches. It uses Python 3.12, installs the serving and model-test dependencies, runs `python -m pytest tests -q`, and builds `payguard-api:ci` with the Dockerfile. Synthetic tests and `.dockerignore` mean CI does not require private datasets or `models/payguard-lightgbm.joblib`. CI builds the image but does not deploy it.

## Limitations and prerequisites

- Training requires the labeled IEEE-CIS training CSVs and tuning results; neither dataset nor model artifact is stored in Git.
- Serving and Compose require a compatible model package at the configured path. If it has not been supplied locally, the API health endpoint reports that the model is unavailable.
- `/predict` requires every one of the package’s 431 features. The feature names and order are stored in the package; callers should use the training/serving preprocessing contract.
- Missing or unseen categorical values are encoded as missing values and use LightGBM’s learned missing-value behavior.
- The current tuning results lack the holdout metadata field, so final training warns that its evaluation may be optimistic.
- The API has no authentication, streaming ingestion, online feature store, drift monitoring, or automated retraining integration yet.
- No cloud or Kubernetes deployment is configured.
- This workstation’s current Python 3.13 installation has a `ctypes` native-library loading issue; Docker inference was verified successfully. A clean Python installation is needed for reliable local native LightGBM imports on that workstation.

## Branching and workflow

- `main` — demo-ready state.
- `dev` — integration branch; feature branches merge here first.
- `feature/<short-name>` — one branch per task (for example, `feature/fastapi-serving`).
- Open a pull request into `dev`; at least one teammate reviews before merge. Merge `dev` to `main` before reviews or demos.

## Documentation index

| Document | Purpose |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Problem definition, scope, and success metrics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design and data flow |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Model comparisons and evaluation notes |
| [`docs/FEATURE_STORE_DESIGN.md`](docs/FEATURE_STORE_DESIGN.md) | Proposed feature-store design |
| [`data/README.md`](data/README.md) | Dataset source and local download instructions |
