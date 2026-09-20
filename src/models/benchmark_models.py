"""Compare several model types with time-aware cross-validation.

Run from the repo root (with the venv active):
    python -m src.models.benchmark_models

Why time-aware CV? Normal K-fold shuffles rows, so a model would train on
transactions from AFTER the ones it is tested on (data leakage). Instead we use
TimeSeriesSplit: each fold trains on an earlier chunk of time and tests on the
chunk right after it, which is how the model is used in production.
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.train_baseline import TIME_COL, load_data, make_features, precision_at_k, time_split

CONFIG_PATH = Path("configs/benchmark.json")
RESULTS_PATH = Path("reports/benchmark_results.csv")


def encode_for_sklearn(X):
    """Turn category columns into integer codes (missing -> NaN).

    XGBoost/HistGradientBoosting read categories natively, but Logistic
    Regression / Random Forest / Isolation Forest need plain numbers.
    """
    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        codes = X[col].cat.codes.astype("float64")
        X[col] = codes.where(codes >= 0)
    return X


def build_models(cfg):
    """name -> (model, needs_numeric_input, is_unsupervised)."""
    seed = cfg["random_state"]
    models = {
        # Predicts the fraud rate for everyone. Any real model must beat this.
        "dummy_prior": (DummyClassifier(strategy="prior"), True, False),
        "logistic_regression": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced"),
            ),
            True,
            False,
        ),
        "random_forest": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=seed,
                ),
            ),
            True,
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=seed),
            True,
            False,
        ),
        # Never sees the labels: scores how "unusual" a transaction looks.
        "isolation_forest": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                IsolationForest(n_estimators=100, n_jobs=-1, random_state=seed),
            ),
            True,
            True,
        ),
    }
    xgb_params = dict(cfg["xgboost_params"])
    models["xgboost"] = (xgb.XGBClassifier(**xgb_params), False, False)
    try:  # LightGBM is optional; only benchmarked if it is installed.
        import lightgbm as lgb

        models["lightgbm"] = (
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=seed, verbose=-1),
            False,
            False,
        )
    except ImportError:
        print("lightgbm not installed - skipping it")
    return models


def fit_and_score(model, unsupervised, X_train, y_train, X_test):
    """Fit one model and return a fraud score per test row (higher = riskier)."""
    if unsupervised:
        model.fit(X_train)  # no labels
        return -model.score_samples(X_test)  # sklearn: lower = more unusual, so flip the sign
    if isinstance(model, xgb.XGBClassifier):  # up-weight the rare fraud rows
        model.set_params(scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def cross_validate(df, cfg):
    """Return one row of averaged metrics per model."""
    # Lock away the newest part for the final test; models are compared on the older part only.
    df, _locked_holdout = time_split(df, 1 - cfg.get("holdout_fraction", 0.2))
    if cfg.get("sample_rows"):  # keep the first N rows in time order for a quick run
        df = df.iloc[: cfg["sample_rows"]]
    X, y = make_features(df)
    X_numeric = encode_for_sklearn(X)
    splitter = TimeSeriesSplit(n_splits=cfg["n_splits"])

    rows = []
    for name, (model, needs_numeric, unsupervised) in build_models(cfg).items():
        features = X_numeric if needs_numeric else X
        fold_metrics = []
        for train_idx, test_idx in splitter.split(features):
            X_tr, X_te = features.iloc[train_idx], features.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if y_te.sum() == 0:  # PR-AUC is undefined without any fraud in the fold
                continue
            start = time.time()
            scores = fit_and_score(model, unsupervised, X_tr, y_tr, X_te)
            fold_metrics.append(
                {
                    "pr_auc": average_precision_score(y_te, scores),
                    "roc_auc": roc_auc_score(y_te, scores),
                    "precision_at_k": precision_at_k(y_te, scores, cfg["precision_at_k_fraction"]),
                    "fit_seconds": time.time() - start,
                }
            )
        if not fold_metrics:
            raise ValueError("No fold had any fraud in its test chunk; use more rows (sample_rows).")
        folds = pd.DataFrame(fold_metrics)
        row = {"model": name, "folds_used": len(folds)}
        for metric in folds.columns:
            row[f"{metric}_mean"] = folds[metric].mean()
            row[f"{metric}_std"] = folds[metric].std()
        rows.append(row)
        print(f"{name}: PR-AUC {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f}")
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    df = load_data(cfg["raw_dir"])
    mlflow.set_experiment(cfg["mlflow_experiment"])
    results = cross_validate(df, cfg)
    for _, row in results.iterrows():  # one MLflow run per model
        with mlflow.start_run(run_name=row["model"]):
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "model" and pd.notna(v)})
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print("\n", results.round(4).to_string(index=False))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
