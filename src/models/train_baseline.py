"""Baseline fraud model: plain XGBoost on the raw IEEE-CIS data.

Run from the repo root (with the venv active):
    python -m src.models.train_baseline
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

CONFIG_PATH = Path("configs/baseline.json")
TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


def load_data(raw_dir):
    """Read the two Kaggle files and join them on TransactionID.

    Not every transaction has identity info, so we use a left join:
    keep every transaction, fill identity columns with NaN when missing.
    """
    raw_dir = Path(raw_dir)
    transactions = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    return transactions.merge(identity, on=ID_COL, how="left")


def time_split(df, train_fraction):
    """Oldest rows -> train, newest rows -> validation.

    In production the model is trained on the past and scores the future,
    so we evaluate the same way. A random split would leak future patterns
    into training and make the scores look better than they really are.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


def make_features(df):
    """Split a dataframe into (X features, y labels) that XGBoost accepts."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL, TIME_COL])
    # Text columns become 'category' so XGBoost can use them natively.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):  # works on pandas 2 and 3
            X[col] = X[col].astype("category")
    return X, y


def precision_at_k(y_true, scores, k_fraction):
    """Of the top k% highest-scored transactions, what share are real fraud?

    This mimics a review queue: analysts can only look at the riskiest few.
    """
    k = max(1, int(len(scores) * k_fraction))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y_true)[top_idx].mean())


def train_and_evaluate(df, cfg):
    """Train on the past, score the future. Returns (model, metrics dict)."""
    # Convert text columns on the FULL data first so train and validation
    # share identical category lists.
    X_all, y_all = make_features(df.sort_values(TIME_COL).reset_index(drop=True))
    cut = int(len(X_all) * cfg["train_fraction"])
    X_train, y_train = X_all.iloc[:cut], y_all.iloc[:cut]
    X_val, y_val = X_all.iloc[cut:], y_all.iloc[cut:]

    # Fraud is rare, so up-weight fraud rows: (# legit) / (# fraud).
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **cfg["xgboost_params"])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    scores = model.predict_proba(X_val)[:, 1]  # probability of fraud
    metrics = {
        "pr_auc": float(average_precision_score(y_val, scores)),
        "roc_auc": float(roc_auc_score(y_val, scores)),
        "precision_at_k": precision_at_k(y_val, scores, cfg["precision_at_k_fraction"]),
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
    }
    return model, metrics


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    df = load_data(cfg["raw_dir"])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run():
        mlflow.log_params(cfg["xgboost_params"])
        mlflow.log_param("train_fraction", cfg["train_fraction"])
        start = time.time()
        model, metrics = train_and_evaluate(df, cfg)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("train_seconds", time.time() - start)
        mlflow.xgboost.log_model(model, name="model")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
