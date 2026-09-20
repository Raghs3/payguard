"""Train the final LightGBM with the best tuned settings and register it in MLflow.

Run from the repo root (with the venv active):
    python -m src.models.train_final

Uses the same time split as train_baseline.py (oldest 80% train, newest 20% test),
so the score is directly comparable to the first XGBoost baseline.
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.benchmark_models import encode_for_sklearn
from src.models.train_baseline import TIME_COL, load_data, make_features, precision_at_k
from src.models.tune_models import SEARCH_SPACE, make_model

CONFIG_PATH = Path("configs/final.json")
TUNING_RESULTS = Path("reports/tuning_results.csv")


def best_lightgbm_params(results_path):
    """Pick the LightGBM trial with the highest mean PR-AUC from the tuning run."""
    results = pd.read_csv(results_path)
    best = results[results["model"] == "lightgbm"].sort_values("pr_auc_mean", ascending=False).iloc[0]
    params = {k: best[k] for k in SEARCH_SPACE}
    # csv gives numpy types; convert whole-number floats back to ints for LightGBM
    for key in ("n_estimators", "max_depth", "min_child_weight"):
        params[key] = int(params[key])
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def train_final(df, params, cfg):
    """Train on the oldest part, score on the newest. Returns (model, metrics)."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    X, y = make_features(df)
    X = encode_for_sklearn(X)
    cut = int(len(X) * cfg["train_fraction"])
    X_train, y_train, X_test, y_test = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]

    model = make_model("lightgbm", params, y_train, cfg["random_state"])
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "precision_at_k": precision_at_k(y_test, scores, cfg["precision_at_k_fraction"]),
    }
    return model, metrics, X_train.head(5)


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    params = best_lightgbm_params(TUNING_RESULTS)
    print("Best tuned LightGBM settings:", params)
    df = load_data(cfg["raw_dir"])

    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name="lightgbm-final"):
        mlflow.log_params(params)
        model, metrics, example = train_final(df, params, cfg)
        mlflow.log_metrics(metrics)
        # registered_model_name puts a versioned copy in the MLflow model registry
        mlflow.lightgbm.log_model(
            model.booster_, name="model", registered_model_name=cfg["registered_model_name"],
            input_example=example,
        )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
