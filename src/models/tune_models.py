"""Fairly tune LightGBM and XGBoost, then compare their best settings.

Run from the repo root (with the venv active):
    python -m src.models.tune_models

Fairness rules (so the ranking can be trusted):
  - both models get the same number of random-search trials
  - both are scored on the exact same time-series CV folds
  - both search the same kinds of knobs, including how strongly to up-weight fraud
  - the winner is picked by mean PR-AUC across folds
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

from src.models.benchmark_models import encode_for_sklearn
from src.models.train_baseline import TIME_COL, load_data, make_features, precision_at_k

CONFIG_PATH = Path("configs/tuning.json")
RESULTS_PATH = Path("reports/tuning_results.csv")

# Same shape of search space for both models. Each list is what random search picks from.
SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 20],
    "reg_lambda": [0.0, 1.0, 10.0],
    # 1 = no extra weight; 0.5/1.0 = fraction of the full (#legit / #fraud) ratio
    "fraud_weight_fraction": [0.0, 0.25, 0.5, 1.0],
}


def make_model(name, params, y_train, seed):
    """Build a model from one sampled parameter set."""
    params = dict(params)
    fraction = params.pop("fraud_weight_fraction")
    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    weight = 1.0 + fraction * (ratio - 1.0)
    if name == "xgboost":
        return xgb.XGBClassifier(
            tree_method="hist", enable_categorical=True, scale_pos_weight=weight,
            random_state=seed, **params,
        )
    # LightGBM: max_depth alone is fine; num_leaves is left at its default (31)
    return lgb.LGBMClassifier(scale_pos_weight=weight, random_state=seed, verbose=-1, **params)


def score_params(name, params, X, y, cfg):
    """Mean/std of the metrics across the time-series folds for one parameter set."""
    folds = []
    start = time.time()
    for train_idx, test_idx in TimeSeriesSplit(n_splits=cfg["n_splits"]).split(X):
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_te.sum() == 0:
            continue
        model = make_model(name, params, y_tr, cfg["random_state"])
        model.fit(X.iloc[train_idx], y_tr)
        scores = model.predict_proba(X.iloc[test_idx])[:, 1]
        folds.append(
            {
                "pr_auc": average_precision_score(y_te, scores),
                "roc_auc": roc_auc_score(y_te, scores),
                "precision_at_k": precision_at_k(y_te, scores, cfg["precision_at_k_fraction"]),
            }
        )
    folds = pd.DataFrame(folds)
    out = {f"{m}_mean": folds[m].mean() for m in folds.columns}
    out["pr_auc_std"] = folds["pr_auc"].std()
    out["seconds"] = time.time() - start
    return out


def tune(df, cfg):
    """Random search for each model. Returns a dataframe with one row per trial."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if cfg.get("sample_rows"):
        df = df.iloc[: cfg["sample_rows"]]
    X, y = make_features(df)
    X_num = encode_for_sklearn(X)  # LightGBM sklearn API is happiest with numeric input

    # Same list of parameter sets is used for BOTH models: an identical search budget.
    trials = list(ParameterSampler(SEARCH_SPACE, n_iter=cfg["n_trials"], random_state=cfg["random_state"]))
    rows = []
    for name in ["lightgbm", "xgboost"]:
        features = X_num if name == "lightgbm" else X
        for i, params in enumerate(trials):
            result = score_params(name, params, features, y, cfg)
            rows.append({"model": name, "trial": i, **params, **result})
            print(f"{name} trial {i}: PR-AUC {result['pr_auc_mean']:.4f} ({result['seconds']:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    df = load_data(cfg["raw_dir"])
    results = tune(df, cfg)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    mlflow.set_experiment(cfg["mlflow_experiment"])
    param_cols = list(SEARCH_SPACE)
    for _, row in results.iterrows():  # log every trial so the search is auditable
        with mlflow.start_run(run_name=f"{row['model']}-trial{int(row['trial'])}"):
            mlflow.log_param("model", row["model"])
            mlflow.log_params({c: row[c] for c in param_cols})
            mlflow.log_metrics({k: float(row[k]) for k in row.index if k.endswith("_mean") or k in ("pr_auc_std", "seconds")})

    best = results.loc[results.groupby("model")["pr_auc_mean"].idxmax()]
    print("\nBest trial per model:\n", best.round(4).T.to_string())
    print(f"\nAll trials saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
