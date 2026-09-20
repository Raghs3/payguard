"""Smoke test: the baseline pipeline runs on tiny fake data (no Kaggle files needed)."""
import numpy as np
import pandas as pd

from src.models.train_baseline import precision_at_k, time_split, train_and_evaluate


def fake_transactions(n=2000):
    rng = np.random.default_rng(0)
    is_fraud = (rng.random(n) < 0.1).astype(int)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 60,
            "isFraud": is_fraud,
            # Fraud rows get bigger amounts, so the model has a signal to find.
            "TransactionAmt": rng.normal(50, 10, n) + is_fraud * 100,
            "ProductCD": rng.choice(["W", "C", "H"], n),
        }
    )


def test_time_split_keeps_order():
    train, val = time_split(fake_transactions(), 0.8)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()


def test_precision_at_k():
    y = [1, 0, 0, 1]
    scores = [0.9, 0.1, 0.2, 0.3]
    assert precision_at_k(y, scores, 0.25) == 1.0  # top 1 of 4 is a fraud


def test_train_and_evaluate_beats_random():
    cfg = {
        "train_fraction": 0.8,
        "precision_at_k_fraction": 0.05,
        "xgboost_params": {
            "n_estimators": 20,
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 0,
        },
    }
    _, metrics = train_and_evaluate(fake_transactions(), cfg)
    assert metrics["roc_auc"] > 0.9


def test_benchmark_cross_validation_runs():
    from src.models.benchmark_models import cross_validate

    cfg = {
        "n_splits": 3,
        "sample_rows": None,
        "random_state": 0,
        "precision_at_k_fraction": 0.05,
        "xgboost_params": {"n_estimators": 10, "tree_method": "hist", "enable_categorical": True},
    }
    results = cross_validate(fake_transactions(), cfg)
    assert "xgboost" in set(results["model"])
    top = results.iloc[0]
    assert top["pr_auc_mean"] > 0.5


def test_tuning_runs_and_compares_both_models():
    from src.models.tune_models import tune

    cfg = {
        "n_splits": 3,
        "n_trials": 2,
        "sample_rows": None,
        "random_state": 0,
        "precision_at_k_fraction": 0.05,
    }
    results = tune(fake_transactions(), cfg)
    assert set(results["model"]) == {"lightgbm", "xgboost"}
    assert (results.groupby("model").size() == 2).all()  # same budget for both


def test_final_model_trains_and_picks_best_params(tmp_path):
    from src.models.train_final import best_lightgbm_params, train_final
    from src.models.tune_models import SEARCH_SPACE

    base = {k: v[0] for k, v in SEARCH_SPACE.items()}
    rows = [
        {"model": "lightgbm", "pr_auc_mean": 0.3, **base},
        {"model": "lightgbm", "pr_auc_mean": 0.9, **{**base, "max_depth": 8}},
        {"model": "xgboost", "pr_auc_mean": 0.99, **base},  # other model must be ignored
    ]
    csv = tmp_path / "tuning.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    params = best_lightgbm_params(csv)
    assert params["max_depth"] == 8 and isinstance(params["n_estimators"], int)

    cfg = {"holdout_fraction": 0.2, "precision_at_k_fraction": 0.05, "random_state": 0}
    _, metrics, _ = train_final(fake_transactions(), params, cfg)
    assert metrics["roc_auc"] > 0.9


def test_tuning_never_touches_holdout_and_final_checks_it(tmp_path, monkeypatch):
    import pytest
    from src.models import tune_models
    from src.models.train_final import best_lightgbm_params

    df = fake_transactions(1000)
    cfg = {"n_splits": 3, "n_trials": 1, "random_state": 0,
           "precision_at_k_fraction": 0.05, "holdout_fraction": 0.3}

    seen_rows = []
    real_score = tune_models.score_params

    def spy(name, params, X, y, cfg_):
        seen_rows.append(len(X))  # how many rows tuning is allowed to use
        return real_score(name, params, X, y, cfg_)

    monkeypatch.setattr(tune_models, "score_params", spy)
    results = tune_models.tune(df, cfg)
    assert seen_rows and all(n == 700 for n in seen_rows)  # 1000 rows minus the locked 30%
    assert (results["holdout_fraction"] == 0.3).all()

    csv = tmp_path / "t.csv"
    results.to_csv(csv, index=False)
    best_lightgbm_params(csv, expected_holdout=0.3)  # matches: fine
    with pytest.raises(ValueError):
        best_lightgbm_params(csv, expected_holdout=0.2)  # mismatch: refuses
