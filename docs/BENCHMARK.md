# Model benchmark

Run with `python -m src.models.benchmark_models` (config: `configs/benchmark.json`).
Data: IEEE-CIS train set, 590,540 transactions. Evaluation: 5-fold `TimeSeriesSplit`
(each fold trains on earlier transactions, tests on the next time chunk, so nothing
leaks from the future). Mean over folds; raw numbers in `reports/benchmark_results.csv`.
Hyperparameters are untuned defaults, so the ranking is a first comparison, not final.

| Model | PR-AUC | PR-AUC std | ROC-AUC | Precision@1% | Fit time (s) |
|---|---|---|---|---|---|
| **LightGBM** | **0.562** | 0.017 | 0.900 | 0.923 | 10.0 |
| HistGradientBoosting | 0.539 | 0.020 | 0.895 | 0.913 | 34.2 |
| XGBoost | 0.498 | 0.025 | 0.870 | 0.892 | 20.1 |
| Random forest | 0.450 | 0.025 | 0.866 | 0.849 | 22.1 |
| Logistic regression | 0.351 | 0.104 | 0.837 | 0.621 | 36.3 |
| Isolation forest (unsupervised) | 0.178 | 0.043 | 0.778 | 0.306 | 12.5 |
| Dummy (fraud rate) | 0.037 | 0.003 | 0.500 | 0.044 | 0.0 |

## Decision
LightGBM: best PR-AUC and precision@1%, lowest variance among the top models, and the
fastest boosted-tree model to train. Isolation forest is far weaker alone; it may still
help as an extra anomaly feature later. Next: tune LightGBM and XGBoost fairly before
finalizing.

## Tuned comparison (LightGBM vs XGBoost)

Run with `python -m src.models.tune_models` (config: `configs/tuning.json`). Both models
got the same 12 random-search settings, the same 5 time-series CV folds, and both searched
how strongly to up-weight fraud. Every trial is in `reports/tuning_results.csv` and in the
MLflow experiment `payguard-tuning`.

| | LightGBM | XGBoost |
|---|---|---|
| Best PR-AUC (mean of 5 folds) | **0.576** | 0.566 |
| Fold-to-fold std of that PR-AUC | 0.022 | 0.024 |
| Best ROC-AUC | 0.900 | 0.899 |
| Best precision@1% | 0.927 | 0.924 |
| Average PR-AUC over all 12 trials | 0.547 | 0.532 |
| Time per trial (avg, 5 folds) | 76 s | 133 s |
| Untuned PR-AUC (from the table above) | 0.562 | 0.498 |

Best settings found: LightGBM `n_estimators=600, learning_rate=0.1, max_depth=8,
subsample=1.0, colsample_bytree=0.4, min_child_weight=5, reg_lambda=1`. XGBoost
`n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6,
min_child_weight=5, reg_lambda=10`. Neither preferred extra fraud up-weighting
(`fraud_weight_fraction=0` won for both).

### Reading the result
- Tuning helped XGBoost a lot (0.498 to 0.566), so the untuned ranking was partly unfair to it.
- LightGBM is still ahead, but the gap (0.010) is smaller than the fold-to-fold spread
  (about 0.02), so the two are close to a tie on accuracy. LightGBM is also about 1.75x faster
  to train, and that speed is the clearer advantage.
- Caveats: only 12 random trials each; the same folds were used to pick and to report the
  best settings, so the best-of-12 scores are slightly optimistic; no separate final holdout yet.

### Decision
Keep **LightGBM** as the working model. Before calling it final: retrain the best settings on
the training period and score once on a held-out latest time slice.

## Final model

`python -m src.models.train_final` trains LightGBM with the best tuned settings on the oldest
80% of transactions and scores the newest 20%. This is the same split as the first XGBoost
baseline, so the two are directly comparable. The model is registered in MLflow as
`payguard-lightgbm` (version 1).

| | PR-AUC | ROC-AUC | Precision@1% |
|---|---|---|---|
| First baseline (XGBoost, default settings) | 0.494 | 0.888 | 0.843 |
| **Final (LightGBM, tuned)** | **0.571** | **0.908** | **0.893** |

Caveat: tuning used time-series folds over all the data, so the newest 20% was not completely
unseen during tuning. There is no fully untouched holdout (the Kaggle test file has no
labels), so treat these numbers as slightly optimistic.

## Rule for tuning and model selection (now enforced in code)

The newest 20% of transactions (`holdout_fraction` in the configs) is locked away before any
tuning. `benchmark_models.py` and `tune_models.py` only see the older 80%, and `train_final.py`
scores the locked 20% once. `tune_models.py` records the `holdout_fraction` it used in
`reports/tuning_results.csv`, and `train_final.py` refuses to run if it doesn't match its own
setting (and warns if the tuning file has no holdout recorded).

The results above were produced BEFORE this change, so they are slightly optimistic (see the
caveat). Any tuning or benchmark run from now on is clean. To get a clean final number, rerun
`tune_models` then `train_final`.
