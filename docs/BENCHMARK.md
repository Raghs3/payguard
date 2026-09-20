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
