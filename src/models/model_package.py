"""Reusable preprocessing and serialization for the final PayGuard model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Keep these aligned with train_baseline.make_features without importing the
# training module (and its training-only XGBoost/MLflow runtime) during serving.
TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"


class PayGuardPreprocessor:
    """Preserve training feature order and pandas category-code mappings.

    Missing and unseen categories become NaN, matching the existing missing-value
    encoding and allowing LightGBM to follow its learned missing-value branches.
    """

    def __init__(self, feature_names, categorical_mappings):
        self.feature_names = list(feature_names)
        self.categorical_mappings = {
            name: list(values) for name, values in categorical_mappings.items()
        }

    @classmethod
    def fit(cls, X):
        mappings = {
            name: X[name].cat.categories.tolist()
            for name in X.select_dtypes(include="category").columns
        }
        return cls(X.columns.tolist(), mappings)

    def transform(self, X):
        """Encode a feature frame; known categories retain pandas' exact codes."""
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
        X = X.loc[:, self.feature_names].copy()
        for name, categories in self.categorical_mappings.items():
            # Casting against the fitted categories makes missing and unseen values -1.
            values = pd.Categorical(X[name], categories=categories)
            codes = pd.Series(values.codes, index=X.index, dtype="float64")
            X[name] = codes.where(codes >= 0, np.nan)
        return X

    def transform_transactions(self, transactions):
        """Apply the same raw-column selection and encoding as final training."""
        ignored = [name for name in (TARGET, ID_COL, TIME_COL) if name in transactions.columns]
        X = transactions.drop(columns=ignored).copy()
        categorical = set(self.categorical_mappings)
        for name in X.columns:
            if name in categorical:
                X[name] = X[name].astype("category")
            elif not pd.api.types.is_numeric_dtype(X[name]):
                # JSON nulls can make a numeric column object-typed in a one-row
                # frame. Use the fitted schema to keep it numeric at inference.
                X[name] = pd.to_numeric(X[name], errors="coerce")
        return self.transform(X)

    def to_dict(self):
        return {
            "format_version": 1,
            "feature_names": self.feature_names,
            "categorical_mappings": self.categorical_mappings,
            "unknown_category_policy": "encode as NaN (LightGBM missing-value path)",
        }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_model_package(path, model, preprocessor):
    """Write a self-contained model + preprocessing package without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model package: {path}")
    joblib.dump({"model": model, "preprocessor": preprocessor,
                 "feature_names": preprocessor.feature_names,
                 "format_version": 1}, path)


def load_model_package(path):
    """Load a package created by :func:`save_model_package`. Only load trusted files."""
    package = joblib.load(path)
    if package.get("format_version") != 1:
        raise ValueError("Unsupported PayGuard model package format")
    if package["feature_names"] != package["preprocessor"].feature_names:
        raise ValueError("Model package feature order does not match preprocessing metadata")
    return package


def save_preprocessing_json(path, preprocessor):
    Path(path).write_text(json.dumps(preprocessor.to_dict(), indent=2, default=_json_default))
