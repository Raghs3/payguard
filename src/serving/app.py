"""FastAPI service for inference with a packaged PayGuard LightGBM model."""
from __future__ import annotations

import logging
import math
import os
from numbers import Real
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from src.models.model_package import load_model_package

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("models/payguard-lightgbm.joblib")


def _validate_transaction(transaction: dict[str, Any], preprocessor):
    if not transaction:
        raise HTTPException(status_code=422, detail="Transaction must contain feature values")

    missing = [name for name in preprocessor.feature_names if name not in transaction]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})

    categorical = set(preprocessor.categorical_mappings)
    for name in preprocessor.feature_names:
        value = transaction[name]
        if isinstance(value, (dict, list)):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be a scalar")
        if name in categorical:
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise HTTPException(status_code=422, detail=f"Feature '{name}' has an invalid category")
        elif value is not None:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise HTTPException(status_code=422, detail=f"Feature '{name}' must be a finite number or null")
    return transaction


def create_app(model_path=None, model_loader=load_model_package):
    """Create an app; model loading is lazy and cached after the first request."""
    app = FastAPI(title="PayGuard Fraud Prediction API", version="1.0.0")
    configured_path = Path(model_path or os.getenv("PAYGUARD_MODEL_PATH", DEFAULT_MODEL_PATH))
    app.state.model_package = None

    def get_package():
        if app.state.model_package is None:
            try:
                app.state.model_package = model_loader(configured_path)
            except Exception as exc:
                logger.exception("Unable to load PayGuard model package")
                raise HTTPException(
                    status_code=503,
                    detail="Model package is unavailable or could not be loaded",
                ) from exc
        return app.state.model_package

    @app.get("/health")
    def health():
        try:
            get_package()
        except HTTPException:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "model_loaded": False},
            )
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict")
    def predict(transaction: dict[str, Any]):
        package = get_package()
        preprocessor = package["preprocessor"]
        validated = _validate_transaction(transaction, preprocessor)
        try:
            features = preprocessor.transform_transactions(pd.DataFrame([validated]))
            model = package["model"]
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(features)[0][1])
                prediction = probability >= float(os.getenv("PAYGUARD_FRAUD_THRESHOLD", "0.5"))
            else:
                raw_prediction = model.predict(features)[0]
                prediction = bool(raw_prediction)
                probability = None
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("PayGuard prediction failed")
            raise HTTPException(status_code=422, detail="Transaction could not be scored") from exc
        return {"fraud_prediction": prediction, "fraud_probability": probability}

    return app


app = create_app()
