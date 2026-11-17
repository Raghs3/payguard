import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.models.model_package import PayGuardPreprocessor
from src.serving.app import create_app


class TestOnlyModel:
    """Deterministic stand-in used only by API unit tests."""

    def predict_proba(self, features):
        return np.array([[0.2, 0.8] for _ in range(len(features))])


class PredictionOnlyModel:
    def predict(self, features):
        return np.ones(len(features), dtype=int)


def make_client():
    preprocessor = PayGuardPreprocessor(
        feature_names=["amount", "ProductCD"],
        categorical_mappings={"ProductCD": ["C", "W"]},
    )
    package = {"model": TestOnlyModel(), "preprocessor": preprocessor, "feature_names": preprocessor.feature_names,
               "format_version": 1}
    return create_app(model_path="not-used.joblib", model_loader=lambda _path: package)


def route(app, path):
    return next(item.endpoint for item in app.routes if getattr(item, "path", None) == path)


def test_health_reports_loaded_model():
    assert route(make_client(), "/health")() == {"status": "ok", "model_loaded": True}


def test_predict_preprocesses_and_returns_probability():
    result = route(make_client(), "/predict")({"amount": 12.5, "ProductCD": "W"})
    assert result == {"fraud_prediction": True, "fraud_probability": 0.8}


def test_predict_rejects_missing_or_invalid_features():
    client = make_client()
    predict = route(client, "/predict")
    with pytest.raises(HTTPException) as missing:
        predict({"amount": 12.5})
    with pytest.raises(HTTPException) as invalid:
        predict({"amount": "many", "ProductCD": "W"})
    assert missing.value.status_code == 422
    assert invalid.value.status_code == 422


def test_predict_returns_null_probability_when_model_does_not_support_probabilities():
    app = make_client()
    route(app, "/health")()
    package = app.state.model_package
    package["model"] = PredictionOnlyModel()
    result = route(app, "/predict")({"amount": 12.5, "ProductCD": "W"})
    assert result == {"fraud_prediction": True, "fraud_probability": None}


def test_json_null_in_numeric_feature_stays_numeric():
    preprocessor = PayGuardPreprocessor(
        feature_names=["amount", "ProductCD"],
        categorical_mappings={"ProductCD": ["C", "W"]},
    )
    transformed = preprocessor.transform_transactions(
        pd.DataFrame([{"amount": None, "ProductCD": "W"}])
    )
    assert np.isnan(transformed.loc[0, "amount"])
    assert transformed["amount"].dtype == np.dtype("float64")
    assert transformed.loc[0, "ProductCD"] == 1.0


def test_model_loading_failure_returns_service_unavailable():
    def fail_loader(_path):
        raise FileNotFoundError("missing model")

    client = create_app(model_path="missing.joblib", model_loader=fail_loader)
    health = route(client, "/health")()
    assert health.status_code == 503
    with pytest.raises(HTTPException) as failed:
        route(client, "/predict")({"amount": 1, "ProductCD": "W"})
    assert failed.value.status_code == 503
