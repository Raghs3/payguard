import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.models.benchmark_models import encode_for_sklearn
from src.models.model_package import (
    PayGuardPreprocessor,
    load_model_package,
    save_model_package,
)
from src.models.train_baseline import make_features


def transactions():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "TransactionDT": [10, 20, 30, 40, 50, 60],
        "isFraud": [0, 1, 0, 1, 0, 1],
        "amount": [10.0, 100.0, 12.0, 90.0, 14.0, 80.0],
        "ProductCD": ["W", "C", "W", "C", "W", "C"],
    })


def test_packaged_preprocessing_matches_training_codes_for_known_categories():
    X, _ = make_features(transactions())
    preprocessor = PayGuardPreprocessor.fit(X)
    expected = encode_for_sklearn(X)
    actual = preprocessor.transform(X)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.columns.tolist() == X.columns.tolist()


def test_missing_and_unseen_categories_become_lightgbm_missing_values():
    X, _ = make_features(transactions())
    preprocessor = PayGuardPreprocessor.fit(X)
    request = pd.DataFrame({"amount": [20.0, 30.0], "ProductCD": [None, "NEW"]})
    transformed = preprocessor.transform_transactions(request)
    assert np.isnan(transformed.loc[0, "ProductCD"])
    assert np.isnan(transformed.loc[1, "ProductCD"])


def test_model_package_round_trip_preserves_predictions_and_refuses_overwrite(tmp_path):
    X, y = make_features(transactions())
    preprocessor = PayGuardPreprocessor.fit(X)
    encoded = preprocessor.transform(X)
    model = LGBMClassifier(n_estimators=5, verbosity=-1, random_state=0).fit(encoded, y)
    path = tmp_path / "package.joblib"
    save_model_package(path, model, preprocessor)
    loaded = load_model_package(path)
    np.testing.assert_allclose(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X)),
        model.predict_proba(encoded),
    )
    with pytest.raises(FileExistsError):
        save_model_package(path, model, preprocessor)
