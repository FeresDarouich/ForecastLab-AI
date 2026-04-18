from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.core.predictor import Predictor


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TSId": ["series_0", "series_0", "series_1"],
            "ds": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01"]),
            "y": [149, 156, 114],
            "yhat": [150.5, 157.5, 115.5],
        }
    )


def test_predict_returns_all_artifact_predictions_without_input() -> None:
    predictor = Predictor({"algorithm": "xgboost", "frequency": "monthly", "predictions": make_predictions()})

    predictions = predictor.predict()

    assert predictions.equals(make_predictions())


def test_predict_filters_artifact_predictions_to_requested_rows() -> None:
    predictor = Predictor({"algorithm": "xgboost", "frequency": "monthly", "predictions": make_predictions()})
    request = pd.DataFrame(
        {
            "TSId": ["series_1", "series_0"],
            "ds": ["2025-01-01", "2025-02-01"],
        }
    )

    predictions = predictor.predict(request)

    assert predictions["TSId"].tolist() == ["series_1", "series_0"]
    assert predictions["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-02-01"]
    assert predictions["yhat"].tolist() == [115.5, 157.5]


def test_from_file_loads_pickled_artifact_predictions(tmp_path) -> None:
    artifact_path = tmp_path / "model.pkl"
    with open(artifact_path, "wb") as file:
        pickle.dump({"algorithm": "xgboost", "frequency": "monthly", "predictions": make_predictions()}, file)

    predictor = Predictor.from_file(artifact_path)

    assert predictor.predict().equals(make_predictions())


def test_predict_raises_when_artifact_has_no_predictions() -> None:
    with pytest.raises(ValueError, match="predictions"):
        Predictor({"algorithm": "xgboost", "frequency": "monthly", "models": {}})