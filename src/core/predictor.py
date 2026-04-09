from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..utils.prophet.model import ProphetModel
from ..utils.settings import (
    MODEL_OUTPUT_PATH,
    PREDICTIONS_OUTPUT_PATH,
    TEST_DATA_PATH,
    ensure_directories,
)
from ..utils.xgboost.model import XGBoostModel


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact: Any, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or LOGGER
        self.artifact = artifact

        if isinstance(artifact, dict):
            self.algorithm = str(artifact.get("algorithm", "")).strip().lower()
            self.frequency = str(artifact.get("frequency", "daily")).strip().lower()
            self.models = artifact.get("models", {}) or {}
        elif isinstance(artifact, ProphetModel):
            self.algorithm = "prophet"
            self.frequency = "daily"
            self.models = {"series_0": artifact}
        elif isinstance(artifact, XGBoostModel):
            self.algorithm = "xgboost"
            self.frequency = "daily"
            self.models = {"global": artifact}
        else:
            raise TypeError("Unsupported artifact type for predictor.")

        if self.algorithm not in {"prophet", "xgboost"}:
            raise ValueError("Artifact must contain a supported algorithm: prophet or xgboost.")

    @classmethod
    def from_file(
        cls,
        artifact_path: str | Path = MODEL_OUTPUT_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> "Predictor":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "rb") as file:
            artifact = pickle.load(file)

        return cls(artifact=artifact, logger=logger)

    def _freq_alias(self) -> str:
        return {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
        }.get(self.frequency, "D")

    def _read_data(self, data_path: str | Path) -> pd.DataFrame:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(path)

        raise ValueError("Supported input formats are .csv and .parquet")

    def _get_first_model(self) -> Any:
        if not self.models:
            raise ValueError("No models found in artifact.")
        return next(iter(self.models.values()))

    def _get_xgb_feature_names(self, model: XGBoostModel) -> list[str]:
        feature_names: list[str] = []

        try:
            names = getattr(model.model, "feature_names_in_", None)
            if names is not None:
                feature_names = list(names)
        except Exception:
            feature_names = []

        if not feature_names:
            try:
                booster = model.model.get_booster()
                names = booster.feature_names
                if names:
                    feature_names = list(names)
            except Exception:
                feature_names = []

        return feature_names

    def _build_xgboost_features(self, data: pd.DataFrame, expected_columns: Optional[list[str]] = None) -> pd.DataFrame:
        if "ds" not in data.columns:
            raise ValueError("XGBoost prediction data must contain a 'ds' column.")

        ds = pd.to_datetime(data["ds"])
        features = pd.DataFrame(index=data.index)
        features["year"] = ds.dt.year
        features["month"] = ds.dt.month
        features["day"] = ds.dt.day
        features["dayofweek"] = ds.dt.dayofweek
        features["quarter"] = ds.dt.quarter
        features["weekofyear"] = ds.dt.isocalendar().week.astype(int)

        ignored = {"ds", "y", "TSId"}
        extra = [col for col in data.columns if col not in ignored]

        if extra:
            numeric_cols = [col for col in extra if pd.api.types.is_numeric_dtype(data[col])]
            categorical_cols = [col for col in extra if col not in numeric_cols]

            if numeric_cols:
                features = pd.concat([features, data[numeric_cols]], axis=1)

            if categorical_cols:
                encoded = pd.get_dummies(
                    data[categorical_cols].astype("category"),
                    prefix=categorical_cols,
                    dummy_na=True,
                )
                features = pd.concat([features, encoded], axis=1)

        features = features.fillna(0)

        if expected_columns:
            for col in expected_columns:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_columns]

        return features

    def _predict_prophet(
        self,
        data: Optional[pd.DataFrame] = None,
        periods: Optional[int] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        if not self.models:
            raise ValueError("No Prophet models available in artifact.")

        frames: list[pd.DataFrame] = []

        if data is not None:
            if "ds" not in data.columns:
                raise ValueError("Prediction data must contain a 'ds' column for Prophet.")

            data = data.copy()
            data["ds"] = pd.to_datetime(data["ds"])

            if "TSId" in data.columns and len(self.models) > 1:
                for ts_id, group in data.groupby("TSId", sort=False):
                    key = str(ts_id)
                    if key not in self.models:
                        self.logger.warning("Skipping unknown TSId=%s", ts_id)
                        continue

                    model = self.models[key]
                    forecast = model.predict(
                        future_df=group[["ds"]].copy(),
                        freq=self._freq_alias(),
                        include_history=include_history,
                    )
                    forecast["TSId"] = ts_id
                    if "y" in group.columns:
                        forecast = forecast.merge(group[["ds", "y"]], on="ds", how="left")
                    frames.append(forecast)
            else:
                model = self._get_first_model()
                forecast = model.predict(
                    future_df=data[["ds"]].copy(),
                    freq=self._freq_alias(),
                    include_history=include_history,
                )
                if "TSId" in data.columns:
                    forecast = forecast.merge(data[["ds", "TSId"]], on="ds", how="left")
                if "y" in data.columns:
                    forecast = forecast.merge(data[["ds", "y"]], on="ds", how="left")
                frames.append(forecast)

        else:
            if periods is None:
                raise ValueError("For Prophet, provide either prediction data or periods.")

            for key, model in self.models.items():
                forecast = model.predict(
                    periods=periods,
                    freq=self._freq_alias(),
                    include_history=include_history,
                )
                if len(self.models) > 1:
                    forecast["TSId"] = key
                frames.append(forecast)

        if not frames:
            raise ValueError("No predictions were produced.")

        result = pd.concat(frames, ignore_index=True)

        preferred = ["TSId", "ds", "y", "yhat", "yhat_lower", "yhat_upper", "trend"]
        ordered = [col for col in preferred if col in result.columns]
        ordered += [col for col in result.columns if col not in ordered]
        return result[ordered]

    def _predict_xgboost(self, data: pd.DataFrame) -> pd.DataFrame:
        model = self._get_first_model()
        expected_columns = self._get_xgb_feature_names(model)
        features = self._build_xgboost_features(data.copy(), expected_columns=expected_columns or None)

        result = data.copy()
        result["yhat"] = model.predict(features)

        preferred = ["TSId", "ds", "y", "yhat"]
        ordered = [col for col in preferred if col in result.columns]
        ordered += [col for col in result.columns if col not in ordered]
        return result[ordered]

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        periods: Optional[int] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        if self.algorithm == "prophet":
            return self._predict_prophet(data=data, periods=periods, include_history=include_history)

        if self.algorithm == "xgboost":
            if data is None:
                raise ValueError("XGBoost prediction requires input data.")
            return self._predict_xgboost(data=data)

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def predict_from_file(
        self,
        data_path: Optional[str | Path] = TEST_DATA_PATH,
        periods: Optional[int] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        data = self._read_data(data_path) if data_path else None
        return self.predict(data=data, periods=periods, include_history=include_history)

    def save_predictions(self, predictions: pd.DataFrame, output_path: str | Path = PREDICTIONS_OUTPUT_PATH) -> None:
        ensure_directories()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".csv":
            predictions.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in [".parquet", ".pq"]:
            predictions.to_parquet(output_path, index=False)
        else:
            with open(output_path, "wb") as file:
                pickle.dump(predictions, file)

        metadata = {
            "algorithm": self.algorithm,
            "frequency": self.frequency,
            "rows": int(len(predictions)),
            "columns": list(predictions.columns),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.logger.info("Saved predictions to %s", output_path)

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Run predictions from a trained artifact")
        parser.add_argument(
            "--artifact",
            default=str(MODEL_OUTPUT_PATH),
            type=str,
            help="Path to trained artifact (.pkl)",
        )
        parser.add_argument(
            "--data",
            default=str(TEST_DATA_PATH),
            type=str,
            help="Path to prediction input (.csv or .parquet)",
        )
        parser.add_argument("--periods", type=int, help="Future periods for Prophet prediction")
        parser.add_argument(
            "--include-history",
            action="store_true",
            help="Include training history for Prophet predictions",
        )
        parser.add_argument(
            "--output",
            default=str(PREDICTIONS_OUTPUT_PATH),
            type=str,
            help="Output predictions path",
        )
        return parser


def main() -> None:
    parser = Predictor.build_arg_parser()
    args = parser.parse_args()

    predictor = Predictor.from_file(args.artifact)
    predictions = predictor.predict_from_file(
        data_path=args.data,
        periods=args.periods,
        include_history=args.include_history,
    )
    predictor.save_predictions(predictions, args.output)


if __name__ == "__main__":
    main()