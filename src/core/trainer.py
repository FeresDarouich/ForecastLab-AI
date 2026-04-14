from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .prepare import Prepare
from ..utils.prophet.model import ProphetModel
from ..utils.settings import (
    HYPERPARAMETERS_PATH,
    MODEL_OUTPUT_PATH,
    TRAIN_DATA_PATH,
    ensure_directories,
)
from ..utils.xgboost.model import XGBoostModel


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, hyperparameters: dict, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or LOGGER
        self.hyperparameters = hyperparameters

        self.frequency = str(hyperparameters.get("frequency", "daily")).strip().lower()
        if self.frequency not in ["monthly", "weekly", "daily"]:
            raise ValueError("Frequency should be daily, weekly, or monthly.")

        alg = hyperparameters.get("algorithm", {})
        self.algorithm = str(alg.get("name", "auto")).strip().lower()
        if self.algorithm not in ["prophet", "xgboost", "auto"]:
            raise ValueError("algorithm should be prophet, xgboost, or auto")

        self.models = alg.get("models", {})
        self.cutoffs = alg.get("cutoffs", {})
        self.fbpt_hyperparameters = hyperparameters.get("prophet", {})
        self.xgb_hyperparameters = hyperparameters.get("xgboost", {})
        self.seasonality = hyperparameters.get("seasonality", "auto")
        self.probabilistic_forecast = hyperparameters.get("probabilistic_forecast", {})

    @classmethod
    def from_hyperparameters_file(
        cls,
        path_to_hyperparameters_file: str | Path = HYPERPARAMETERS_PATH,
        logger: Optional[logging.Logger] = None,
    ) -> "Trainer":
        hyperparameters = Prepare.load_hyperparameters(path_to_hyperparameters_file)
        return cls(hyperparameters=hyperparameters, logger=logger)

    def _freq_alias(self) -> str:
        return {
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
        }[self.frequency]

    def _resolve_algorithm(self, data: pd.DataFrame) -> str:
        if self.algorithm != "auto":
            return self.algorithm

        selected = str((self.models or {}).get("smooth", "prophet")).strip().lower()
        if selected not in ["prophet", "xgboost"]:
            selected = "prophet"

        self.logger.info("Algorithm set to auto. Resolved to '%s'.", selected)
        return selected

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return Prepare.prepare_training_data(data)

    def _get_prophet_kwargs(self) -> Dict[str, Any]:
        raw = dict(self.fbpt_hyperparameters or {})
        raw.pop("exogenous", None)

        allowed = {
            "growth",
            "changepoint_prior_scale",
            "changepoint_range",
            "seasonality_mode",
            "seasonality_prior_scale",
            "holidays_prior_scale",
            "interval_width",
            "uncertainty_samples",
        }
        kwargs = {k: v for k, v in raw.items() if k in allowed}

        if kwargs.get("growth") == "auto":
            kwargs.pop("growth")

        return kwargs

    def _get_xgboost_kwargs(self) -> Dict[str, Any]:
        raw = dict((self.xgb_hyperparameters or {}).get("model_parameters", {}) or {})
        if "eta" in raw and "learning_rate" not in raw:
            raw["learning_rate"] = raw.pop("eta")

        if raw.get("objective") == "reg:quantileerror":
            quantiles = list((self.probabilistic_forecast or {}).get("quantiles", []))
            level_method = str((self.xgb_hyperparameters or {}).get("level_method", "median")).strip().lower()

            if quantiles:
                raw.setdefault("quantile_alpha", quantiles[0] if len(quantiles) == 1 else quantiles)
            elif level_method == "median":
                raw.setdefault("quantile_alpha", 0.5)
            else:
                self.logger.info(
                    "XGBoost objective 'reg:quantileerror' requested without quantiles; "
                    "falling back to 'reg:squarederror' for point forecasts."
                )
                raw["objective"] = "reg:squarederror"

        return raw

    def _build_xgboost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)

        ds = pd.to_datetime(data["ds"])
        features["year"] = ds.dt.year
        features["month"] = ds.dt.month
        features["day"] = ds.dt.day
        features["dayofweek"] = ds.dt.dayofweek
        features["quarter"] = ds.dt.quarter
        features["weekofyear"] = ds.dt.isocalendar().week.astype(int)

        exogenous = (self.xgb_hyperparameters or {}).get("exogenous", {}) or {}
        numerical = [col for col in exogenous.get("numerical", []) if col in data.columns]
        categorical = [col for col in exogenous.get("categorical", []) if col in data.columns]

        if numerical:
            features = pd.concat([features, data[numerical]], axis=1)

        if categorical:
            encoded = pd.get_dummies(data[categorical].astype("category"), prefix=categorical, dummy_na=True)
            features = pd.concat([features, encoded], axis=1)

        features = features.fillna(0)

        if features.empty:
            raise ValueError("No features available for XGBoost training.")

        return features

    def apply_prophet_model(self, ts_id: Any, group: pd.DataFrame) -> tuple[ProphetModel, pd.DataFrame]:
        model = ProphetModel(model_kwargs=self._get_prophet_kwargs())
        train_df = group[["ds", "y"]].copy()

        model.fit(train_df)
        forecast = model.predict(future_df=train_df[["ds"]], freq=self._freq_alias())

        cols = ["ds", "yhat"]
        for extra in ["yhat_lower", "yhat_upper", "trend"]:
            if extra in forecast.columns:
                cols.append(extra)

        result = forecast[cols].copy()
        result["TSId"] = ts_id
        result = result.merge(train_df, on="ds", how="left")

        ordered_cols = ["TSId", "ds", "y"]
        ordered_cols += [c for c in ["yhat", "yhat_lower", "yhat_upper", "trend"] if c in result.columns]
        result = result[ordered_cols]

        return model, result

    def apply_xgboost_model(self, data: pd.DataFrame) -> tuple[XGBoostModel, pd.DataFrame]:
        features = self._build_xgboost_features(data)
        target = data["y"]

        model = XGBoostModel(model_kwargs=self._get_xgboost_kwargs())
        model.fit(features, target)

        pred_df = data[["ds", "y"]].copy()
        if "TSId" in data.columns:
            pred_df["TSId"] = data["TSId"]
        pred_df["yhat"] = model.predict(features)

        ordered_cols = ["ds", "y", "yhat"]
        if "TSId" in pred_df.columns:
            ordered_cols = ["TSId"] + ordered_cols

        return model, pred_df[ordered_cols]

    def _train(self, data: Any) -> Dict[str, Any]:
        data = self._prepare_data(data)
        algorithm = self._resolve_algorithm(data)

        artifact: Dict[str, Any] = {
            "algorithm": algorithm,
            "frequency": self.frequency,
            "input_data_header": list(data.columns),
            "predictions": None,
            "models": {},
        }

        if algorithm == "prophet":
            if "TSId" in data.columns:
                forecast_list = []
                for ts_id, group in data.groupby("TSId", sort=False):
                    self.logger.info("Training Prophet for TSId=%s", ts_id)
                    model, forecast = self.apply_prophet_model(ts_id, group.copy())
                    artifact["models"][str(ts_id)] = model
                    forecast_list.append(forecast)
                artifact["predictions"] = pd.concat(forecast_list, ignore_index=True)
            else:
                self.logger.info("Training Prophet for single series")
                model, forecast = self.apply_prophet_model("series_0", data.copy())
                artifact["models"]["series_0"] = model
                artifact["predictions"] = forecast

        elif algorithm == "xgboost":
            self.logger.info("Training XGBoost")
            model, forecast = self.apply_xgboost_model(data.copy())
            artifact["models"]["global"] = model
            artifact["predictions"] = forecast

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return artifact

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        return self._train(data)

    def train_from_file(self, data_path: str | Path = TRAIN_DATA_PATH) -> Dict[str, Any]:
        data = Prepare.load_training_data(data_path)
        return self.train(data)

    def save_output(self, artifact: Dict[str, Any], output_path: str | Path = MODEL_OUTPUT_PATH) -> None:
        ensure_directories()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions = artifact.get("predictions")
        metadata = {
            "algorithm": artifact.get("algorithm"),
            "frequency": artifact.get("frequency"),
            "input_data_header": artifact.get("input_data_header"),
            "prediction_rows": int(len(predictions)) if isinstance(predictions, pd.DataFrame) else 0,
        }

        if output_path.suffix.lower() == ".csv":
            if not isinstance(predictions, pd.DataFrame):
                raise ValueError("No predictions available to save.")
            predictions.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in [".parquet", ".pq"]:
            if not isinstance(predictions, pd.DataFrame):
                raise ValueError("No predictions available to save.")
            predictions.to_parquet(output_path, index=False)
        else:
            with open(output_path, "wb") as file:
                pickle.dump(artifact, file)

        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.logger.info("Saved output to %s", output_path)

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train forecasting model")
        parser.add_argument(
            "--hyperparameters",
            default=str(HYPERPARAMETERS_PATH),
            type=str,
            help="Path to hyperparameters json file",
        )
        parser.add_argument(
            "--data",
            default=str(TRAIN_DATA_PATH),
            type=str,
            help="Path to training data file",
        )
        parser.add_argument(
            "--output",
            default=str(MODEL_OUTPUT_PATH),
            type=str,
            help="Output artifact path",
        )
        return parser


def main() -> None:
    parser = Trainer.build_arg_parser()
    args = parser.parse_args()

    trainer = Trainer.from_hyperparameters_file(args.hyperparameters)
    artifact = trainer.train_from_file(args.data)
    trainer.save_output(artifact, args.output)


if __name__ == "__main__":
    main()