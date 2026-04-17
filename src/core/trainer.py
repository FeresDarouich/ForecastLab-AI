from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from .prepare import Prepare
from ..utils.prophet.model import ProphetModel
from ..utils.settings import (
    HYPERPARAMETERS_PATH,
    MODEL_OUTPUT_PATH,
    TRAIN_DATA_PATH,
    ensure_directories,
)
from ..utils.xgboost.model import XGBoostModel
from ..utils.modeling import model_selection
from sklearn.preprocessing import TargetEncoder

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

    def _resolve_algorithm(self, series: pd.series) -> str:
        if self.algorithm != "auto":
            return self.algorithm

        selected = str(model_selection(series, self.cutoffs, self.models)).strip().lower()
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
    
    def apply_level(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        tr = data[data["TestIndicator"] == 0]
        ts = data[data["TestIndicator"] == 1]
        if "TSId" not in data.columns:
            raise KeyError("input data missing TSId !")
        elif method not in ["mean", "median"]:
            raise ValueError(f"level method {method} is not supported !!")
        else:
            level_map = tr.groupby("TSId")["y"].agg(method).to_dict()
            tr["level"] = tr["TSId"].map(level_map)
            ts["level"] = tr["TSId"].map(level_map)
            ts["level"] = ts["level"].fillna(tr["y"].agg(method))
        return pd.concat([tr,ts], ignore_index=True)
    
    def apply_encoding(self, data: pd.DataFrame, column: str) -> pd.Series:
        encoder = TargetEncoder()
        encoded = encoder.fit_transform(data[[column]], data["y"])
        return encoded
    
    def parse_seasonality(self,seas: str) -> list[tuple]:
        if seas.lower() == "auto":
            defaults = {
                "daily": [(365,5), (30.5,3), (7,2)],
                "weekly": [(52,5), (4.5,2)],
                "monthly": [(12, 3)],
            }
            return defaults[self.frequency]
        else:
            terms = []
            seas_terms = seas.split(",")
            if seas_terms:
                seasonalities = [seasonality.split("-") for seasonality in seas_terms]
                for s in seasonalities:
                    try:
                        period, n_component = s
                        terms.append(float(period), int(n_component))
                    except:
                        raise ValueError("Seasonalities values schema is incorrect!!")
            return terms
        
    def compute_fourier(self, data: pd.DataFrame, period: float, term: int) -> pd.DataFrame:
        data["ds"] = pd.to_datetime(data["ds"])
        ds = data["ds"].drop_duplicates().reset_index(drop=True)
        if self.frequency == "monthly":
            t = (ds.dt.to_period("M").astype(int) - ds.dt.to_period("M").astype(int).min())/period
        elif self.frequency == "weekly":
            t = (ds.dt.to_period("W").astype(int) - ds.dt.to_period("W").astype(int).min())/period
        elif self.frequency == "daily":
            t = (ds - ds.min()).dt.days / period
        else: 
            raise ValueError("Unable to compute fourier terms. unsupported frequency !!")
        fourier_cos = np.zeros(len(ds))
        fourier_sin = np.zeros(len(ds))
        fourier_cos += np.cos(2 * np.pi * term * t)
        fourier_sin += np.sin(2 * np.pi * term * t)
        F_component = pd.DataFrame(
            {"ds": ds, "fourier_cos": fourier_cos, "fourier_sin": fourier_sin}
        )
        return F_component

    def add_seasonality(self, seasonality: list, data: pd.DataFrame) -> pd.DataFrame:
        if len(seasonality) != 0:
            for seas in seasonality:
                for term in range(1, seas[1] + 1):
                    name = "seasonality_" + str(seas[0]) + "_component_" + str(term)
                    F_compoent = self.compute_fourier(data, period= seas[0], term= term) 
                    data[name + "_cos"] = data["ds"].map(F_compoent.set_index("ds")["fourier_cos"])
                    data[name + "_sin"] = data["ds"].map(F_compoent.set_index("ds")["fourier_sin"])
        else: 
            pass
        return data


    def apply_seaonality(self, data: pd.DataFrame) -> pd.DataFrame:
        seasonalities = self.parse_seasonality(self.seasonality)
        data = self.add_seasonality(seasonalities, data)


    def _build_xgboost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        exogenous = (self.xgb_hyperparameters or {}).get("exogenous", {}) or {}
        numerical = [col for col in exogenous.get("numerical", []) if col in data.columns]
        categorical = [col for col in exogenous.get("categorical", []) if col in data.columns]
        level_method = (self.xgb_hyperparameters or {}).get("level_method", {})
        features = self.apply_level(data.copy(), level_method)
        for col in categorical:
            features[col] = self.apply_encoding(features.copy(), col)
        features = self.apply_seasonality(data.copy())
        return features

    def apply_prophet_model(self, group: pd.DataFrame) -> pd.DataFrame:
        model = ProphetModel(model_kwargs=self._get_prophet_kwargs())
        train_df = group[group["TestIndicator"] == 0].copy()
        future_df = group[group["TestIndicator"] == 1].copy()
        model.fit(train_df)
        forecast = model.predict(future_df=future_df, freq=self._freq_alias())

        return forecast

    def apply_xgboost_model(self, data: pd.DataFrame) -> pd.DataFrame:
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

        return pred_df[ordered_cols]

    def _train(self, data: Any) -> Dict[str, Any]:
        data = self._prepare_data(data)

        artifact: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "frequency": self.frequency,
            "input_data_header": list(data.columns),
            "predictions": None,
        }
        forecast = pd.DataFrame()
        if self.algorithm in ["auto", "xgboost"]:
            self.logger.info("Training XGBoost")
            forecast = self.apply_xgboost_model(data.copy())
            if self.algorithm == "xgboost":
                artifact["predictions"] = forecast
                return artifact

        
        if "TSId" not in data.columns:
                raise KeyError("TSId column must be provided!!")
        forecast_list = []
        for ts_id, group in data.groupby("TSId", sort=False):
            self.logger.info("Training Prophet for TSId=%s", ts_id)
            forecast = self.apply_prophet_model(ts_id, group.copy())
            forecast_list.append(forecast)

        artifact["predictions"] = pd.concat(forecast_list + list(forecast), ignore_index=True)

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