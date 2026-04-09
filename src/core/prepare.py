from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.hyperparameters import Hyperparameters
from ..utils.settings import HYPERPARAMETERS_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH


class Prepare:
    """Shared data and configuration loading/validation for training and prediction."""

    @staticmethod
    def read_dataframe(data_path: str | Path) -> pd.DataFrame:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            data = pd.read_parquet(path)
        else:
            raise ValueError("Supported input formats are .csv and .parquet")

        if data.empty:
            raise ValueError(f"Input data file is empty: {path}")

        data.columns = [str(column).strip() for column in data.columns]
        return data

    @staticmethod
    def load_training_data(data_path: str | Path = TRAIN_DATA_PATH) -> pd.DataFrame:
        data = Prepare.read_dataframe(data_path)
        return Prepare.prepare_training_data(data)

    @staticmethod
    def load_prediction_data(data_path: str | Path = TEST_DATA_PATH) -> pd.DataFrame:
        data = Prepare.read_dataframe(data_path)
        return Prepare.prepare_prediction_data(data)

    @staticmethod
    def load_hyperparameters(path: str | Path = HYPERPARAMETERS_PATH) -> dict[str, Any]:
        parsed = Hyperparameters.parse(str(path))
        if is_dataclass(parsed):
            return asdict(parsed)
        if hasattr(parsed, "to_dict"):
            return parsed.to_dict()
        if hasattr(parsed, "data"):
            return dict(parsed.data)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
        return Prepare._prepare_dataframe(data, require_target=True)

    @staticmethod
    def prepare_prediction_data(data: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
        return Prepare._prepare_dataframe(data, require_target=require_target)

    @staticmethod
    def validate_prophet_input(data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        required_columns = {"ds"}
        if require_target:
            required_columns.add("y")
        return Prepare._validate_required_columns(data, required_columns)

    @staticmethod
    def validate_xgboost_input(data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        required_columns = {"ds"}
        if require_target:
            required_columns.add("y")
        return Prepare._validate_required_columns(data, required_columns)

    @staticmethod
    def _prepare_dataframe(data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("data must not be empty")

        prepared = data.copy()
        prepared.columns = [str(column).strip() for column in prepared.columns]
        prepared = Prepare._validate_required_columns(prepared, {"ds"} | ({"y"} if require_target else set()))

        prepared["ds"] = pd.to_datetime(prepared["ds"], errors="coerce")
        if prepared["ds"].isna().any():
            raise ValueError("Column 'ds' contains invalid datetime values.")

        if getattr(prepared["ds"].dt, "tz", None) is not None:
            prepared["ds"] = prepared["ds"].dt.tz_localize(None)

        if "y" in prepared.columns:
            prepared["y"] = pd.to_numeric(prepared["y"], errors="coerce")
            if require_target and prepared["y"].isna().any():
                raise ValueError("Column 'y' contains invalid numeric values.")

        sort_columns = [column for column in ["TSId", "ds"] if column in prepared.columns]
        prepared = prepared.sort_values(sort_columns).reset_index(drop=True)
        return prepared

    @staticmethod
    def _validate_required_columns(data: pd.DataFrame, required_columns: set[str]) -> pd.DataFrame:
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return data
