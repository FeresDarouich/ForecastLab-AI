from typing import Any, Dict, Optional, Union
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor



class XGBoostModel:
    """
    Thin wrapper around xgboost.XGBRegressor to standardize fit / predict / save / load.
    - fit accepts either (X, y) or a single DataFrame containing a 'y' column.
    - predict returns numpy array of predictions.
    """

    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        model_kwargs = model_kwargs or {}
        if seed is not None:
            model_kwargs.setdefault("random_state", seed)
        self._model = XGBRegressor(**model_kwargs)
        self._is_fitted = False

    @property
    def model(self) -> XGBRegressor:
        return self._model

    def fit(
        self,
        X: Union[pd.DataFrame, Any],
        y: Optional[Union[pd.Series, Any]] = None,
        **fit_kwargs,
    ) -> "XGBoostModel":
        """
        Fit the internal XGBRegressor.
        Usage:
        - fit(X_df_with_y) where X_df_with_y contains column 'y'
        - fit(X, y)
        """
        if y is None:
            if isinstance(X, pd.DataFrame) and "y" in X.columns:
                df = X.copy()
                y = df.pop("y")
                X = df
            else:
                raise ValueError("Provide y or pass a DataFrame containing a 'y' column.")
        # let XGBoost handle numpy / DataFrame conversion
        self._model.fit(X, y, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, Any]) -> Any:
        """
        Return model predictions for X. Model must be fitted first.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._model.predict(X)

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist wrapper + internal model to disk using pickle.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> "XGBoostModel":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, XGBoostModel):
            raise TypeError("Loaded object is not an XGBoostModel instance.")
        return obj

    def set_seed(self, seed: int) -> None:
        """
        Recreate internal estimator with the same init params but updated random_state.
        Fitted state is cleared.
        """
        params = {}
        try:
            params = self._model.get_params()
        except Exception:
            params = {}
        params["random_state"] = seed
        # recreate model with same parameters
        self._model = XGBRegressor(**params)
        self._is_fitted = False