"""
Encoders: Smart encoding strategies for tabular ML.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes cyclical/periodic features using sine and cosine transforms.

    Handles columns like hour (0-23), month (1-12), day-of-week (0-6),
    angle (0-360), or any periodic numeric variable — ensuring the model
    understands that value 0 and value 23 (for hours) are close together.

    Parameters
    ----------
    columns : dict
        Mapping of column name → period (max value of the cycle).
        Example: {"hour": 24, "month": 12, "day_of_week": 7}
    drop_original : bool, default=True
        Whether to drop the original columns after encoding.

    Attributes
    ----------
    columns_ : dict
        Validated column → period mapping from fit.

    Examples
    --------
    >>> from autofeature import CyclicalEncoder
    >>> enc = CyclicalEncoder(columns={"hour": 24, "month": 12})
    >>> X_enc = enc.fit_transform(X)
    >>> # Produces: hour_sin, hour_cos, month_sin, month_cos
    """

    def __init__(
        self,
        columns: Optional[Dict[str, float]] = None,
        drop_original: bool = True,
    ) -> None:
        self.columns = columns or {}
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None) -> "CyclicalEncoder":
        X = self._validate(X)
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        self.columns_ = self.columns.copy()
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        X = self._validate(X)
        X_out = X.copy()

        for col, period in self.columns_.items():
            if col not in X_out.columns:
                raise ValueError(f"Column '{col}' missing in transform input.")
            X_out[f"{col}_sin"] = np.sin(2 * np.pi * X_out[col] / period)
            X_out[f"{col}_cos"] = np.cos(2 * np.pi * X_out[col] / period)

        if self.drop_original:
            X_out = X_out.drop(columns=list(self.columns_.keys()))

        return X_out

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X)}")
        return X


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Automatically selects the best encoding strategy per categorical column.

    Strategy selection rules:
    - Binary columns (2 unique values) → Label encoding (0/1)
    - Low cardinality columns (≤ max_onehot_cardinality) → One-hot encoding
    - High cardinality columns → Target (mean) encoding with smoothing

    Parameters
    ----------
    max_onehot_cardinality : int, default=10
        Columns with more unique values than this get target-encoded.
    smoothing : float, default=1.0
        Smoothing factor for target encoding (higher = more regularisation).
    task : str, default="auto"
        "classification" or "regression". Only affects target encoding.
    handle_unknown : str, default="mean"
        How to handle unseen categories at transform time: "mean" or "zero".

    Attributes
    ----------
    encoding_map_ : dict
        Per-column encoding strategy chosen during fit.
    target_encodings_ : dict
        For target-encoded columns: mapping of category → encoded value.
    onehot_categories_ : dict
        For one-hot columns: list of categories seen during fit.

    Examples
    --------
    >>> from autofeature import SmartCategoricalEncoder
    >>> enc = SmartCategoricalEncoder()
    >>> X_enc = enc.fit_transform(X, y)
    """

    def __init__(
        self,
        max_onehot_cardinality: int = 10,
        smoothing: float = 1.0,
        task: str = "auto",
        handle_unknown: str = "mean",
    ) -> None:
        self.max_onehot_cardinality = max_onehot_cardinality
        self.smoothing = smoothing
        self.task = task
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y=None) -> "SmartCategoricalEncoder":
        X = self._validate(X)
        self.feature_names_in_ = list(X.columns)
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)

        self.encoding_map_: Dict[str, str] = {}
        self.target_encodings_: Dict[str, Dict] = {}
        self.onehot_categories_: Dict[str, List] = {}
        self._label_maps_: Dict[str, Dict] = {}
        self._global_mean_: Optional[float] = None

        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            self._global_mean_ = float(np.mean(y_arr))

        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique <= 2:
                self.encoding_map_[col] = "label"
                uniques = list(X[col].dropna().unique())
                self._label_maps_[col] = {v: i for i, v in enumerate(uniques)}
            elif n_unique <= self.max_onehot_cardinality:
                self.encoding_map_[col] = "onehot"
                self.onehot_categories_[col] = list(X[col].dropna().unique())
            else:
                self.encoding_map_[col] = "target"
                if y is None:
                    raise ValueError(
                        f"Target encoding for '{col}' requires y. "
                        "Pass y to fit()."
                    )
                self.target_encodings_[col] = self._compute_target_encoding(
                    X[col], y_arr
                )

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        check_is_fitted(self, "encoding_map_")
        X = self._validate(X)
        X_out = X.copy()
        to_drop = []

        for col, strategy in self.encoding_map_.items():
            if col not in X_out.columns:
                raise ValueError(f"Column '{col}' missing in transform.")

            if strategy == "label":
                mapping = self._label_maps_[col]
                fallback = -1
                X_out[col] = X_out[col].map(mapping).fillna(fallback).astype(int)

            elif strategy == "onehot":
                for cat in self.onehot_categories_[col]:
                    X_out[f"{col}__{cat}"] = (X_out[col] == cat).astype(int)
                to_drop.append(col)

            elif strategy == "target":
                enc = self.target_encodings_[col]
                fallback = (
                    self._global_mean_
                    if self.handle_unknown == "mean"
                    else 0.0
                )
                X_out[col] = (
                    X_out[col].map(enc).fillna(fallback).astype(float)
                )

        X_out = X_out.drop(columns=to_drop)
        return X_out

    def _compute_target_encoding(
        self, col: pd.Series, y: np.ndarray
    ) -> Dict:
        global_mean = np.mean(y)
        df = pd.DataFrame({"col": col, "y": y})
        stats = df.groupby("col")["y"].agg(["mean", "count"])
        # Smoothed target encoding
        smoothed = (
            stats["count"] * stats["mean"] + self.smoothing * global_mean
        ) / (stats["count"] + self.smoothing)
        return smoothed.to_dict()

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X)}")
        return X
