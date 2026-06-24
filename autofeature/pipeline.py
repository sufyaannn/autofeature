"""
AutoFeaturePipeline: End-to-end feature engineering pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .detector import LeakageDetector
from .encoders import CyclicalEncoder, SmartCategoricalEncoder
from .engineer import AutoFeatureEngineer
from .selector import TargetAwareSelector


class AutoFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Full end-to-end AutoFeature pipeline.

    Executes all AutoFeature steps in sequence:
    1. Leakage detection (optional, fit only — warns but doesn't drop by default)
    2. Smart categorical encoding
    3. Cyclical encoding (if cyclical_columns provided)
    4. Interaction feature generation
    5. Target-aware feature selection

    Parameters
    ----------
    cyclical_columns : dict or None
        Mapping of column → period for cyclical encoding. e.g. {"hour": 24}
    max_interaction_features : int, default=20
    k : int or "all", default=20
        Number of features to select at the end.
    task : str, default="auto"
    detect_leakage : bool, default=True
        Whether to run leakage detection during fit (warns, does not drop).
    remove_leaky : bool, default=False
        If True AND detect_leakage=True, removes flagged leaky columns.
    random_state : int or None, default=42
    verbose : bool, default=False

    Examples
    --------
    >>> from autofeature import AutoFeaturePipeline
    >>> pipeline = AutoFeaturePipeline(cyclical_columns={"hour": 24}, k=15)
    >>> X_out = pipeline.fit_transform(X_train, y_train)
    >>> X_test_out = pipeline.transform(X_test)
    """

    def __init__(
        self,
        cyclical_columns: Optional[Dict[str, float]] = None,
        max_interaction_features: int = 20,
        k: int | str = 20,
        task: str = "auto",
        detect_leakage: bool = True,
        remove_leaky: bool = False,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.cyclical_columns = cyclical_columns
        self.max_interaction_features = max_interaction_features
        self.k = k
        self.task = task
        self.detect_leakage = detect_leakage
        self.remove_leaky = remove_leaky
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoFeaturePipeline":
        X = self._check_df(X)

        # Step 1: Leakage detection
        self.leakage_detector_: Optional[LeakageDetector] = None
        if self.detect_leakage:
            self.leakage_detector_ = LeakageDetector(verbose=self.verbose)
            self.leakage_detector_.fit(X, y)
            if self.remove_leaky:
                X = self.leakage_detector_.remove_leaky(X)

        # Step 2: Categorical encoding
        self.cat_encoder_ = SmartCategoricalEncoder(task=self.task)
        X = self.cat_encoder_.fit_transform(X, y)

        # Step 3: Cyclical encoding
        self.cyc_encoder_: Optional[CyclicalEncoder] = None
        if self.cyclical_columns:
            valid_cols = {
                k: v for k, v in self.cyclical_columns.items() if k in X.columns
            }
            if valid_cols:
                self.cyc_encoder_ = CyclicalEncoder(columns=valid_cols)
                X = self.cyc_encoder_.fit_transform(X)

        # Step 4: Interaction features
        self.engineer_ = AutoFeatureEngineer(
            max_interaction_features=self.max_interaction_features,
            task=self.task,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        X = self.engineer_.fit_transform(X, y)

        # Step 5: Target-aware selection
        self.selector_ = TargetAwareSelector(
            k=self.k,
            task=self.task,
            random_state=self.random_state,
        )
        self.selector_.fit(X, y)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        X = self._check_df(X)

        if self.remove_leaky and self.leakage_detector_ is not None:
            X = self.leakage_detector_.remove_leaky(X)

        X = self.cat_encoder_.transform(X)

        if self.cyc_encoder_ is not None:
            X = self.cyc_encoder_.transform(X)

        X = self.engineer_.transform(X)
        X = self.selector_.transform(X)

        return X

    def get_summary(self) -> dict:
        """
        Returns a summary of what the pipeline did.

        Returns
        -------
        dict with keys: leaky_columns, categorical_strategies,
                        interaction_features, selected_features
        """
        check_is_fitted(self, "is_fitted_")
        return {
            "leaky_columns": (
                self.leakage_detector_.leaky_columns_
                if self.leakage_detector_
                else []
            ),
            "categorical_strategies": (
                self.cat_encoder_.encoding_map_
                if hasattr(self.cat_encoder_, "encoding_map_")
                else {}
            ),
            "interaction_features": self.engineer_.get_interaction_report().to_dict(
                orient="records"
            ),
            "selected_features": self.selector_.selected_features_,
        }

    @staticmethod
    def _check_df(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X)}")
        return X
