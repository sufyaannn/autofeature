"""
AutoFeatureEngineer: Automatic interaction feature detection and generation.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_is_fitted


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automatically detects and generates useful interaction features for tabular data.

    For each pair of numeric columns, evaluates whether their interaction
    (product, ratio, difference) improves predictive signal using a lightweight
    mutual information proxy. Only statistically useful interactions are retained.

    Parameters
    ----------
    max_interaction_features : int, default=20
        Maximum number of interaction features to generate.
    interaction_types : list of str, default=["product", "ratio", "difference"]
        Types of interactions to consider.
    interaction_threshold : float, default=0.01
        Minimum relative importance gain to include an interaction feature.
    n_estimators : int, default=50
        Number of trees used in the internal importance evaluator.
    task : str, default="auto"
        "classification", "regression", or "auto" (inferred from target dtype).
    random_state : int or None, default=42
    verbose : bool, default=False

    Attributes
    ----------
    selected_interactions_ : list of tuples
        Each tuple: (col_a, col_b, interaction_type, feature_name)
    feature_names_in_ : list of str
        Column names seen during fit.
    numeric_cols_ : list of str
        Numeric columns used for interactions.

    Examples
    --------
    >>> import pandas as pd
    >>> from autofeature import AutoFeatureEngineer
    >>> X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    >>> y = pd.Series([0, 1, 0])
    >>> afe = AutoFeatureEngineer()
    >>> X_new = afe.fit_transform(X, y)
    """

    _INTERACTION_TYPES = {"product", "ratio", "difference", "sum"}

    def __init__(
        self,
        max_interaction_features: int = 20,
        interaction_types: Optional[List[str]] = None,
        interaction_threshold: float = 0.01,
        n_estimators: int = 50,
        task: str = "auto",
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.max_interaction_features = max_interaction_features
        self.interaction_types = interaction_types or ["product", "ratio", "difference"]
        self.interaction_threshold = interaction_threshold
        self.n_estimators = n_estimators
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoFeatureEngineer":
        """
        Fit: identify which interaction features provide real signal.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series

        Returns
        -------
        self
        """
        X = self._validate_input(X)
        y = np.asarray(y)

        self.feature_names_in_ = list(X.columns)
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)

        invalid = set(self.interaction_types) - self._INTERACTION_TYPES
        if invalid:
            raise ValueError(
                f"Unknown interaction types: {invalid}. "
                f"Valid options: {self._INTERACTION_TYPES}"
            )

        task = self._infer_task(y) if self.task == "auto" else self.task
        self._task = task

        # Baseline importance of original features
        baseline_importances = self._fit_importances(X[self.numeric_cols_], y, task)

        # Evaluate candidate interactions
        candidates: List[Tuple] = []
        pairs = list(combinations(self.numeric_cols_, 2))

        if self.verbose:
            print(f"[AutoFeature] Evaluating {len(pairs)} column pairs...")

        for col_a, col_b in pairs:
            for itype in self.interaction_types:
                feat = self._compute_interaction(X[col_a], X[col_b], itype)
                if feat is None:
                    continue
                gain = self._estimate_gain(
                    feat, X[self.numeric_cols_], y, task, baseline_importances
                )
                candidates.append((col_a, col_b, itype, gain))

        # Sort by gain and keep top interactions above threshold
        candidates.sort(key=lambda x: x[3], reverse=True)
        self.selected_interactions_ = [
            (a, b, t, f"{a}__{t}__{b}")
            for a, b, t, g in candidates
            if g >= self.interaction_threshold
        ][: self.max_interaction_features]

        if self.verbose:
            print(
                f"[AutoFeature] Selected {len(self.selected_interactions_)} interactions."
            )

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add selected interaction features to X.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame with original + interaction columns
        """
        check_is_fitted(self, "is_fitted_")
        X = self._validate_input(X)
        X_out = X.copy()

        for col_a, col_b, itype, fname in self.selected_interactions_:
            if col_a not in X_out.columns or col_b not in X_out.columns:
                warnings.warn(
                    f"Columns '{col_a}' or '{col_b}' not found in transform input. "
                    f"Skipping '{fname}'.",
                    UserWarning,
                )
                continue
            feat = self._compute_interaction(X_out[col_a], X_out[col_b], itype)
            if feat is not None:
                X_out[fname] = feat

        return X_out

    def get_interaction_report(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarising selected interactions.

        Returns
        -------
        pd.DataFrame with columns: feature_name, col_a, col_b, interaction_type
        """
        check_is_fitted(self, "is_fitted_")
        rows = [
            {
                "feature_name": fname,
                "col_a": col_a,
                "col_b": col_b,
                "interaction_type": itype,
            }
            for col_a, col_b, itype, fname in self.selected_interactions_
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("X is empty.")
        return X

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        if np.issubdtype(y.dtype, np.floating):
            return "regression"
        unique = np.unique(y)
        if len(unique) <= 20:
            return "classification"
        return "regression"

    def _fit_importances(
        self, X: pd.DataFrame, y: np.ndarray, task: str
    ) -> np.ndarray:
        clf = (
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if task == "classification"
            else RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
        )
        clf.fit(X.fillna(X.median(numeric_only=True)), y)
        return clf.feature_importances_

    def _estimate_gain(
        self,
        feat: pd.Series,
        X_num: pd.DataFrame,
        y: np.ndarray,
        task: str,
        baseline_importances: np.ndarray,
    ) -> float:
        """Estimate importance gain of adding this feature."""
        X_aug = X_num.copy()
        X_aug["__candidate__"] = feat.values
        imps = self._fit_importances(X_aug, y, task)
        # Gain = importance assigned to the candidate feature
        return float(imps[-1])

    @staticmethod
    def _compute_interaction(
        a: pd.Series, b: pd.Series, itype: str
    ) -> Optional[pd.Series]:
        """Compute a single interaction between two series."""
        with np.errstate(divide="ignore", invalid="ignore"):
            if itype == "product":
                return a * b
            elif itype == "ratio":
                denom = b.replace(0, np.nan)
                result = a / denom
                if result.isna().all():
                    return None
                return result
            elif itype == "difference":
                return a - b
            elif itype == "sum":
                return a + b
        return None
