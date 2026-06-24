"""
TargetAwareSelector: Select features based on their relationship with the target.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_is_fitted


class TargetAwareSelector(BaseEstimator, TransformerMixin):
    """
    Selects features based on mutual information with the target variable.

    Unlike variance-based selectors, this is *target-aware*: it measures how
    much each feature actually tells us about what we're predicting.

    Parameters
    ----------
    k : int or "all", default=10
        Number of top features to keep. Use "all" to rank without dropping.
    task : str, default="auto"
        "classification", "regression", or "auto".
    threshold : float or None, default=None
        If set, keep only features with MI score >= threshold (overrides k).
    random_state : int or None, default=42

    Attributes
    ----------
    scores_ : pd.Series
        Mutual information scores indexed by feature name.
    selected_features_ : list of str
        Features selected after fit.

    Examples
    --------
    >>> from autofeature import TargetAwareSelector
    >>> sel = TargetAwareSelector(k=5)
    >>> X_sel = sel.fit_transform(X, y)
    >>> print(sel.scores_)
    """

    def __init__(
        self,
        k: int | str = 10,
        task: str = "auto",
        threshold: Optional[float] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        self.k = k
        self.task = task
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetAwareSelector":
        X = self._validate(X)
        y_arr = np.asarray(y)
        task = self._infer_task(y_arr) if self.task == "auto" else self.task

        X_filled = X.select_dtypes(include=[np.number]).fillna(
            X.select_dtypes(include=[np.number]).median()
        )
        self.feature_names_in_ = list(X_filled.columns)

        if task == "classification":
            scores = mutual_info_classif(
                X_filled, y_arr, random_state=self.random_state
            )
        else:
            scores = mutual_info_regression(
                X_filled, y_arr, random_state=self.random_state
            )

        self.scores_ = pd.Series(scores, index=self.feature_names_in_).sort_values(
            ascending=False
        )

        if self.threshold is not None:
            self.selected_features_ = list(
                self.scores_[self.scores_ >= self.threshold].index
            )
        elif self.k == "all":
            self.selected_features_ = list(self.scores_.index)
        else:
            self.selected_features_ = list(self.scores_.head(int(self.k)).index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        X = self._validate(X)
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Columns missing in transform: {missing}")
        return X[self.selected_features_]

    def get_feature_scores(self) -> pd.DataFrame:
        """Return a ranked DataFrame of feature scores."""
        check_is_fitted(self, "scores_")
        return pd.DataFrame(
            {
                "feature": self.scores_.index,
                "mutual_info_score": self.scores_.values,
                "selected": [
                    f in self.selected_features_ for f in self.scores_.index
                ],
            }
        ).reset_index(drop=True)

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X)}")
        return X

    @staticmethod
    def _infer_task(y: np.ndarray) -> str:
        if np.issubdtype(y.dtype, np.floating):
            return "regression"
        return "classification" if len(np.unique(y)) <= 20 else "regression"
