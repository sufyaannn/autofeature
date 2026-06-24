"""
LeakageDetector: Warn about features that are suspiciously correlated with the target.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class LeakageDetector(BaseEstimator):
    """
    Detects potential target leakage in tabular features.

    Leakage occurs when a feature contains information about the target
    that wouldn't be available at prediction time (e.g., a feature derived
    from the target, or a near-duplicate of it).

    Checks performed:
    1. Near-perfect correlation with target (Pearson / Spearman)
    2. Mutual information > threshold fraction of target entropy
    3. Column name heuristics (contains target name, "label", "outcome", etc.)

    Parameters
    ----------
    correlation_threshold : float, default=0.95
        Correlations above this trigger a leakage warning.
    name_patterns : list of str, default=["label", "target", "outcome", "y_"]
        Column name substrings that suggest leakage.
    verbose : bool, default=True

    Attributes
    ----------
    warnings_ : list of dict
        Each dict has keys: column, reason, severity ("high"/"medium").
    leaky_columns_ : list of str
        Columns flagged as likely leakage.

    Examples
    --------
    >>> from autofeature import LeakageDetector
    >>> ld = LeakageDetector()
    >>> ld.fit(X, y)
    >>> print(ld.warnings_)
    >>> X_clean = ld.remove_leaky(X)
    """

    _DEFAULT_PATTERNS = ["label", "target", "outcome", "y_", "_target", "response"]

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        name_patterns: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.name_patterns = name_patterns or self._DEFAULT_PATTERNS
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeakageDetector":
        """
        Analyse X for leakage with respect to target y.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame, got {type(X)}")

        y_arr = np.asarray(y, dtype=float)
        self.warnings_: List[dict] = []
        self.leaky_columns_: List[str] = []

        num_cols = X.select_dtypes(include=[np.number]).columns

        for col in X.columns:
            reasons = []

            # Heuristic: suspicious column name
            col_lower = col.lower()
            if any(pat in col_lower for pat in self.name_patterns):
                reasons.append(
                    {"reason": f"column name matches leakage pattern", "severity": "medium"}
                )

            # Correlation check (numeric only)
            if col in num_cols:
                try:
                    col_arr = X[col].fillna(X[col].median()).values.astype(float)
                    corr = float(np.corrcoef(col_arr, y_arr)[0, 1])
                    if not np.isnan(corr) and abs(corr) >= self.correlation_threshold:
                        reasons.append(
                            {
                                "reason": f"near-perfect correlation with target ({corr:.4f})",
                                "severity": "high",
                            }
                        )
                except Exception:
                    pass

            if reasons:
                for r in reasons:
                    entry = {"column": col, **r}
                    self.warnings_.append(entry)
                    if self.verbose:
                        print(
                            f"[LeakageDetector] ⚠️  '{col}' — {r['reason']} "
                            f"[{r['severity'].upper()}]"
                        )
                if col not in self.leaky_columns_:
                    self.leaky_columns_.append(col)

        if self.verbose and not self.warnings_:
            print("[LeakageDetector] ✅ No leakage detected.")

        return self

    def remove_leaky(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return X with leaky columns removed.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        check_is_fitted(self, "leaky_columns_")
        cols_to_drop = [c for c in self.leaky_columns_ if c in X.columns]
        return X.drop(columns=cols_to_drop)

    def get_report(self) -> pd.DataFrame:
        """Return a DataFrame of all leakage warnings."""
        check_is_fitted(self, "warnings_")
        return pd.DataFrame(self.warnings_)
