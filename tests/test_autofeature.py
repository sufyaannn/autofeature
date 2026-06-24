"""
Test suite for AutoFeature library.
"""

import numpy as np
import pandas as pd
import pytest

from autofeature import (
    AutoFeatureEngineer,
    AutoFeaturePipeline,
    CyclicalEncoder,
    LeakageDetector,
    SmartCategoricalEncoder,
    TargetAwareSelector,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "age": rng.randint(18, 70, n).astype(float),
        "income": rng.exponential(50000, n),
        "score": rng.uniform(0, 1, n),
        "category": rng.choice(["A", "B", "C"], n),
        "binary_flag": rng.choice(["yes", "no"], n),
    })
    y = pd.Series((X["income"] > 50000).astype(int))
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(0)
    n = 150
    X = pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "x3": rng.randn(n),
        "hour": rng.randint(0, 24, n).astype(float),
    })
    y = pd.Series(2 * X["x1"] - X["x2"] + rng.randn(n) * 0.1)
    return X, y


# ─── AutoFeatureEngineer ─────────────────────────────────────────────────────

class TestAutoFeatureEngineer:

    def test_fit_transform_adds_columns(self, classification_data):
        X, y = classification_data
        afe = AutoFeatureEngineer(n_estimators=10, random_state=42)
        X_out = afe.fit_transform(X.select_dtypes(include=[np.number]), y)
        assert isinstance(X_out, pd.DataFrame)
        assert X_out.shape[1] >= X.select_dtypes(include=[np.number]).shape[1]

    def test_transform_without_fit_raises(self, classification_data):
        X, _ = classification_data
        afe = AutoFeatureEngineer()
        with pytest.raises(Exception):
            afe.transform(X)

    def test_non_dataframe_raises(self):
        afe = AutoFeatureEngineer()
        with pytest.raises(TypeError):
            afe.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_interaction_report(self, classification_data):
        X, y = classification_data
        afe = AutoFeatureEngineer(n_estimators=10)
        afe.fit(X.select_dtypes(include=[np.number]), y)
        report = afe.get_interaction_report()
        assert isinstance(report, pd.DataFrame)
        assert set(report.columns) == {"feature_name", "col_a", "col_b", "interaction_type"}

    def test_invalid_interaction_type(self, classification_data):
        X, y = classification_data
        afe = AutoFeatureEngineer(interaction_types=["invalid_type"])
        with pytest.raises(ValueError):
            afe.fit(X.select_dtypes(include=[np.number]), y)

    def test_max_interaction_features_respected(self, regression_data):
        X, y = regression_data
        afe = AutoFeatureEngineer(max_interaction_features=2, n_estimators=10)
        afe.fit(X, y)
        assert len(afe.selected_interactions_) <= 2


# ─── TargetAwareSelector ─────────────────────────────────────────────────────

class TestTargetAwareSelector:

    def test_selects_k_features(self, regression_data):
        X, y = regression_data
        sel = TargetAwareSelector(k=2)
        X_out = sel.fit_transform(X, y)
        assert X_out.shape[1] == 2

    def test_scores_are_non_negative(self, classification_data):
        X, y = classification_data
        X_num = X.select_dtypes(include=[np.number])
        sel = TargetAwareSelector(k="all")
        sel.fit(X_num, y)
        assert (sel.scores_ >= 0).all()

    def test_threshold_mode(self, regression_data):
        X, y = regression_data
        sel = TargetAwareSelector(threshold=0.0)
        X_out = sel.fit_transform(X, y)
        assert X_out.shape[1] >= 1

    def test_feature_scores_report(self, regression_data):
        X, y = regression_data
        sel = TargetAwareSelector(k=2)
        sel.fit(X, y)
        df = sel.get_feature_scores()
        assert "feature" in df.columns
        assert "selected" in df.columns
        assert df["selected"].sum() == 2


# ─── CyclicalEncoder ─────────────────────────────────────────────────────────

class TestCyclicalEncoder:

    def test_produces_sin_cos(self):
        X = pd.DataFrame({"hour": [0, 6, 12, 18, 23]})
        enc = CyclicalEncoder(columns={"hour": 24})
        X_out = enc.fit_transform(X)
        assert "hour_sin" in X_out.columns
        assert "hour_cos" in X_out.columns
        assert "hour" not in X_out.columns  # dropped

    def test_keeps_original_if_not_drop(self):
        X = pd.DataFrame({"month": [1, 6, 12]})
        enc = CyclicalEncoder(columns={"month": 12}, drop_original=False)
        X_out = enc.fit_transform(X)
        assert "month" in X_out.columns
        assert "month_sin" in X_out.columns

    def test_cyclical_values_in_range(self):
        X = pd.DataFrame({"hour": list(range(24))})
        enc = CyclicalEncoder(columns={"hour": 24})
        X_out = enc.fit_transform(X)
        assert X_out["hour_sin"].between(-1, 1).all()
        assert X_out["hour_cos"].between(-1, 1).all()

    def test_missing_column_raises(self):
        X = pd.DataFrame({"day": [1, 2, 3]})
        enc = CyclicalEncoder(columns={"hour": 24})
        with pytest.raises(ValueError):
            enc.fit(X)


# ─── SmartCategoricalEncoder ─────────────────────────────────────────────────

class TestSmartCategoricalEncoder:

    def test_binary_label_encoded(self):
        X = pd.DataFrame({"flag": ["yes", "no", "yes", "no"]})
        y = pd.Series([1, 0, 1, 0])
        enc = SmartCategoricalEncoder()
        X_out = enc.fit_transform(X, y)
        assert X_out["flag"].dtype in [int, np.int64, np.int32]
        assert set(X_out["flag"].unique()).issubset({0, 1})

    def test_low_cardinality_onehot(self):
        X = pd.DataFrame({"color": ["red", "blue", "green", "red", "blue"]})
        y = pd.Series([0, 1, 0, 1, 0])
        enc = SmartCategoricalEncoder(max_onehot_cardinality=10)
        X_out = enc.fit_transform(X, y)
        assert "color__red" in X_out.columns
        assert "color" not in X_out.columns

    def test_high_cardinality_target_encoded(self):
        rng = np.random.RandomState(1)
        cats = [f"cat_{i}" for i in range(15)]
        X = pd.DataFrame({"city": rng.choice(cats, 100)})
        y = pd.Series(rng.randn(100))
        enc = SmartCategoricalEncoder(max_onehot_cardinality=10)
        X_out = enc.fit_transform(X, y)
        assert "city" in X_out.columns
        assert X_out["city"].dtype == float

    def test_no_categoricals_passthrough(self, regression_data):
        X, y = regression_data
        enc = SmartCategoricalEncoder()
        X_out = enc.fit_transform(X, y)
        assert list(X_out.columns) == list(X.columns)


# ─── LeakageDetector ─────────────────────────────────────────────────────────

class TestLeakageDetector:

    def test_detects_high_correlation(self):
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "leaky": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfect correlation
        })
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ld = LeakageDetector(verbose=False)
        ld.fit(X, y)
        assert "leaky" in ld.leaky_columns_

    def test_detects_name_pattern(self):
        X = pd.DataFrame({
            "feature": [1, 2, 3],
            "target_value": [10, 20, 30],
        })
        y = pd.Series([0, 1, 0])
        ld = LeakageDetector(verbose=False)
        ld.fit(X, y)
        assert "target_value" in ld.leaky_columns_

    def test_remove_leaky_drops_columns(self):
        rng = np.random.RandomState(7)
        X = pd.DataFrame({
            "ok": rng.randn(50),               # uncorrelated
            "leaky_col": np.arange(50, dtype=float),  # perfectly correlated
        })
        y = pd.Series(np.arange(50, dtype=float))
        ld = LeakageDetector(verbose=False, correlation_threshold=0.95)
        ld.fit(X, y)
        X_clean = ld.remove_leaky(X)
        assert "leaky_col" not in X_clean.columns
        assert "ok" in X_clean.columns

    def test_no_leakage_clean_data(self, regression_data):
        X, y = regression_data
        ld = LeakageDetector(verbose=False)
        ld.fit(X, y)
        assert isinstance(ld.warnings_, list)


# ─── AutoFeaturePipeline ─────────────────────────────────────────────────────

class TestAutoFeaturePipeline:

    def test_full_pipeline_classification(self, classification_data):
        X, y = classification_data
        pipeline = AutoFeaturePipeline(k=5, verbose=False)
        X_out = pipeline.fit_transform(X, y)
        assert isinstance(X_out, pd.DataFrame)
        assert X_out.shape[1] <= 5

    def test_full_pipeline_with_cyclical(self, regression_data):
        X, y = regression_data
        pipeline = AutoFeaturePipeline(
            cyclical_columns={"hour": 24},
            k=4,
            verbose=False,
        )
        X_out = pipeline.fit_transform(X, y)
        assert isinstance(X_out, pd.DataFrame)

    def test_transform_consistent_with_fit_transform(self, classification_data):
        X, y = classification_data
        pipeline = AutoFeaturePipeline(k=5, verbose=False)
        X_fit_transform = pipeline.fit_transform(X, y)
        X_transform = pipeline.transform(X)
        assert list(X_fit_transform.columns) == list(X_transform.columns)

    def test_get_summary(self, classification_data):
        X, y = classification_data
        pipeline = AutoFeaturePipeline(k=5, verbose=False)
        pipeline.fit(X, y)
        summary = pipeline.get_summary()
        assert "selected_features" in summary
        assert "interaction_features" in summary
        assert "categorical_strategies" in summary

    def test_non_dataframe_raises(self):
        pipeline = AutoFeaturePipeline()
        with pytest.raises(TypeError):
            pipeline.fit(np.array([[1, 2]]), np.array([1]))
