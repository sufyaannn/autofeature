# AutoFeature

**Intelligent automatic feature engineering for tabular ML.**

[![PyPI version](https://badge.fury.io/py/autofeature.svg)](https://badge.fury.io/py/autofeature)
[![Python](https://img.shields.io/pypi/pyversions/autofeature.svg)](https://pypi.org/project/autofeature/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is AutoFeature?

AutoFeature is a scikit-learn compatible library that automates the most impactful parts of tabular feature engineering:

| Component | What it does |
|---|---|
| `AutoFeatureEngineer` | Detects and generates useful interaction features (products, ratios, differences) using importance-guided search |
| `TargetAwareSelector` | Selects features by mutual information with the target — not just variance |
| `CyclicalEncoder` | Encodes periodic variables (hour, month, day) with sin/cos to preserve cyclical structure |
| `SmartCategoricalEncoder` | Automatically picks the right encoding per column: label / one-hot / target encoding |
| `LeakageDetector` | Warns about features that suspiciously correlate with the target |
| `AutoFeaturePipeline` | Runs everything end-to-end in one call |

---

## Installation

```bash
pip install sufyaan-autofeature
```

Requires Python ≥ 3.8, scikit-learn ≥ 1.0, pandas ≥ 1.3, numpy ≥ 1.21.

---

## Quickstart

### Full Pipeline (recommended)

```python
import pandas as pd
from autofeature import AutoFeaturePipeline

pipeline = AutoFeaturePipeline(
    cyclical_columns={"hour": 24, "month": 12},
    max_interaction_features=15,
    k=20,                  # keep top 20 features
    task="classification",
    verbose=True,
)

X_train_out = pipeline.fit_transform(X_train, y_train)
X_test_out  = pipeline.transform(X_test)

print(pipeline.get_summary())
```

### Individual Components

```python
from autofeature import (
    AutoFeatureEngineer,
    TargetAwareSelector,
    CyclicalEncoder,
    SmartCategoricalEncoder,
    LeakageDetector,
)

# 1. Detect leakage
ld = LeakageDetector()
ld.fit(X_train, y_train)
X_train = ld.remove_leaky(X_train)

# 2. Encode categoricals automatically
enc = SmartCategoricalEncoder()
X_train = enc.fit_transform(X_train, y_train)
X_test  = enc.transform(X_test)

# 3. Encode cyclical columns
cyc = CyclicalEncoder(columns={"hour": 24, "day_of_week": 7})
X_train = cyc.fit_transform(X_train)
X_test  = cyc.transform(X_test)

# 4. Generate interaction features
afe = AutoFeatureEngineer(max_interaction_features=20)
X_train = afe.fit_transform(X_train, y_train)
X_test  = afe.transform(X_test)

# See what interactions were selected
print(afe.get_interaction_report())

# 5. Select top features by target mutual information
sel = TargetAwareSelector(k=15)
X_train = sel.fit_transform(X_train, y_train)
X_test  = sel.transform(X_test)

print(sel.get_feature_scores())
```

---

## API Reference

### `AutoFeatureEngineer`

```python
AutoFeatureEngineer(
    max_interaction_features=20,   # max interactions to add
    interaction_types=["product", "ratio", "difference"],
    interaction_threshold=0.01,    # minimum importance gain
    n_estimators=50,               # trees in internal evaluator
    task="auto",                   # "classification" | "regression" | "auto"
    random_state=42,
    verbose=False,
)
```

Methods: `fit(X, y)`, `transform(X)`, `fit_transform(X, y)`, `get_interaction_report()`

### `TargetAwareSelector`

```python
TargetAwareSelector(
    k=10,             # number of features to keep, or "all"
    task="auto",
    threshold=None,   # MI threshold (overrides k if set)
    random_state=42,
)
```

Methods: `fit(X, y)`, `transform(X)`, `fit_transform(X, y)`, `get_feature_scores()`

### `CyclicalEncoder`

```python
CyclicalEncoder(
    columns={"hour": 24, "month": 12},  # column → period mapping
    drop_original=True,
)
```

Produces `{col}_sin` and `{col}_cos` columns.

### `SmartCategoricalEncoder`

```python
SmartCategoricalEncoder(
    max_onehot_cardinality=10,   # >10 unique values → target encoding
    smoothing=1.0,               # regularisation for target encoding
    task="auto",
    handle_unknown="mean",       # or "zero"
)
```

### `LeakageDetector`

```python
LeakageDetector(
    correlation_threshold=0.95,
    name_patterns=["label", "target", "outcome"],
    verbose=True,
)
```

Methods: `fit(X, y)`, `remove_leaky(X)`, `get_report()`

### `AutoFeaturePipeline`

```python
AutoFeaturePipeline(
    cyclical_columns=None,
    max_interaction_features=20,
    k=20,
    task="auto",
    detect_leakage=True,
    remove_leaky=False,
    random_state=42,
    verbose=False,
)
```

Methods: `fit(X, y)`, `transform(X)`, `fit_transform(X, y)`, `get_summary()`

---

## Why AutoFeature?

- **Target-aware**: selections and interactions are evaluated against the actual prediction target, not generic statistics
- **Scikit-learn compatible**: works with `Pipeline`, `GridSearchCV`, and any estimator
- **Production-safe**: `fit` on train, `transform` on test — no leakage from transform step
- **Interpretable**: every decision (which interaction, which encoding, which feature) is inspectable

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

```bash
git clone https://github.com/yourusername/autofeature
cd autofeature
pip install -e ".[dev]"
pytest
```

---

## License

MIT
