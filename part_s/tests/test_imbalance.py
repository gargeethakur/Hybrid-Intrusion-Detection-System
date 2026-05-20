"""
Tests for the class imbalance handling module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eda.imbalance import diagnose_imbalance, run_balancing_pipeline


def make_imbalanced_df():
    """Simulate severe imbalance: class 0 dominant, class 9 tiny."""
    rng = np.random.default_rng(42)
    rows = []
    counts = {0: 5000, 1: 300, 2: 200, 9: 50}
    for label, n in counts.items():
        data = rng.normal(label, 1, (n, 5))
        for row in data:
            rows.append(list(row) + [label])
    df = pd.DataFrame(rows, columns=[f"f{i}" for i in range(5)] + ["label"])
    return df


def test_diagnose_imbalance_returns_correct_shape():
    df = make_imbalanced_df()
    report = diagnose_imbalance(df)
    assert len(report) == df["label"].nunique()
    assert "imbalance_ratio" in report.columns
    assert "count" in report.columns


def test_diagnose_imbalance_majority_has_ratio_1():
    df = make_imbalanced_df()
    report = diagnose_imbalance(df)
    # Majority class (0) should have imbalance_ratio = 1.0
    majority_row = report[report["class"] == "Benign"]
    assert majority_row["imbalance_ratio"].values[0] == pytest.approx(1.0)


def test_balancing_pipeline_raises_minority_counts():
    df = make_imbalanced_df()
    balanced = run_balancing_pipeline(
        df,
        smote_min_samples=200,
        undersample_max=3000,
        random_state=42
    )
    counts = Counter(balanced["label"])
    # All minority classes should now be at least 200
    for label in [1, 2, 9]:
        assert counts[label] >= 200, f"Class {label} still below threshold: {counts[label]}"


def test_balancing_pipeline_caps_majority():
    df = make_imbalanced_df()
    balanced = run_balancing_pipeline(
        df,
        smote_min_samples=200,
        undersample_max=1000,
        random_state=42
    )
    counts = Counter(balanced["label"])
    assert counts[0] <= 1000, f"Majority class not capped: {counts[0]}"


def test_balancing_pipeline_preserves_features():
    df = make_imbalanced_df()
    balanced = run_balancing_pipeline(df, smote_min_samples=200, undersample_max=3000)
    # All original feature columns should still be present
    for col in [f"f{i}" for i in range(5)]:
        assert col in balanced.columns
