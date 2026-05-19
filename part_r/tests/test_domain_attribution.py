"""
Tests for domain attribution computation.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from shap_analysis.shap_runner import compute_domain_attribution


def test_attribution_percentages_sum_to_100():
    """CS% + DS% + Other% should be 100 for each class."""
    # Minimal mock: 2 samples, 3 features (cs_ddos_flag, ds_anomaly_score, other)
    X = pd.DataFrame({
        "cs_ddos_flag": [1.0, 0.0],
        "ds_anomaly_score": [0.0, 1.0],
        "other_feature": [0.5, 0.5],
    })
    y_true = pd.Series([3, 3])  # Both DoS
    # Mock shap_values: list of arrays (one per class), only class 3 matters
    mock_shap = [np.zeros((2, 3))] * 10
    mock_shap[3] = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])

    result = compute_domain_attribution(mock_shap, X, y_true)
    assert len(result) == 1
    row = result.iloc[0]
    total = row["cs_weight_pct"] + row["ds_weight_pct"] + row["other_pct"]
    assert abs(total - 100.0) < 0.5


def test_attribution_has_expected_columns():
    X = pd.DataFrame({
        "cs_ddos_flag": [1.0],
        "ds_anomaly_score": [0.9],
    })
    y_true = pd.Series([3])
    mock_shap = [np.zeros((1, 2))] * 10
    mock_shap[3] = np.array([[0.6, 0.4]])
    result = compute_domain_attribution(mock_shap, X, y_true)
    for col in ["class", "cs_weight_pct", "ds_weight_pct", "other_pct"]:
        assert col in result.columns
