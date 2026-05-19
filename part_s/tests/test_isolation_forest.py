"""
Tests for Isolation Forest anomaly scorer.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from anomaly.isolation_forest import train_isolation_forest, score


def make_X():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, (100, 5))
    anomalous = rng.normal(10, 1, (10, 5))
    X = np.vstack([normal, anomalous])
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])


def test_scores_in_range():
    X = make_X()
    model = train_isolation_forest(X, contamination=0.1)
    scores = score(model, X)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_anomalies_score_higher():
    X = make_X()
    model = train_isolation_forest(X, contamination=0.1)
    scores = score(model, X)
    normal_mean = scores[:100].mean()
    anomaly_mean = scores[100:].mean()
    assert anomaly_mean > normal_mean
