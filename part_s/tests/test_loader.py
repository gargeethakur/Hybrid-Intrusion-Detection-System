"""
Tests for data loading and cleaning utilities.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eda.loader import clean, class_distribution


def make_sample_df():
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, np.inf, 4.0],
        "feature_b": [0.5, np.nan, 1.5, 2.0],
        "label": [0, 1, 2, 0],
    })


def test_clean_removes_inf():
    df = clean(make_sample_df())
    assert not np.isinf(df.select_dtypes(include=[np.number])).any().any()


def test_clean_standardises_columns():
    df = pd.DataFrame({"Feature A": [1], " Feature B ": [2], "label": [0]})
    df = clean(df)
    assert "feature_a" in df.columns
    assert "feature_b" in df.columns


def test_class_distribution_shape():
    df = pd.DataFrame({"label": [0, 0, 1, 2, 2, 2]})
    dist = class_distribution(df)
    assert len(dist) == 3
    assert "count" in dist.columns
    assert abs(dist["ratio"].sum() - 1.0) < 1e-9
