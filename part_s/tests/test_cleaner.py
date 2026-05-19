"""
Tests for the data cleaning pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from eda.cleaner import (
    drop_constant_columns,
    drop_duplicate_columns,
    clip_outliers_iqr,
    fix_skewed_features,
    fill_remaining_nulls,
    scale_features,
    run_cleaning_pipeline,
)


def make_df():
    return pd.DataFrame({
        "normal_feat":    [1.0, 2.0, 3.0, 4.0, 5.0],
        "constant_feat":  [0.0, 0.0, 0.0, 0.0, 0.0],
        "skewed_feat":    [0.0, 0.0, 0.0, 0.0, 1000.0],
        "outlier_feat":   [1.0, 2.0, 2.0, 2.0, 999.0],
        "null_feat":      [1.0, np.nan, 3.0, np.nan, 5.0],
        "label":          [0, 1, 2, 0, 1],
    })


def test_drop_constant_columns():
    df = make_df()
    result = drop_constant_columns(df)
    assert "constant_feat" not in result.columns
    assert "normal_feat" in result.columns


def test_drop_duplicate_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1, 2, 3],   # exact duplicate of a
        "c": [4, 5, 6],
        "label": [0, 1, 0],
    })
    result = drop_duplicate_columns(df)
    assert result.shape[1] < df.shape[1]


def test_clip_outliers_does_not_affect_label():
    df = make_df().drop(columns=["null_feat", "constant_feat", "skewed_feat"])
    result = clip_outliers_iqr(df, factor=1.5)
    assert result["label"].tolist() == df["label"].tolist()


def test_clip_outliers_reduces_max():
    df = make_df().drop(columns=["null_feat", "constant_feat", "skewed_feat"])
    original_max = df["outlier_feat"].max()
    result = clip_outliers_iqr(df, factor=1.5)
    assert result["outlier_feat"].max() <= original_max


def test_fill_remaining_nulls_no_nulls_after():
    df = make_df().drop(columns=["constant_feat", "skewed_feat", "outlier_feat"])
    result = fill_remaining_nulls(df)
    assert result.isnull().sum().sum() == 0


def test_fix_skewed_features_reduces_skew():
    df = make_df().drop(columns=["constant_feat", "null_feat", "outlier_feat"])
    original_skew = df["skewed_feat"].skew()
    result, transformed = fix_skewed_features(df, skew_threshold=0.5)
    assert "skewed_feat" in transformed
    assert abs(result["skewed_feat"].skew()) < abs(original_skew)


def test_scale_features_in_01_range():
    df = make_df().drop(columns=["constant_feat", "null_feat", "skewed_feat"])
    result, scaler = scale_features(df, method="minmax")
    numeric = result.drop(columns=["label"]).select_dtypes(include=[np.number])
    assert numeric.min().min() >= -1e-9
    assert numeric.max().max() <= 1.0 + 1e-9


def test_run_cleaning_pipeline_returns_smaller_or_equal_shape():
    df = make_df()
    result, scaler, log_cols = run_cleaning_pipeline(df)
    assert result.shape[0] == df.shape[0]   # no rows dropped
    assert result.shape[1] <= df.shape[1]   # constant/dup cols removed
    assert result.isnull().sum().sum() == 0  # no nulls remain
