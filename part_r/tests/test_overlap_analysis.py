"""
Tests for the class overlap analysis module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_fixes.overlap_analysis import (
    compute_pairwise_separability,
    find_most_overlapping_pairs,
    compute_lda_separability,
)


def make_df():
    rng = np.random.default_rng(0)
    # 3 well-separated classes
    c0 = rng.normal(0,  1, (100, 4))
    c1 = rng.normal(10, 1, (100, 4))
    c2 = rng.normal(20, 1, (100, 4))
    X = pd.DataFrame(np.vstack([c0, c1, c2]), columns=["a","b","c","d"])
    y = pd.Series([0]*100 + [1]*100 + [2]*100)
    return X, y


def test_separability_matrix_shape():
    X, y = make_df()
    mat = compute_pairwise_separability(X, y)
    n = y.nunique()
    assert mat.shape == (n, n)


def test_separability_diagonal_is_zero():
    X, y = make_df()
    mat = compute_pairwise_separability(X, y)
    assert np.allclose(np.diag(mat.values), 0)


def test_separability_is_symmetric():
    X, y = make_df()
    mat = compute_pairwise_separability(X, y)
    assert np.allclose(mat.values, mat.values.T)


def test_well_separated_classes_have_high_J():
    X, y = make_df()
    mat = compute_pairwise_separability(X, y)
    # All pairs should have high separability since means are far apart
    off_diag = mat.values[np.triu_indices(3, k=1)]
    assert all(v > 100 for v in off_diag), f"Expected high J, got: {off_diag}"


def test_overlapping_pairs_returns_correct_count():
    X, y = make_df()
    mat = compute_pairwise_separability(X, y)
    pairs = find_most_overlapping_pairs(mat, n_pairs=2)
    assert len(pairs) == 2
    assert "class_a" in pairs.columns
    assert "risk" in pairs.columns


def test_lda_separability_returns_dict():
    X, y = make_df()
    result = compute_lda_separability(X, y)
    assert "total_variance_explained" in result
    assert 0 < result["total_variance_explained"] <= 1.0
