"""
Tests for Venn diagram detection set computation.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluation.venn import compute_detection_sets


def make_df():
    return pd.DataFrame({
        "cs_any_flag":      [1, 0, 1, 0],
        "ds_anomaly_score": [0.8, 0.8, 0.2, 0.2],
        "label":            [3, 8, 1, 0],
    })


def test_detection_sets_correct_counts():
    df = make_df()
    sets = compute_detection_sets(df, ds_threshold=0.5)
    assert len(sets["both"]) == 1      # row 0: cs=1, ds>=0.5
    assert len(sets["ds_only"]) == 1   # row 1: cs=0, ds>=0.5
    assert len(sets["cs_only"]) == 1   # row 2: cs=1, ds<0.5
    assert len(sets["neither"]) == 1   # row 3: cs=0, ds<0.5


def test_detection_sets_cover_all_rows():
    df = make_df()
    sets = compute_detection_sets(df, ds_threshold=0.5)
    total = sum(len(v) for v in sets.values())
    assert total == len(df)
