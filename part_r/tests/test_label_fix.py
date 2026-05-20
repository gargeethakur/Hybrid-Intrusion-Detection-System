"""
Tests for the label correction module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_fixes.label_fix import (
    detect_portscan_candidates,
    drop_artefact_flows,
    relabel_portscan_flows,
    fix_duplicate_timestamps,
    run_label_fix_pipeline,
)


def make_df():
    return pd.DataFrame({
        "flow_packets/s":       [500.0, 10.0,  600.0, 5.0],
        "destination_port":     [2000,  80,    3000,  443],
        "flow_duration":        [100,   50000, 50,    60000],
        "average_packet_size":  [40,    500,   35,    800],
        "label":                [0,     0,     0,     3],   # row 0,2 suspected portscans
    })


def test_detect_portscan_candidates_finds_suspects():
    df = make_df()
    mask = detect_portscan_candidates(df, rate_threshold=50.0)
    # rows 0 and 2 match all conditions and are not labelled Reconnaissance
    assert mask.iloc[0] == True
    assert mask.iloc[2] == True
    assert mask.iloc[1] == False   # low rate
    assert mask.iloc[3] == False   # large packet size


def test_relabel_changes_label_correctly():
    df = make_df()
    mask = detect_portscan_candidates(df)
    fixed = relabel_portscan_flows(df, mask, dry_run=False)
    assert fixed.loc[mask, "label"].eq(7).all()


def test_relabel_dry_run_does_not_change_labels():
    df = make_df()
    mask = detect_portscan_candidates(df)
    fixed = relabel_portscan_flows(df, mask, dry_run=True)
    assert (fixed["label"] == df["label"]).all()


def test_drop_artefact_flows_removes_zero_duration():
    df = pd.DataFrame({
        "flow_duration": [0.0, -1.0, 100.0, 500.0],
        "label": [0, 1, 2, 3],
    })
    fixed = drop_artefact_flows(df, min_duration_us=0.0)
    assert len(fixed) == 2
    assert (fixed["flow_duration"] > 0).all()


def test_fix_duplicate_timestamps_resolves_conflicts():
    df = pd.DataFrame({
        "timestamp": ["t1", "t1", "t2"],
        "label":     [0,    3,    1],
        "feature":   [1.0,  1.0,  2.0],
    })
    fixed = fix_duplicate_timestamps(df, ts_col="timestamp")
    # t1 had conflicting labels — should be resolved to one row
    assert len(fixed[fixed["timestamp"] == "t1"]) == 1


def test_run_label_fix_pipeline_returns_smaller_or_equal():
    df = make_df()
    df["flow_duration"] = [0, 50000, 50, 60000]  # row 0 is artefact
    fixed, report = run_label_fix_pipeline(df, fix_portscans=True,
                                            fix_duplicates=False,
                                            drop_artefacts=True)
    assert len(fixed) < len(df)  # artefact row removed
    assert "artefacts_dropped" in report
