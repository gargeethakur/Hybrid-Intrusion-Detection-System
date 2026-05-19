"""
Tests for CS rule functions.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cs_rules.rules import (
    apply_cs_ddos_flag, apply_cs_portscan_flag,
    apply_cs_flood_flag, apply_all_rules
)


def make_row(**kwargs) -> pd.DataFrame:
    defaults = {
        "flow_packets/s": 0,
        "syn_flag_count": 0,
        "destination_port": 80,
        "average_packet_size": 500,
        "flow_duration": 1000,
        "total_length_of_fwd_packets": 5000,
        "urg_flag_count": 0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


def test_ddos_flag_fires_on_high_syn_rate():
    df = make_row(**{"flow_packets/s": 1500, "syn_flag_count": 5})
    assert apply_cs_ddos_flag(df).iloc[0] == 1


def test_ddos_flag_silent_on_normal_traffic():
    df = make_row(**{"flow_packets/s": 100, "syn_flag_count": 0})
    assert apply_cs_ddos_flag(df).iloc[0] == 0


def test_flood_flag_fires_on_small_fast_packets():
    df = make_row(**{"average_packet_size": 40, "flow_packets/s": 600})
    assert apply_cs_flood_flag(df).iloc[0] == 1


def test_flood_flag_silent_on_normal_traffic():
    df = make_row(**{"average_packet_size": 500, "flow_packets/s": 100})
    assert apply_cs_flood_flag(df).iloc[0] == 0


def test_apply_all_rules_adds_flag_columns():
    df = make_row()
    result = apply_all_rules(df)
    expected_cols = [
        "cs_ddos_flag", "cs_portscan_flag", "cs_flood_flag",
        "cs_exploit_flag", "cs_backdoor_flag", "cs_shellcode_flag",
        "cs_worm_flag", "cs_any_flag"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_cs_any_flag_is_max_of_all_flags():
    df = make_row(**{"flow_packets/s": 1500, "syn_flag_count": 5})
    result = apply_all_rules(df)
    assert result["cs_any_flag"].iloc[0] == 1
