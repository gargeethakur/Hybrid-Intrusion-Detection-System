"""
Step 2 — Cybersecurity Rule-Based Detection Layer
Snort-style explicit rules applied row-wise to the feature dataframe.

Each rule returns a binary flag column (0 or 1).
All rules are deterministic, hand-coded domain knowledge — not learned from data.
"""
import pandas as pd
import numpy as np

# ── Thresholds (adjust based on EDA findings from Part S) ──────────────────
DDOS_SYN_RATE_THRESHOLD = 1000     # packets/sec
PORTSCAN_UNIQUE_PORTS_THRESHOLD = 100  # unique dst ports/sec
FLOOD_MAX_PKT_SIZE = 60            # bytes
FLOOD_RATE_THRESHOLD = 500         # packets/sec
EXPLOIT_PAYLOAD_THRESHOLD = 1400   # bytes (near MTU)
BACKDOOR_DURATION_THRESHOLD = 300  # seconds (long-lived connection)
SHELLCODE_MAX_BYTES = 500          # very small total bytes
WORM_UNIQUE_DST_IPS = 50          # unique dst IPs from same src


def apply_cs_ddos_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 1 — DDoS Check
    IF SYN rate > 1000 packets/sec THEN cs_ddos_flag = 1
    Target classes: DoS (3), Generic (6)
    """
    # CICFlowMeter columns: 'flow_packets/s', 'syn_flag_count'
    pkt_rate = df.get("flow_packets/s", pd.Series(0, index=df.index))
    syn_flags = df.get("syn_flag_count", pd.Series(0, index=df.index))
    flag = ((pkt_rate > DDOS_SYN_RATE_THRESHOLD) & (syn_flags > 0)).astype(int)
    return flag


def apply_cs_portscan_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 2 — Port Scan Check
    IF unique dst ports > 100/sec THEN cs_portscan_flag = 1
    Target classes: Reconnaissance (7), Analysis (1)
    NOTE: True per-source unique-port counting requires groupby on src IP.
          Here we use destination_port variance as a proxy for single-row features.
          Implement full groupby version in the notebook with raw flow data.
    """
    dst_port = df.get("destination_port", pd.Series(0, index=df.index))
    pkt_rate = df.get("flow_packets/s", pd.Series(0, index=df.index))
    # Proxy: high dst port number variability + high rate
    flag = ((dst_port > 1024) & (pkt_rate > PORTSCAN_UNIQUE_PORTS_THRESHOLD)).astype(int)
    return flag


def apply_cs_flood_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 3 — Small Packet Flood Check
    IF avg packet size < 60 bytes AND rate > 500/sec THEN cs_flood_flag = 1
    Target classes: Fuzzers (5), DoS (3)
    """
    avg_pkt_size = df.get("average_packet_size", pd.Series(9999, index=df.index))
    pkt_rate = df.get("flow_packets/s", pd.Series(0, index=df.index))
    flag = ((avg_pkt_size < FLOOD_MAX_PKT_SIZE) & (pkt_rate > FLOOD_RATE_THRESHOLD)).astype(int)
    return flag


def apply_cs_exploit_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 4 — Exploit Check
    IF avg packet size near MTU AND unusual flag combinations THEN cs_exploit_flag = 1
    Target classes: Exploits (4)
    """
    avg_pkt_size = df.get("average_packet_size", pd.Series(0, index=df.index))
    urg_flags = df.get("urg_flag_count", pd.Series(0, index=df.index))
    flag = ((avg_pkt_size > EXPLOIT_PAYLOAD_THRESHOLD) | (urg_flags > 0)).astype(int)
    return flag


def apply_cs_backdoor_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 5 — Backdoor Check
    IF long-duration low-rate flow on non-standard port THEN cs_backdoor_flag = 1
    Target classes: Backdoor (2)
    """
    duration = df.get("flow_duration", pd.Series(0, index=df.index))  # microseconds
    pkt_rate = df.get("flow_packets/s", pd.Series(9999, index=df.index))
    dst_port = df.get("destination_port", pd.Series(80, index=df.index))
    non_std_port = ~dst_port.isin([80, 443, 22, 21, 25, 53])
    flag = (
        (duration > BACKDOOR_DURATION_THRESHOLD * 1e6) &
        (pkt_rate < 10) &
        non_std_port
    ).astype(int)
    return flag


def apply_cs_shellcode_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 6 — Shellcode Check
    IF total bytes very small AND non-HTTP port THEN cs_shellcode_flag = 1
    Target classes: Shellcode (8)
    """
    total_fwd = df.get("total_length_of_fwd_packets", pd.Series(9999, index=df.index))
    dst_port = df.get("destination_port", pd.Series(80, index=df.index))
    non_http = ~dst_port.isin([80, 443, 8080, 8443])
    flag = ((total_fwd < SHELLCODE_MAX_BYTES) & non_http).astype(int)
    return flag


def apply_cs_worm_flag(df: pd.DataFrame) -> pd.Series:
    """
    CS Rule 7 — Worm Check
    High unique destination IPs from same source (requires groupby — proxy here).
    Target classes: Worms (9)
    """
    # Proxy: very high packet rate + low bytes per packet (scanning behaviour)
    pkt_rate = df.get("flow_packets/s", pd.Series(0, index=df.index))
    avg_pkt_size = df.get("average_packet_size", pd.Series(9999, index=df.index))
    flag = ((pkt_rate > 200) & (avg_pkt_size < 100)).astype(int)
    return flag


def apply_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all CS rules and return df with new flag columns."""
    df = df.copy()
    df["cs_ddos_flag"]      = apply_cs_ddos_flag(df)
    df["cs_portscan_flag"]  = apply_cs_portscan_flag(df)
    df["cs_flood_flag"]     = apply_cs_flood_flag(df)
    df["cs_exploit_flag"]   = apply_cs_exploit_flag(df)
    df["cs_backdoor_flag"]  = apply_cs_backdoor_flag(df)
    df["cs_shellcode_flag"] = apply_cs_shellcode_flag(df)
    df["cs_worm_flag"]      = apply_cs_worm_flag(df)
    df["cs_any_flag"]       = df[[c for c in df.columns if c.startswith("cs_") and c != "cs_any_flag"]].max(axis=1)
    return df
