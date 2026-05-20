"""
Problem Fix 1 — Labelling Error Correction
Port scan flows are mislabelled in UNSW-NB15 / CICFlowMeter output.
Documented by Lanvin et al. 2023 and Engelen et al. 2021/2022.

This module detects and corrects the known labelling issues before any
model is trained. Must run BEFORE Part S cleaning pipeline.

Known issues fixed here:
  A) Port scan flows incorrectly labelled as Benign or other classes
  B) Flows with duplicate timestamps that have conflicting labels
  C) Near-zero duration flows that can't be real connections (artefacts)
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("../../shared/data/processed")
OUTPUTS_PATH   = Path("outputs")

LABEL_MAP = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

# Reconnaissance = label 7 in your dataset (closest to port scan)
PORTSCAN_LABEL = 7


# ── 1. Detect suspected mislabelled port scan flows ──────────────────────────

def detect_portscan_candidates(df: pd.DataFrame,
                                unique_port_threshold: int = 20,
                                rate_threshold: float = 50.0) -> pd.Series:
    """
    Identify flows that behaviorally match port scanning but are NOT labelled
    as Reconnaissance (7).

    Port scan signatures (based on Lanvin et al. 2023):
      - High destination port number (ephemeral ports being probed)
      - High packet rate
      - Very short flow duration
      - Low bytes per packet (no payload — just probing)

    Returns a boolean mask of suspected mislabelled rows.
    """
    col = lambda name: df.get(name, pd.Series(np.nan, index=df.index))

    high_dst_port  = col("destination_port") > 1024
    high_pkt_rate  = col("flow_packets/s")   > rate_threshold
    short_duration = col("flow_duration")    < 1000        # microseconds
    low_payload    = col("average_packet_size") < 80

    looks_like_portscan = high_dst_port & high_pkt_rate & short_duration & low_payload
    already_labelled    = df["label"] == PORTSCAN_LABEL

    suspected = looks_like_portscan & ~already_labelled
    n = suspected.sum()
    print(f"[LABEL_FIX] Detected {n:,} suspected mislabelled port scan flows")
    if n > 0:
        print(f"[LABEL_FIX] Current labels of suspected rows:\n"
              f"{df.loc[suspected, 'label'].map(LABEL_MAP).value_counts().to_string()}\n")
    return suspected


# ── 2. Fix duplicate-timestamp conflicting labels ────────────────────────────

def fix_duplicate_timestamps(df: pd.DataFrame,
                               ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Flows with identical timestamps and conflicting labels are artefacts
    of the CICFlowMeter labelling process (Engelen et al. 2021).

    Strategy: for groups of duplicate-timestamp rows with different labels,
    keep the row whose label appears most often in that group.
    If tied, keep the non-Benign label (conservative — prefer flagging).
    """
    if ts_col not in df.columns:
        print(f"[LABEL_FIX] Column '{ts_col}' not found — skipping duplicate timestamp fix.")
        return df

    before = len(df)
    dupes = df.duplicated(subset=[ts_col], keep=False)
    n_dupes = dupes.sum()

    if n_dupes == 0:
        print("[LABEL_FIX] No duplicate timestamps found.")
        return df

    print(f"[LABEL_FIX] Found {n_dupes:,} rows with duplicate timestamps")

    def resolve_group(group):
        if group["label"].nunique() == 1:
            return group.iloc[[0]]
        # Prefer the most common label; tie-break: prefer non-Benign
        mode_label = group["label"].mode()
        if len(mode_label) > 1:
            non_benign = [l for l in mode_label if l != 0]
            chosen = non_benign[0] if non_benign else mode_label[0]
        else:
            chosen = mode_label[0]
        row = group[group["label"] == chosen].iloc[[0]]
        return row

    clean = df.groupby(ts_col, group_keys=False).apply(resolve_group)
    after = len(clean)
    print(f"[LABEL_FIX] Duplicate resolution: {before:,} → {after:,} rows "
          f"(removed {before - after:,} conflicting duplicates)")
    return clean.reset_index(drop=True)


# ── 3. Remove zero/near-zero duration artefact flows ────────────────────────

def drop_artefact_flows(df: pd.DataFrame,
                         min_duration_us: float = 0.0,
                         duration_col: str = "flow_duration") -> pd.DataFrame:
    """
    Flows with zero or negative duration are measurement artefacts from
    CICFlowMeter — they cannot represent real network connections.
    Drop them entirely.
    """
    if duration_col not in df.columns:
        print(f"[LABEL_FIX] Column '{duration_col}' not found — skipping artefact removal.")
        return df

    mask = df[duration_col] > min_duration_us
    n_dropped = (~mask).sum()
    df = df[mask].reset_index(drop=True)
    print(f"[LABEL_FIX] Dropped {n_dropped:,} zero/negative duration artefact flows")
    return df


# ── 4. Apply port scan relabelling ───────────────────────────────────────────

def relabel_portscan_flows(df: pd.DataFrame,
                            mask: pd.Series,
                            dry_run: bool = False) -> pd.DataFrame:
    """
    Relabel suspected port scan flows to Reconnaissance (label=7).
    Set dry_run=True to see what would change without modifying data.
    """
    n = mask.sum()
    if n == 0:
        print("[LABEL_FIX] No flows to relabel.")
        return df

    if dry_run:
        print(f"[LABEL_FIX] DRY RUN — would relabel {n:,} flows to Reconnaissance")
        return df

    df = df.copy()
    df.loc[mask, "label"] = PORTSCAN_LABEL
    print(f"[LABEL_FIX] Relabelled {n:,} flows → Reconnaissance (label={PORTSCAN_LABEL})")
    return df


# ── 5. Full label correction pipeline ────────────────────────────────────────

def run_label_fix_pipeline(df: pd.DataFrame,
                             fix_portscans: bool = True,
                             fix_duplicates: bool = True,
                             drop_artefacts: bool = True,
                             dry_run: bool = False) -> tuple:
    """
    Full label correction pipeline:
      1. Drop zero-duration artefact flows
      2. Resolve duplicate-timestamp conflicting labels
      3. Detect and relabel mislabelled port scan flows

    Returns (fixed_df, fix_report_dict)
    """
    print(f"\n[LABEL_FIX] Starting label correction — shape: {df.shape}")
    report = {}

    if drop_artefacts:
        before = len(df)
        df = drop_artefact_flows(df)
        report["artefacts_dropped"] = before - len(df)

    if fix_duplicates:
        before = len(df)
        df = fix_duplicate_timestamps(df)
        report["duplicates_resolved"] = before - len(df)

    if fix_portscans:
        mask = detect_portscan_candidates(df)
        report["portscans_relabelled"] = int(mask.sum())
        df = relabel_portscan_flows(df, mask, dry_run=dry_run)

    print(f"[LABEL_FIX] Pipeline complete — final shape: {df.shape}\n")
    return df, report


# ── 6. Export ─────────────────────────────────────────────────────────────────

def export_fixed(df: pd.DataFrame,
                  report: dict,
                  path: Path = PROCESSED_PATH) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "label_fixed_data.csv", index=False)

    import json
    with open(path / "label_fix_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[LABEL_FIX] Exported fixed data and report to {path}")


if __name__ == "__main__":
    # Run as dry_run=True first to inspect what would change
    # Then set dry_run=False to apply
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "part_s" / "src"))
    # from eda.loader import load_and_merge, clean as basic_clean
    # df = load_and_merge(); df = basic_clean(df)
    # fixed, report = run_label_fix_pipeline(df, dry_run=True)
    pass
