"""
Step 2.3 — Rule Evaluation
Computes per-rule precision, recall, F1 against ground-truth labels.
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_rules(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    For each CS flag column, treat it as a binary anomaly detector.
    Compute precision, recall, F1 vs any non-benign label (binary: 0=benign, 1=attack).
    """
    binary_true = (df[label_col] != 0).astype(int)
    flag_cols = [c for c in df.columns if c.startswith("cs_") and c != "cs_any_flag"]

    results = []
    for col in flag_cols:
        pred = df[col]
        results.append({
            "rule": col,
            "fires_total": pred.sum(),
            "precision": precision_score(binary_true, pred, zero_division=0),
            "recall": recall_score(binary_true, pred, zero_division=0),
            "f1": f1_score(binary_true, pred, zero_division=0),
            "fp_on_benign": ((pred == 1) & (df[label_col] == 0)).sum(),
        })

    return pd.DataFrame(results).sort_values("f1", ascending=False)
