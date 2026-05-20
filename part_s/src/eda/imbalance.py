"""
Problem Fix 2 — Class Imbalance Handler
Worms: 174 records vs Benign: 2.2M — model will ignore rare classes without fixing this.

Applies SMOTE oversampling on minority classes and optional undersampling on Benign.
Must be run AFTER cleaning and BEFORE model training (Part G Step 4).
Exports the balanced dataset to shared/data/processed/balanced_data.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

PROCESSED_PATH = Path("../../shared/data/processed")
OUTPUTS_PATH   = Path("outputs")

LABEL_MAP = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}


# ── 1. Diagnose imbalance ────────────────────────────────────────────────────

def diagnose_imbalance(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Print and return a table of class counts, percentages, and imbalance ratio.
    Imbalance ratio = majority count / class count. Higher = more imbalanced.
    """
    counts = df[label_col].value_counts().sort_index()
    majority = counts.max()

    report = pd.DataFrame({
        "class":            [LABEL_MAP[i] for i in counts.index],
        "count":            counts.values,
        "pct":              (counts.values / counts.sum() * 100).round(2),
        "imbalance_ratio":  (majority / counts.values).round(1),
    })

    print("\n[IMBALANCE] Class distribution:")
    print(report.to_string(index=False))
    print(f"\n[IMBALANCE] Majority class: {majority:,} samples")
    print(f"[IMBALANCE] Most underrepresented: {counts.min():,} samples "
          f"({LABEL_MAP[counts.idxmin()]})\n")
    return report


# ── 2. SMOTE on minority classes ─────────────────────────────────────────────

def apply_smote(X: pd.DataFrame, y: pd.Series,
                min_samples: int = 1000,
                random_state: int = 42) -> tuple:
    """
    Apply SMOTE to classes with fewer than min_samples records.
    Classes already above min_samples are left untouched.

    Returns (X_resampled, y_resampled).

    Requires: pip install imbalanced-learn
    """
    from imblearn.over_sampling import SMOTE

    counts = Counter(y)
    # Build sampling strategy: only oversample classes below min_samples
    strategy = {
        cls: max(count, min_samples)
        for cls, count in counts.items()
        if count < min_samples
    }

    if not strategy:
        print("[IMBALANCE] No classes below threshold — SMOTE not applied.")
        return X, y

    print(f"[IMBALANCE] Applying SMOTE to {len(strategy)} classes: "
          f"{[LABEL_MAP[c] for c in strategy]}")

    smote = SMOTE(
        sampling_strategy=strategy,
        random_state=random_state,
        k_neighbors=min(5, min(counts[c] for c in strategy) - 1)
    )
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[IMBALANCE] After SMOTE: {len(X_res):,} total samples")
    return X_res, y_res


# ── 3. Undersample majority class (Benign) ───────────────────────────────────

def undersample_majority(X: pd.DataFrame, y: pd.Series,
                          max_majority: int = 100_000,
                          random_state: int = 42) -> tuple:
    """
    Randomly undersample the majority class (Benign, label=0) to max_majority.
    Leaves all other classes untouched.
    This is run AFTER SMOTE to prevent Benign from still dominating.
    """
    from imblearn.under_sampling import RandomUnderSampler

    counts = Counter(y)
    majority_label = max(counts, key=counts.get)

    if counts[majority_label] <= max_majority:
        print(f"[IMBALANCE] Majority class already at {counts[majority_label]:,} — no undersampling needed.")
        return X, y

    strategy = {majority_label: max_majority}
    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    print(f"[IMBALANCE] After undersampling: {len(X_res):,} total samples "
          f"(Benign capped at {max_majority:,})")
    return X_res, y_res


# ── 4. Full balancing pipeline ───────────────────────────────────────────────

def run_balancing_pipeline(df: pd.DataFrame,
                            label_col: str = "label",
                            smote_min_samples: int = 1000,
                            undersample_max: int = 100_000,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Full pipeline:
      1. Diagnose imbalance (print report)
      2. SMOTE on all classes below smote_min_samples
      3. Undersample Benign to undersample_max
      4. Return balanced dataframe

    Recommended values based on UNSW-NB15 distribution:
      smote_min_samples = 1000  (Worms=174, Shellcode=1511 → both get boosted)
      undersample_max   = 100_000 (reduces Benign from 2.2M to manageable size)
    """
    print(f"\n[IMBALANCE] Starting balancing pipeline — shape: {df.shape}")
    diagnose_imbalance(df, label_col)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X, y = apply_smote(X, y, min_samples=smote_min_samples, random_state=random_state)
    X, y = undersample_majority(X, y, max_majority=undersample_max, random_state=random_state)

    balanced_df = X.copy()
    balanced_df[label_col] = y.values

    print(f"\n[IMBALANCE] Pipeline complete — final shape: {balanced_df.shape}")
    diagnose_imbalance(balanced_df, label_col)
    return balanced_df


# ── 5. Export ─────────────────────────────────────────────────────────────────

def export_balanced(df: pd.DataFrame,
                     path: Path = PROCESSED_PATH,
                     filename: str = "balanced_data.csv") -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / filename, index=False)
    print(f"[IMBALANCE] Exported balanced data to {path / filename}")


if __name__ == "__main__":
    from loader import load_and_merge, clean as basic_clean
    from cleaner import run_cleaning_pipeline

    df = load_and_merge()
    df = basic_clean(df)
    df, _, _ = run_cleaning_pipeline(df)
    balanced = run_balancing_pipeline(df)
    export_balanced(balanced)
