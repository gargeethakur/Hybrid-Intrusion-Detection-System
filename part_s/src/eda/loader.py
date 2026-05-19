"""
Step 1.1 — Data Loading & Cleaning
Loads Data.csv + Label.csv, merges, cleans, and validates.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("../../shared/data/raw")
PROCESSED_PATH = Path("../../shared/data/processed")


def load_and_merge(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load Data.csv and Label.csv, merge on index."""
    data = pd.read_csv(data_path / "Data.csv")
    labels = pd.read_csv(data_path / "Label.csv", header=None, names=["label"])
    df = pd.concat([data, labels], axis=1)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle nulls, infinities, duplicates, and column name standardisation."""
    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop full duplicates
    df.drop_duplicates(inplace=True)
    # Report nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"[WARN] Null values found:\n{null_counts[null_counts > 0]}")
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class sample counts and ratios."""
    label_map = {
        0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
        4: "Exploits", 5: "Fuzzers", 6: "Generic",
        7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
    }
    counts = df["label"].value_counts().sort_index()
    dist = pd.DataFrame({
        "class": [label_map[i] for i in counts.index],
        "count": counts.values,
        "ratio": counts.values / counts.sum()
    })
    return dist


if __name__ == "__main__":
    df = load_and_merge()
    df = clean(df)
    print(df.shape)
    print(class_distribution(df))
