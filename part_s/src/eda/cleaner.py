"""
Step 1.1 (extended) — Data Cleaning
Full cleaning pipeline applied after loading.
Handles: outliers, skewed features, constant/duplicate columns,
         feature scaling, and exports the cleaned dataframe.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

PROCESSED_PATH = Path("../../shared/data/processed")


# ── 1. Drop constant and near-constant columns ─────────────────────────────

def drop_constant_columns(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Drop columns where the most frequent value appears in more than
    (1 - threshold) fraction of rows — they carry no signal.
    """
    before = df.shape[1]
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols)
    print(f"[CLEAN] Dropped {len(constant_cols)} constant columns: {constant_cols}")
    return df


# ── 2. Drop duplicate columns ───────────────────────────────────────────────

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are exact duplicates of another column."""
    before = df.shape[1]
    df = df.T.drop_duplicates().T
    dropped = before - df.shape[1]
    print(f"[CLEAN] Dropped {dropped} duplicate columns")
    return df


# ── 3. Outlier handling ─────────────────────────────────────────────────────

def clip_outliers_iqr(df: pd.DataFrame, factor: float = 3.0,
                       exclude_cols: list = None) -> pd.DataFrame:
    """
    Clip outliers using IQR method.
    Values beyond median ± factor * IQR are clipped to the fence.
    Skips label column and any columns listed in exclude_cols.
    """
    exclude_cols = exclude_cols or ["label"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        clipped = df[col].clip(lower, upper)
        n_clipped = (df[col] != clipped).sum()
        if n_clipped > 0:
            df[col] = clipped
            # (Uncomment to see per-column clip counts)
            # print(f"[CLEAN] Clipped {n_clipped} outliers in '{col}'")

    print(f"[CLEAN] Outlier clipping done (IQR factor={factor})")
    return df


# ── 4. Skew correction ──────────────────────────────────────────────────────

def fix_skewed_features(df: pd.DataFrame, skew_threshold: float = 1.0,
                         exclude_cols: list = None) -> tuple:
    """
    Apply log1p transform to highly skewed positive features.
    Returns (transformed_df, list_of_transformed_columns).
    """
    exclude_cols = exclude_cols or ["label"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    skewed = df[numeric_cols].skew()
    skewed_cols = skewed[abs(skewed) > skew_threshold].index.tolist()

    transformed = []
    for col in skewed_cols:
        if df[col].min() >= 0:  # log1p only valid for non-negative values
            df[col] = np.log1p(df[col])
            transformed.append(col)

    print(f"[CLEAN] Log1p applied to {len(transformed)} skewed features")
    return df, transformed


# ── 5. Feature scaling ──────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame, method: str = "minmax",
                    exclude_cols: list = None) -> tuple:
    """
    Scale numeric features. Returns (scaled_df, fitted_scaler).
    method: 'minmax' (default, keeps [0,1]) or 'standard' (z-score)
    Note: fit on training data only; transform train + test separately in model code.
    """
    exclude_cols = exclude_cols or ["label"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df = df.copy()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"[CLEAN] Scaled {len(numeric_cols)} features using {method} scaler")
    return df, scaler


# ── 6. Handle remaining nulls ───────────────────────────────────────────────

def fill_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    After outlier clipping, fill any remaining nulls with column median.
    Only applied to numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    null_counts = df[numeric_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()

    for col in cols_with_nulls:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"[CLEAN] Filled {null_counts[col]} nulls in '{col}' with median={median_val:.4f}")

    return df


# ── 7. Full pipeline ────────────────────────────────────────────────────────

def run_cleaning_pipeline(df: pd.DataFrame,
                           outlier_factor: float = 3.0,
                           skew_threshold: float = 1.0,
                           scale_method: str = "minmax") -> tuple:
    """
    Run the full cleaning pipeline in order:
      1. Drop constant columns
      2. Drop duplicate columns
      3. Clip outliers (IQR)
      4. Fill remaining nulls (median)
      5. Fix skewed features (log1p)
      6. Scale features

    Returns (cleaned_df, scaler, log_transformed_cols)
    """
    print(f"\n[CLEAN] Starting pipeline — shape: {df.shape}")

    df = drop_constant_columns(df)
    df = drop_duplicate_columns(df)
    df = clip_outliers_iqr(df, factor=outlier_factor)
    df = fill_remaining_nulls(df)
    df, log_cols = fix_skewed_features(df, skew_threshold=skew_threshold)
    df, scaler = scale_features(df, method=scale_method)

    print(f"[CLEAN] Pipeline complete — final shape: {df.shape}\n")
    return df, scaler, log_cols


# ── 8. Export ───────────────────────────────────────────────────────────────

def export_cleaned(df: pd.DataFrame, path: Path = PROCESSED_PATH,
                    filename: str = "cleaned_data.csv") -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / filename, index=False)
    print(f"[CLEAN] Exported cleaned data to {path / filename}")


if __name__ == "__main__":
    from loader import load_and_merge, clean as basic_clean

    df = load_and_merge()
    df = basic_clean(df)                          # nulls, inf, col names
    df, scaler, log_cols = run_cleaning_pipeline(df)
    export_cleaned(df)
    print(f"Log-transformed columns: {log_cols}")
