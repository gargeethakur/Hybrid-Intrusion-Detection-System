"""
Step 1.5 — Pre-model Feature Importance
Mutual information + ANOVA F-scores. Exports feature_rankings.csv.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from pathlib import Path

OUTPUTS_PATH = Path("../../shared/outputs")


def compute_feature_rankings(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute mutual info and ANOVA F-score for each feature."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    f_scores, _ = f_classif(X, y)

    rankings = pd.DataFrame({
        "feature": X.columns,
        "mutual_info": mi_scores,
        "anova_f": f_scores,
    }).sort_values("mutual_info", ascending=False).reset_index(drop=True)

    return rankings


def export_rankings(rankings: pd.DataFrame, path: Path = OUTPUTS_PATH) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(path / "feature_rankings.csv", index=False)
    print(f"[INFO] Feature rankings saved to {path / 'feature_rankings.csv'}")


if __name__ == "__main__":
    # TODO: load cleaned df and call compute_feature_rankings
    pass
