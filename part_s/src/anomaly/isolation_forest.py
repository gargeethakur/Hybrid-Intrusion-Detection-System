"""
Step 3.1 — Isolation Forest Anomaly Scorer
Trains on full feature set (unsupervised) and generates ds_anomaly_score.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

OUTPUTS_PATH = Path("../../shared/outputs")
MODEL_PATH = Path("outputs/isolation_forest_model.pkl")


def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
    """Train Isolation Forest. contamination = estimated fraction of anomalies."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    return model


def score(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """
    Return anomaly scores in [0, 1] range.
    sklearn decision_function: more negative = more anomalous.
    We invert and normalise so 1.0 = most anomalous.
    """
    raw_scores = model.decision_function(X)  # higher = more normal
    normalised = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return normalised


def export_with_scores(df: pd.DataFrame, scores: np.ndarray, path: Path = OUTPUTS_PATH) -> None:
    """Append ds_anomaly_score column and export to shared outputs."""
    df = df.copy()
    df["ds_anomaly_score"] = scores
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "features_with_ds.csv", index=False)
    print(f"[INFO] features_with_ds.csv saved to {path}")


def save_model(model: IsolationForest, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


if __name__ == "__main__":
    # TODO: load cleaned df, train, score, export
    pass
