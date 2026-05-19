"""
Step 4 — Fusion ML Model Training
XGBoost / Random Forest trained on CS flags + DS anomaly score + original features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
from pathlib import Path

SHARED_OUTPUTS = Path("../../shared/outputs")
MODEL_OUTPUT = Path("outputs/fusion_model.pkl")


def build_feature_matrix(features_with_ds_path: Path, features_with_cs_path: Path) -> tuple:
    """Merge DS scores and CS flags with original features into one matrix."""
    ds_df = pd.read_csv(features_with_ds_path)
    cs_df = pd.read_csv(features_with_cs_path)

    cs_flag_cols = [c for c in cs_df.columns if c.startswith("cs_")]
    merged = ds_df.copy()
    for col in cs_flag_cols:
        merged[col] = cs_df[col].values

    X = merged.drop(columns=["label"], errors="ignore")
    y = merged["label"]
    return X, y


def train_xgboost(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> str:
    y_pred = model.predict(X_test)
    label_names = ["Benign","Analysis","Backdoor","DoS","Exploits",
                   "Fuzzers","Generic","Reconnaissance","Shellcode","Worms"]
    return classification_report(y_test, y_pred, target_names=label_names)


def save_model(model, path: Path = MODEL_OUTPUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")


def export_predictions(model, X_test, y_test, path: Path = SHARED_OUTPUTS) -> None:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test).max(axis=1)
    out = X_test.copy()
    out["true_label"] = y_test.values
    out["predicted_label"] = preds
    out["confidence"] = proba
    path.mkdir(parents=True, exist_ok=True)
    out.to_csv(path / "predictions.csv", index=False)
    print(f"[INFO] predictions.csv saved")


if __name__ == "__main__":
    # TODO: wire up build_feature_matrix, train, evaluate, export
    pass
