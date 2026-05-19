"""
Step 5.1 — SHAP Value Computation
Loads the trained fusion model and computes SHAP values for every prediction.
"""
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

SHARED_OUTPUTS = Path("../../shared/outputs")
MODEL_PATH = Path("../../part_g/outputs/fusion_model.pkl")
OUTPUTS_PATH = Path("outputs")


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    predictions = pd.read_csv(SHARED_OUTPUTS / "predictions.csv")
    X = predictions.drop(columns=["true_label", "predicted_label", "confidence"], errors="ignore")
    y_true = predictions["true_label"]
    y_pred = predictions["predicted_label"]
    return model, X, y_true, y_pred


def compute_shap_values(model, X: pd.DataFrame):
    """Use TreeExplainer for XGBoost / Random Forest."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_global_summary(shap_values, X: pd.DataFrame, output_path: Path = OUTPUTS_PATH) -> None:
    """Global beeswarm summary plot."""
    output_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path / "shap_summary_plot.png", dpi=150)
    plt.close()
    print(f"[INFO] SHAP summary plot saved")


def compute_domain_attribution(shap_values, X: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """
    Step 5.2 — For each attack class, compute CS% vs DS% weight.
    shap_values: list of arrays (one per class) for multiclass, shape (n_samples, n_features)
    """
    label_map = {
        0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
        4: "Exploits", 5: "Fuzzers", 6: "Generic",
        7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
    }
    cs_cols = [c for c in X.columns if c.startswith("cs_")]
    ds_cols = ["ds_anomaly_score"]

    rows = []
    for class_idx, class_name in label_map.items():
        mask = (y_true == class_idx).values
        if mask.sum() == 0:
            continue

        if isinstance(shap_values, list):
            sv = np.abs(shap_values[class_idx][mask])  # multiclass
        else:
            sv = np.abs(shap_values[mask])

        total = sv.sum()
        if total == 0:
            continue

        cs_idx = [X.columns.get_loc(c) for c in cs_cols if c in X.columns]
        ds_idx = [X.columns.get_loc(c) for c in ds_cols if c in X.columns]

        cs_weight = sv[:, cs_idx].sum() / total * 100 if cs_idx else 0
        ds_weight = sv[:, ds_idx].sum() / total * 100 if ds_idx else 0

        rows.append({
            "class": class_name,
            "label": class_idx,
            "cs_weight_pct": round(cs_weight, 1),
            "ds_weight_pct": round(ds_weight, 1),
            "other_pct": round(100 - cs_weight - ds_weight, 1),
            "n_samples": mask.sum(),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    model, X, y_true, y_pred = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, X)
    plot_global_summary(shap_values, X)
    attribution = compute_domain_attribution(shap_values, X, y_true)
    print(attribution)
    attribution.to_csv(OUTPUTS_PATH / "domain_attribution.csv", index=False)
