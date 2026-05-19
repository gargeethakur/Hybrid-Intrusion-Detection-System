"""
Step 5.4 — Detection Rate Comparison
Per-class detection rates for CS-only, DS-only, and Fusion model.
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUTS_PATH = Path("outputs")
LABEL_MAP = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}


def compute_detection_rates(df: pd.DataFrame, ds_threshold: float = 0.5) -> pd.DataFrame:
    """
    For each class compute:
    - CS detection rate (cs_any_flag == 1 | true class != benign)
    - DS detection rate (ds_anomaly_score >= threshold | true class != benign)
    - Fusion detection rate (predicted_label == true_label)
    """
    rows = []
    for label, name in LABEL_MAP.items():
        class_df = df[df["true_label"] == label]
        if len(class_df) == 0:
            continue

        n = len(class_df)
        cs_rate = (class_df["cs_any_flag"] == 1).mean() if label != 0 else \
                  (class_df["cs_any_flag"] == 0).mean()  # for benign, rate = correctly not flagged
        ds_rate = (class_df["ds_anomaly_score"] >= ds_threshold).mean() if label != 0 else \
                  (class_df["ds_anomaly_score"] < ds_threshold).mean()
        fusion_rate = (class_df["predicted_label"] == label).mean()

        rows.append({
            "class": name,
            "label": label,
            "n_samples": n,
            "cs_detection_rate": round(cs_rate, 3),
            "ds_detection_rate": round(ds_rate, 3),
            "fusion_accuracy": round(fusion_rate, 3),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
