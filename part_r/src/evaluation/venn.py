"""
Step 5.3 — Venn Diagram of Detections
Compares CS-only, DS-only, Both, and Neither detection regions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUTS_PATH = Path("outputs")


def compute_detection_sets(df: pd.DataFrame, ds_threshold: float = 0.5) -> dict:
    """
    Classify each row into one of four detection categories.
    df must contain: cs_any_flag, ds_anomaly_score, label
    """
    cs = df["cs_any_flag"] == 1
    ds = df["ds_anomaly_score"] >= ds_threshold

    sets = {
        "cs_only":  (cs & ~ds),
        "ds_only":  (~cs & ds),
        "both":     (cs & ds),
        "neither":  (~cs & ~ds),
    }
    return {k: df[v] for k, v in sets.items()}


def plot_venn(detection_sets: dict, output_path: Path = OUTPUTS_PATH) -> None:
    """
    Draw a Venn-style diagram showing CS-only, DS-only, and Both regions.
    Uses matplotlib patches (matplotlib-venn optional).
    """
    try:
        from matplotlib_venn import venn2
        cs_total = len(detection_sets["cs_only"]) + len(detection_sets["both"])
        ds_total = len(detection_sets["ds_only"]) + len(detection_sets["both"])
        both_total = len(detection_sets["both"])

        fig, ax = plt.subplots(figsize=(8, 6))
        v = venn2(subsets=(
            len(detection_sets["cs_only"]),
            len(detection_sets["ds_only"]),
            len(detection_sets["both"])
        ), set_labels=("CS Rules", "DS Anomaly"), ax=ax)
        ax.set_title("Detection Coverage: CS Rules vs DS Anomaly Score", fontsize=14)

    except ImportError:
        # Fallback: manual patch diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        cs_circle  = mpatches.Ellipse((0.35, 0.5), 0.5, 0.6, color="#c8b8f5", alpha=0.6, label="CS Rules")
        ds_circle  = mpatches.Ellipse((0.65, 0.5), 0.5, 0.6, color="#b8f5c8", alpha=0.6, label="DS Anomaly")
        ax.add_patch(cs_circle)
        ax.add_patch(ds_circle)

        ax.text(0.28, 0.5, f"CS Only\n{len(detection_sets['cs_only'])}\nKnown attacks",
                ha="center", va="center", fontsize=10)
        ax.text(0.72, 0.5, f"DS Only\n{len(detection_sets['ds_only'])}\nNovel / zero-day",
                ha="center", va="center", fontsize=10)
        ax.text(0.50, 0.5, f"Both\n{len(detection_sets['both'])}\nHigh confidence",
                ha="center", va="center", fontsize=10, fontweight="bold")

        ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9)
        ax.axis("off")
        ax.set_title("Detection Coverage: CS Rules vs DS Anomaly Score", fontsize=14)
        ax.legend(handles=[cs_circle, ds_circle], loc="lower center", ncol=2)

    output_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path / "venn_diagram.png", dpi=150)
    plt.close()
    print(f"[INFO] Venn diagram saved")


if __name__ == "__main__":
    # TODO: load predictions.csv with cs and ds columns, call compute_detection_sets + plot_venn
    pass
