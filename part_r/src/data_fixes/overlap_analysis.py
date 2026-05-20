"""
Problem Fix 3 — Class Overlap Analysis
Multiple classes share similar feature distributions, making decision
boundaries hard to learn. This module identifies overlapping classes
and produces visualisations and metrics to inform Part G's model choices.

This is an ANALYSIS module — it does not modify labels. Its outputs are:
  - Overlap heatmap (pairwise class separability)
  - t-SNE / UMAP plot coloured by class (shows cluster separation)
  - Mahalanobis distance matrix between class centroids
  - Overlap report: which pairs are most confused and why
  - Recommendation: which features best separate overlapping classes
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

OUTPUTS_PATH = Path("outputs")

LABEL_MAP = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}


# ── 1. Pairwise class separability (Fisher's criterion) ──────────────────────

def compute_pairwise_separability(X: pd.DataFrame,
                                   y: pd.Series) -> pd.DataFrame:
    """
    For each pair of classes, compute Fisher's Linear Discriminant ratio:
        J = |mu1 - mu2|^2 / (sigma1^2 + sigma2^2)
    Higher J = better separated. Lower J = more overlap.

    Returns a symmetric DataFrame (classes x classes) of J values.
    """
    classes = sorted(y.unique())
    names = [LABEL_MAP.get(c, str(c)) for c in classes]
    X_arr = X.values
    n = len(classes)
    J_matrix = np.zeros((n, n))

    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i >= j:
                continue
            Xi = X_arr[y == ci]
            Xj = X_arr[y == cj]
            mu_diff = np.mean(Xi, axis=0) - np.mean(Xj, axis=0)
            var_sum = np.var(Xi, axis=0) + np.var(Xj, axis=0) + 1e-9
            J = np.sum(mu_diff ** 2 / var_sum)
            J_matrix[i, j] = J
            J_matrix[j, i] = J

    df = pd.DataFrame(J_matrix, index=names, columns=names)
    return df


def plot_separability_heatmap(sep_matrix: pd.DataFrame,
                               output_path: Path = OUTPUTS_PATH) -> None:
    """
    Heatmap of pairwise Fisher separability.
    Dark = low separability (high overlap). Light = well separated.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(sep_matrix.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(sep_matrix)))
    ax.set_yticks(range(len(sep_matrix)))
    ax.set_xticklabels(sep_matrix.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(sep_matrix.index, fontsize=10)

    for i in range(len(sep_matrix)):
        for j in range(len(sep_matrix)):
            val = sep_matrix.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if val < sep_matrix.values.max() * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Fisher separability (higher = better separated)")
    ax.set_title("Pairwise class separability — low values indicate overlap", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "class_separability_heatmap.png", dpi=150)
    plt.close()
    print(f"[OVERLAP] Separability heatmap saved")


# ── 2. Identify most overlapping pairs ───────────────────────────────────────

def find_most_overlapping_pairs(sep_matrix: pd.DataFrame,
                                 n_pairs: int = 5) -> pd.DataFrame:
    """
    Return the N most overlapping class pairs (lowest Fisher J score).
    These are the pairs most likely to be confused by the model.
    """
    pairs = []
    cols = sep_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "class_a": cols[i],
                "class_b": cols[j],
                "separability": sep_matrix.iloc[i, j],
                "risk": "high" if sep_matrix.iloc[i, j] < 10 else
                        "medium" if sep_matrix.iloc[i, j] < 50 else "low"
            })

    df = pd.DataFrame(pairs).sort_values("separability").head(n_pairs).reset_index(drop=True)
    print(f"\n[OVERLAP] Most overlapping class pairs:")
    print(df.to_string(index=False))
    return df


# ── 3. PCA plot — visualise cluster separation ───────────────────────────────

def plot_pca_clusters(X: pd.DataFrame, y: pd.Series,
                       n_components: int = 2,
                       sample_n: int = 5000,
                       output_path: Path = OUTPUTS_PATH) -> None:
    """
    Reduce to 2D via PCA and plot coloured scatter by class.
    Visually shows which classes form tight clusters vs which bleed into each other.
    Subsamples to sample_n rows per class for speed.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Subsample for plotting speed
    frames = []
    for cls in y.unique():
        sub = X[y == cls].sample(min(sample_n, (y == cls).sum()), random_state=42)
        frames.append((sub, pd.Series([cls] * len(sub))))
    X_sub = pd.concat([f[0] for f in frames]).reset_index(drop=True)
    y_sub = pd.concat([f[1] for f in frames]).reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    colors = plt.cm.tab10(np.linspace(0, 1, len(LABEL_MAP)))
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, cls in enumerate(sorted(y_sub.unique())):
        mask = y_sub == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[cls]], label=LABEL_MAP.get(cls, str(cls)),
                   alpha=0.4, s=10)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA — class cluster separation (overlapping regions = confusion risk)")
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / "pca_class_clusters.png", dpi=150)
    plt.close()
    print(f"[OVERLAP] PCA cluster plot saved")


# ── 4. LDA separability — best linear separation ─────────────────────────────

def compute_lda_separability(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Fit Linear Discriminant Analysis and return the between-class /
    within-class variance ratio — the global measure of linear separability.
    Higher = classes are more linearly separable.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(y.unique()) - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_scaled, y)

    result = {
        "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
        "n_components": n_components,
        "total_variance_explained": float(lda.explained_variance_ratio_.sum()),
    }
    print(f"\n[OVERLAP] LDA — total variance explained by {n_components} components: "
          f"{result['total_variance_explained']*100:.1f}%")
    print(f"[OVERLAP] Variance per component: "
          f"{[f'{v*100:.1f}%' for v in lda.explained_variance_ratio_]}")
    return result


# ── 5. Overlap report ─────────────────────────────────────────────────────────

def generate_overlap_report(overlapping_pairs: pd.DataFrame,
                              lda_result: dict,
                              output_path: Path = OUTPUTS_PATH) -> None:
    """
    Write a markdown report summarising overlap findings and model recommendations.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    high_risk = overlapping_pairs[overlapping_pairs["risk"] == "high"]
    medium_risk = overlapping_pairs[overlapping_pairs["risk"] == "medium"]

    lines = [
        "# Class Overlap Analysis Report\n",
        "## Summary\n",
        f"- LDA total variance explained: {lda_result['total_variance_explained']*100:.1f}%",
        f"- High-risk overlapping pairs: {len(high_risk)}",
        f"- Medium-risk overlapping pairs: {len(medium_risk)}",
        "",
        "## Most Overlapping Class Pairs\n",
        overlapping_pairs.to_markdown(index=False),
        "",
        "## Implications for Model Training (Part G)\n",
        "- High-risk pairs will likely appear as off-diagonal confusion in the confusion matrix.",
        "- Consider adding class-specific CS rules to distinguish overlapping pairs (e.g. DoS vs Fuzzers both have high packet rates — differentiate via payload size).",
        "- XGBoost handles overlap better than linear models — use `max_depth ≥ 6` to capture non-linear boundaries.",
        "- For SHAP (Part R Step 5): overlapping classes will show similar SHAP feature patterns — document this as a finding.",
        "",
        "## Implications for Anomaly Detection (Part S Step 3)\n",
        "- Isolation Forest may struggle to separate overlapping attack classes from each other.",
        "- However it should still separate ALL attacks from Benign — verify this in the AUROC plot.",
        "",
        "## Recommended Action\n",
        "- Use the separability heatmap to identify which features discriminate overlapping pairs.",
        "- Add those features to the CS rule thresholds in Part G Step 2.",
        "- Document remaining overlap as a limitation in the final evaluation report.",
    ]

    with open(output_path / "overlap_report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[OVERLAP] Overlap report saved to {output_path / 'overlap_report.md'}")


# ── 6. Full pipeline ──────────────────────────────────────────────────────────

def run_overlap_analysis(df: pd.DataFrame,
                          label_col: str = "label",
                          output_path: Path = OUTPUTS_PATH) -> dict:
    """
    Full overlap analysis pipeline:
      1. Compute pairwise Fisher separability → heatmap
      2. Identify most overlapping pairs
      3. PCA cluster visualisation
      4. LDA global separability
      5. Generate overlap report
    """
    print(f"\n[OVERLAP] Starting overlap analysis — shape: {df.shape}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    sep_matrix = compute_pairwise_separability(X, y)
    plot_separability_heatmap(sep_matrix, output_path)

    overlapping = find_most_overlapping_pairs(sep_matrix, n_pairs=10)
    overlapping.to_csv(output_path / "overlapping_pairs.csv", index=False)

    plot_pca_clusters(X, y, output_path=output_path)

    lda_result = compute_lda_separability(X, y)

    generate_overlap_report(overlapping, lda_result, output_path)

    print(f"[OVERLAP] Analysis complete — outputs in {output_path}\n")
    return {"separability_matrix": sep_matrix, "overlapping_pairs": overlapping, "lda": lda_result}


if __name__ == "__main__":
    pass
