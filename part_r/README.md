# Part R — Data Science: SHAP Analysis & Comparative Evaluation

## Overview

Part R owns **Phase 0a, Phase 0d, and Step 5** of the project pipeline.

| Phase / Step | Task | File |
|---|---|---|
| Phase 0a | Label Fix — **runs first of all** | `src/data_fixes/label_fix.py` |
| Phase 0d | Class Overlap Analysis | `src/data_fixes/overlap_analysis.py` |
| Step 5 | SHAP Analysis + Comparative Evaluation | `src/shap_analysis/shap_runner.py`, `src/evaluation/venn.py`, `src/evaluation/detection_rates.py` |

**Phase 0a has no dependencies** — Part R starts before anyone else.
**Step 5 is the last step** — requires `fusion_model.pkl` and `predictions.csv` from Part G.

---

## Label Mapping (from Readme.txt)
| Label | Class | Label | Class |
|---|---|---|---|
| 0 | Benign | 5 | Fuzzers |
| 1 | Analysis | 6 | Generic |
| 2 | Backdoor | 7 | Reconnaissance |
| 3 | DoS | 8 | Shellcode |
| 4 | Exploits | 9 | Worms |

---

## Phase 0a — Label Fix ← Run FIRST of all

> **Run before:** everything — no dependencies
> **File:** `src/data_fixes/label_fix.py`
> **Output:** `shared/data/processed/label_fixed_data.csv`, `label_fix_report.json`

Fixes known labelling errors in the raw UNSW-NB15 / CICFlowMeter data
before any other member starts work. Share `label_fix_report.json` with all members
so everyone knows what changed.

### Known problems being fixed

| Problem | Details | Fix applied |
|---|---|---|
| Mislabelled port scans | Flows behaving like port scans labelled as Benign (Lanvin et al. 2023) | Detect via heuristic → relabel to Reconnaissance (label=7) |
| Conflicting duplicate timestamps | Same timestamp, different label — CICFlowMeter artefact (Engelen et al. 2021) | Keep majority-vote label; tie-break: prefer non-Benign |
| Zero-duration flows | Impossible connections — measurement error in CICFlowMeter | Drop entirely |

### Steps inside `run_label_fix_pipeline()`

| Step | Function | What it does |
|---|---|---|
| 1 | `drop_artefact_flows()` | Removes zero/negative duration flows |
| 2 | `fix_duplicate_timestamps()` | Resolves conflicting labels on same timestamp |
| 3 | `detect_portscan_candidates()` | Flags: high rate + short duration + high dst port + small pkt |
| 4 | `relabel_portscan_flows()` | Moves detected flows to Reconnaissance (label=7) |

```python
from src.data_fixes.label_fix import run_label_fix_pipeline, export_fixed

# ALWAYS run dry_run=True first to review what will change
fixed, report = run_label_fix_pipeline(raw_df, dry_run=True)

# Then apply once happy
fixed, report = run_label_fix_pipeline(raw_df, dry_run=False)

export_fixed(fixed, report)
# → shared/data/processed/label_fixed_data.csv
# → shared/data/processed/label_fix_report.json
```

---

## Phase 0d — Class Overlap Analysis

> **Run after:** Part S delivers `shared/data/processed/cleaned_data.csv`
> **File:** `src/data_fixes/overlap_analysis.py`
> **Output:** `outputs/overlap_report.md` — **Part G must read this before Step 2**

Identifies which attack classes share similar feature distributions so Part G can write
better-differentiated CS rules. This is analysis only — it does NOT modify any data.

### Steps inside `run_overlap_analysis()`

| Step | Function | Output |
|---|---|---|
| 1 | `compute_pairwise_separability()` | Fisher J score per class pair (low = more overlap) |
| 2 | `plot_separability_heatmap()` | `class_separability_heatmap.png` — dark = hard to separate |
| 3 | `find_most_overlapping_pairs()` | `overlapping_pairs.csv` with risk level per pair |
| 4 | `plot_pca_clusters()` | `pca_class_clusters.png` — 2D scatter of class cluster separation |
| 5 | `compute_lda_separability()` | Global linear separability ratio |
| 6 | `generate_overlap_report()` | `overlap_report.md` — recommendations for Part G |

```python
from src.data_fixes.overlap_analysis import run_overlap_analysis

results = run_overlap_analysis(cleaned_df)
# Outputs saved to: part_r/outputs/
# Share overlap_report.md with Part G before they finalise Step 2 thresholds
```

**Handoff to Part G:** Email or share `overlap_report.md` directly.
Part G uses it to tune CS rule thresholds for overlapping class pairs.

---

## Step 5 — SHAP Analysis & Comparative Evaluation ← Runs LAST

> **Run after:** Part G delivers `shared/outputs/fusion_model.pkl` + `predictions.csv`
> **Files:** `src/shap_analysis/shap_runner.py`, `src/evaluation/venn.py`, `src/evaluation/detection_rates.py`
> **Output:** `outputs/domain_attribution.csv`, `outputs/venn_diagram.png`, `outputs/evaluation_report.md`

This is where the core research question is answered:
*For each attack type, did the ML model rely more on CS rules or the DS anomaly score?*

### 5.1 SHAP Value Computation

```python
from src.shap_analysis.shap_runner import load_model_and_data, compute_shap_values, plot_global_summary

model, X, y_true, y_pred = load_model_and_data()
explainer, shap_values = compute_shap_values(model, X)
plot_global_summary(shap_values, X)   # → shap_summary_plot.png
```

### 5.2 Domain Attribution Per Attack Class

For each class: `CS_weight% = sum(|SHAP| for cs_* cols) / sum(|SHAP| all cols) × 100`

```python
from src.shap_analysis.shap_runner import compute_domain_attribution

attribution = compute_domain_attribution(shap_values, X, y_true)
# Expected: DDoS → 72% CS / 18% DS  |  Zero-day → 8% CS / 81% DS
attribution.to_csv("outputs/domain_attribution.csv", index=False)
```

### 5.3 Venn Diagram of Detections

| Region | Definition | Meaning |
|---|---|---|
| CS Only | `cs_any_flag=1` AND `ds_score < threshold` | Known attacks — rules fire, DS score low |
| DS Only | `cs_any_flag=0` AND `ds_score >= threshold` | Novel / zero-day — no rule fires, but anomalous |
| Both | `cs_any_flag=1` AND `ds_score >= threshold` | High confidence — rules fired AND anomalous |
| Neither | `cs_any_flag=0` AND `ds_score < threshold` | Missed by both |

```python
from src.evaluation.venn import compute_detection_sets, plot_venn

sets = compute_detection_sets(df, ds_threshold=0.5)
plot_venn(sets)   # → venn_diagram.png
```

### 5.4 Detection Rate Comparison

```python
from src.evaluation.detection_rates import compute_detection_rates

rates = compute_detection_rates(df, ds_threshold=0.5)
# Per-class: CS-only rate vs DS-only rate vs Fusion accuracy
```

### 5.5 Evaluation Report

Write `outputs/evaluation_report.md` answering:
1. Which domain does the ML model trust more overall?
2. Which attack types are best covered by CS rules vs DS anomaly?
3. What is the incremental value of the fusion over either layer alone?
4. For a real IDS deployment, how should thresholds be set?

---

## Notebook Run Order

| Notebook | Phase/Step | What to do |
|---|---|---|
| `00_label_fix.ipynb` | Phase 0a | `run_label_fix_pipeline(dry_run=True)` → review → `dry_run=False` |
| `01_overlap_analysis.ipynb` | Phase 0d | `run_overlap_analysis()` — heatmap + PCA — share `overlap_report.md` with Part G |
| `02_shap_global_summary.ipynb` | Step 5.1 | Load model + predictions, compute SHAP, global beeswarm plot |
| `03_domain_attribution.ipynb` | Step 5.2 | CS% vs DS% per class — grouped bar chart, export `domain_attribution.csv` |
| `04_venn_diagram.ipynb` | Step 5.3 | `compute_detection_sets()` + `plot_venn()` |
| `05_detection_rate_table.ipynb` | Step 5.4 | `compute_detection_rates()` — CS vs DS vs Fusion table |
| `06_evaluation_report.ipynb` | Step 5.5 | Write final findings and recommendations |

---

## Inputs Required from Other Parts

| File | From | Needed for |
|---|---|---|
| `cleaned_data.csv` | Part S — Phase 0b | Phase 0d (overlap analysis) |
| `eda_summary.md` | Part S — Step 1 | Step 5 context |
| `features_with_cs.csv` | Part G — Step 2 | Step 5 detection sets |
| `fusion_model.pkl` | Part G — Step 4 | Step 5 SHAP |
| `predictions.csv` | Part G — Step 4 | Step 5 all evaluation |

## Deliverables

| File | Saved to | Consumed by |
|---|---|---|
| `label_fixed_data.csv` | `shared/data/processed/` | Part S Phase 0b |
| `label_fix_report.json` | `shared/data/processed/` | All members — review |
| `overlap_report.md` | `outputs/` | Part G — Step 2 (read before rules) |
| `overlapping_pairs.csv` | `outputs/` | Part G — Step 2 reference |
| `domain_attribution.csv` | `outputs/` | Final report |
| `venn_diagram.png` | `outputs/` | Final report |
| `evaluation_report.md` | `outputs/` | Final report |

---

## Tech Stack

```
pandas, numpy, matplotlib, seaborn
shap (TreeExplainer, summary_plot)
matplotlib-venn
scikit-learn (PCA, LDA, StandardScaler)
xgboost, joblib
jupyter notebook
```

---

## Key Questions Part R Must Answer

1. For each attack type, does the ML model rely more on CS rule flags or the DS anomaly score?
2. Which attacks are caught exclusively by CS rules that the anomaly detector misses?
3. Which attacks are caught exclusively by DS anomaly (novel/zero-day) that CS rules miss?
4. What is the quantified accuracy improvement of the fusion over either layer alone?
5. Where should a real-world IDS set the DS threshold to minimise false positives?
