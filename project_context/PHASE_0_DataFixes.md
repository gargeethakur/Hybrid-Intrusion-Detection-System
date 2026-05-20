# Phase 0: Data Problem Fixes (run before everything else)

> These fixes address the 4 known problems with UNSW-NB15 / CICFlowMeter data.
> **Must complete before any EDA, rule writing, or model training.**

---

## Problem 1 — Labelling Errors
**Owner:** Part R | **File:** `part_r/src/data_fixes/label_fix.py`

| Issue | Details | Fix |
|---|---|---|
| Mislabelled port scans | Flows behaving like port scans labelled as Benign (Lanvin et al. 2023) | Detect via rate + port + duration heuristic → relabel to Reconnaissance (7) |
| Conflicting duplicate timestamps | Same timestamp, different label — CICFlowMeter artefact (Engelen et al. 2021) | Keep majority-vote label per timestamp group |
| Zero-duration flows | Impossible real connections — tool measurement error | Drop entirely |

```python
from part_r.src.data_fixes.label_fix import run_label_fix_pipeline
fixed_df, report = run_label_fix_pipeline(raw_df, dry_run=True)  # preview first
fixed_df, report = run_label_fix_pipeline(raw_df, dry_run=False) # then apply
```

---

## Problem 2 — Severe Class Imbalance
**Owner:** Part S | **File:** `part_s/src/eda/imbalance.py`

| Class | Count | Fix |
|---|---|---|
| Benign | 2,218,761 | Undersample to 100,000 |
| Worms | 174 | SMOTE to 1,000 |
| Shellcode | 1,511 | SMOTE to 1,000 (already close, boosted) |
| Others | varies | SMOTE if below 1,000 |

```python
from part_s.src.eda.imbalance import run_balancing_pipeline
balanced_df = run_balancing_pipeline(cleaned_df, smote_min_samples=1000, undersample_max=100_000)
```

---

## Problem 3 — Class Overlap
**Owner:** Part R | **File:** `part_r/src/data_fixes/overlap_analysis.py`

This is an ANALYSIS step — it does not modify data. It identifies which class pairs
share similar features so Part G can write better-differentiated CS rules.

```python
from part_r.src.data_fixes.overlap_analysis import run_overlap_analysis
results = run_overlap_analysis(cleaned_df)
# Outputs: class_separability_heatmap.png, pca_class_clusters.png, overlap_report.md
```

Read `outputs/overlap_report.md` before writing CS rules in Part G Step 2.

---

## Problem 4 — Duplicated Feature (Fwd Header Length)
**Owner:** Part S | **File:** `part_s/src/eda/cleaner.py` → `drop_duplicate_columns()`

This is already handled automatically by the cleaning pipeline.
No extra action needed — `drop_duplicate_columns()` catches it.

---

## Correct Full Pipeline Order

```
[Part R]  Phase 0a: run_label_fix_pipeline()       → label_fixed_data.csv
[Part S]  Phase 0b: run_cleaning_pipeline()         → cleaned_data.csv  (drops dup Fwd Header Length)
[Part S]  Phase 0c: run_balancing_pipeline()        → balanced_data.csv
[Part R]  Phase 0d: run_overlap_analysis()          → overlap_report.md  (read before Phase 2)
[Part S]  Phase 1:  EDA notebooks
[Part G]  Phase 2:  CS Rules (informed by overlap_report.md)
[Part S]  Phase 3:  Isolation Forest → ds_anomaly_score
[Part G]  Phase 4:  Fusion ML model
[Part R]  Phase 5:  SHAP + evaluation
```
