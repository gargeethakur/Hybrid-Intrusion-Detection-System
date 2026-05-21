# Part S — Data Science: EDA & Anomaly Detection

## Overview

Part S owns **Phase 0b, Phase 0c, Step 1, and Step 3** of the project pipeline.

| Phase / Step | Task | File |
|---|---|---|
| Phase 0b | Data Cleaning | `src/eda/cleaner.py` |
| Phase 0c | Class Balancing (SMOTE) | `src/eda/imbalance.py` |
| Step 1 | Deep EDA + Feature Importance | `src/eda/loader.py`, `src/eda/feature_importance.py` |
| Step 3 | Isolation Forest Anomaly Detection | `src/anomaly/isolation_forest.py` |

**Dependency:** Part R must deliver `label_fixed_data.csv` (Phase 0a) before Part S starts Phase 0b.

---

## Datasets

| File | Description |
|---|---|
| `Data.csv` | Raw network flow features (UNSW-NB15 origin, 49 features) |
| `Label.csv` | Ground-truth labels (0–9, see Readme.txt) |
| `CICFlowMeter_out/` | Additional CICFlowMeter output for enriched features |

### Label Mapping
| Label | Class | Label | Class |
|---|---|---|---|
| 0 | Benign | 5 | Fuzzers |
| 1 | Analysis | 6 | Generic |
| 2 | Backdoor | 7 | Reconnaissance |
| 3 | DoS | 8 | Shellcode |
| 4 | Exploits | 9 | Worms |

---

## Phase 0b — Data Cleaning

> **Run after:** Part R delivers `shared/data/processed/label_fixed_data.csv`
> **File:** `src/eda/cleaner.py`
> **Output:** `shared/data/processed/cleaned_data.csv`

Fixes known data quality problems before any EDA or model work.

### Steps inside `run_cleaning_pipeline()`

| Step | Function | What it does |
|---|---|---|
| 1 | `drop_constant_columns()` | Removes zero-variance features — no signal |
| 2 | `drop_duplicate_columns()` | Catches the duplicate `Fwd Header Length` column bug automatically |
| 3 | `clip_outliers_iqr(factor=3.0)` | Clips to fence — does NOT delete rows |
| 4 | `fill_remaining_nulls()` | Fills leftover NaNs with column median |
| 5 | `fix_skewed_features(threshold=1.0)` | log1p on positive skewed columns |
| 6 | `scale_features(method='minmax')` | Normalise all features to [0, 1] |

```python
from src.eda.loader import load_and_merge, clean as basic_clean
from src.eda.cleaner import run_cleaning_pipeline, export_cleaned

df = load_and_merge()          # loads label_fixed_data.csv
df = basic_clean(df)           # nulls, inf, column name standardisation
df, scaler, log_cols = run_cleaning_pipeline(df)
export_cleaned(df)             # → shared/data/processed/cleaned_data.csv
```

---

## Phase 0c — Class Balancing

> **Run after:** Phase 0b delivers `cleaned_data.csv`
> **File:** `src/eda/imbalance.py`
> **Output:** `shared/data/processed/balanced_data.csv`

Fixes the severe class imbalance in UNSW-NB15 before model training.

**Why this is critical:**
- Benign: 2,218,761 records vs Worms: 174 records
- Without fixing this, the model will almost completely ignore Worms and Shellcode

### Steps inside `run_balancing_pipeline()`

| Step | Function | What it does |
|---|---|---|
| 1 | `diagnose_imbalance()` | Prints count, %, imbalance ratio per class |
| 2 | `apply_smote(min_samples=1000)` | SMOTE on all classes below 1,000 samples (Worms: 174→1,000) |
| 3 | `undersample_majority(max_majority=100_000)` | Caps Benign from 2.2M to 100,000 |

```python
from src.eda.imbalance import run_balancing_pipeline, export_balanced

balanced = run_balancing_pipeline(
    df,
    smote_min_samples=1000,   # Worms (174) and any class below get boosted
    undersample_max=100_000   # Benign reduced from 2.2M
)
export_balanced(balanced)     # → shared/data/processed/balanced_data.csv
```

---

## Step 1 — Deep EDA

> **Run after:** Phase 0c delivers `balanced_data.csv`
> **Files:** `src/eda/loader.py`, `src/eda/feature_importance.py`
> **Output:** `shared/outputs/feature_rankings.csv`, `shared/outputs/eda_summary.md`

### Tasks

#### 1.1 Data Loading & Inspection
- Load and merge `Data.csv` + `Label.csv`, standardise column names
- Report shape, dtypes, null counts, class distribution

#### 1.2 Distribution Analysis
- Histograms and KDE plots for all numeric features
- Box plots per feature split by attack class

#### 1.3 Class Imbalance Report
- Bar chart of class frequencies before and after Phase 0c
- Document imbalance ratios for reference in final report

#### 1.4 Correlation & Multicollinearity
- Pearson + Spearman correlation heatmap
- VIF (Variance Inflation Factor) — flag multicollinear features

#### 1.5 Feature Importance (Pre-Model)
- `compute_feature_rankings()` — mutual information + ANOVA F-score
- Export ranked list → `feature_rankings.csv` (Part G uses this for Step 4)

#### 1.6 CICFlowMeter Integration
- Load `CICFlowMeter_out/`, align with `Data.csv`, document new features

Write `eda_summary.md` with key findings — Part R reads this before Step 5.

---

## Step 3 — Isolation Forest Anomaly Detection

> **Run after:** Step 1 EDA complete
> **File:** `src/anomaly/isolation_forest.py`
> **Output:** `shared/outputs/features_with_ds.csv`, `outputs/isolation_forest_model.pkl`

Fully unsupervised — never sees labels. Trains on all features and assigns a
statistical anomaly score to every row.

### Tasks

| Step | Function | What it does |
|---|---|---|
| Train | `train_isolation_forest(X, contamination)` | Fit IsolationForest — contamination from Phase 0c imbalance ratio |
| Score | `score(model, X)` | Normalise raw scores to [0, 1] — 1.0 = most anomalous |
| Validate | plot per class | Verify attack classes score higher than Benign |
| Validate | AUROC | Compute AUROC of `ds_anomaly_score` vs binary (Benign / Attack) label |
| Export | `export_with_scores(df, scores)` | Appends `ds_anomaly_score` column → `features_with_ds.csv` |
| Save | `save_model(model)` | Saves `isolation_forest_model.pkl` |

**Optional:** Train Autoencoder on Benign-only traffic. Compare reconstruction error vs Isolation Forest scores.

---

## Notebook Run Order

| Notebook | Step | What to do |
|---|---|---|
| `01_data_loading.ipynb` | Phase 0b start | `load_and_merge()` + `basic_clean()` — check shape, dtypes, nulls |
| `02_data_cleaning.ipynb` | Phase 0b | `run_cleaning_pipeline()` — before/after distribution plots |
| `02b_class_imbalance_fix.ipynb` | Phase 0c | `run_balancing_pipeline()` — before/after class bar charts |
| `03_class_imbalance_report.ipynb` | Phase 0c | Detailed imbalance ratios, SMOTE effect visualisation |
| `04_feature_importance.ipynb` | Step 1 | `compute_feature_rankings()` — heatmap, export `feature_rankings.csv` |
| `05_isolation_forest.ipynb` | Step 3 | Train IF, plot `ds_anomaly_score` per class, compute AUROC |
| `06_autoencoder.ipynb` | Step 3 (optional) | Train autoencoder on Benign-only, compare vs IF scores |

---

## Deliverables

| File | Saved to | Consumed by |
|---|---|---|
| `cleaned_data.csv` | `shared/data/processed/` | Part S Phase 0c, Part R Phase 0d |
| `balanced_data.csv` | `shared/data/processed/` | Part G Step 4 |
| `feature_rankings.csv` | `shared/outputs/` | Part G Step 4 |
| `eda_summary.md` | `shared/outputs/` | Part R Step 5 |
| `features_with_ds.csv` | `shared/outputs/` | Part G Step 4 |
| `isolation_forest_model.pkl` | `outputs/` | Part R Step 5 (reference) |

---

## Tech Stack

```
pandas, numpy, matplotlib, seaborn
scikit-learn (IsolationForest, mutual_info_classif, f_classif)
imbalanced-learn (SMOTE, RandomUnderSampler)
statsmodels (VIF)
tensorflow/keras or pytorch (Autoencoder — optional)
jupyter notebook
```

---

## Key Questions Part S Must Answer

1. Which features have the highest predictive signal before any model is trained?
2. How severe is class imbalance and what does it mean for model training?
3. Does `ds_anomaly_score` correctly flag known attacks even without labels?
4. Which features are multicollinear and should be dropped before Part G's model?
