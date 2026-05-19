# Part S — Data Science: Deep EDA & Statistical Anomaly Detection

## Overview

Part S is responsible for deeply understanding the dataset before any model is trained, and for building the **statistical anomaly detection layer** (Step 1 and Step 3 of the project pipeline). This is purely data-driven work — no cybersecurity rules, no ML model training.

---

## Datasets

| File | Description |
|---|---|
| `Data.csv` | Raw network flow features from CICFlowMeter |
| `Label.csv` | Ground-truth labels (0–9, see below) |
| `CICFlowMeter_out` | Additional CICFlowMeter output for enriched features |

### Label Mapping (from `Readme.txt`)
| Label | Class |
|---|---|
| 0 | Benign |
| 1 | Analysis |
| 2 | Backdoor |
| 3 | DoS |
| 4 | Exploits |
| 5 | Fuzzers |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

---

## Responsibilities

### Step 1 — Deep Exploratory Data Analysis (EDA) on CICIDS2017 + CICFlowMeter

> *"You don't just clean the data — you analyse it."*

#### 1.1 Data Loading & Cleaning
- Load `Data.csv` and `Label.csv`, merge on index
- Check for nulls, infinite values, duplicate rows
- Standardise column names (strip whitespace, lowercase)
- Handle class imbalance: compute per-class sample counts and ratios

#### 1.2 Distribution Analysis
- Plot feature distributions (histograms, KDE plots) for all numeric columns
- Identify skewed features; apply log-transform where appropriate
- Box plots per feature split by attack class

#### 1.3 Class Imbalance Analysis
- Bar chart of class frequencies (Benign vs each attack type)
- Compute imbalance ratio for each class vs Benign
- Document which classes are severely underrepresented (e.g., Worms, Shellcode)
- Recommend sampling strategy (SMOTE / undersampling) for Part G's ML model

#### 1.4 Correlation & Feature Relationships
- Pearson and Spearman correlation matrices (full feature set)
- Heatmap highlighting top correlated feature pairs
- Variance Inflation Factor (VIF) for multicollinearity detection
- Identify and flag redundant features

#### 1.5 Feature Importance (Pre-Model)
- Use **mutual information** scores between each feature and the label
- Use **ANOVA F-scores** for class separability
- Rank features by importance and export a ranked feature list (`feature_rankings.csv`) for Part G

#### 1.6 CICFlowMeter Integration
- Load `CICFlowMeter_out` and align columns with `Data.csv`
- Identify any additional features available only in CICFlowMeter output
- Document feature overlap and any newly derived features

---

### Step 3 — Statistical Anomaly Detection Layer

> *"Flags statistically unusual traffic patterns from pure data analysis — no CS rules involved."*

This layer runs **independently** from cybersecurity rules. Its output (`ds_anomaly_score`) is a feature fed into Part G's fusion model.

#### 3.1 Isolation Forest
- Train `IsolationForest` on the full feature set (unsupervised)
- Tune `contamination` parameter using the class imbalance ratio from Step 1
- Generate an anomaly score for every row: values closer to 1.0 = more anomalous
- Export scores as `ds_anomaly_score` column

#### 3.2 Autoencoder (Optional / Bonus)
- Build a shallow autoencoder (3–5 layers) using Keras/PyTorch
- Train on Benign-only traffic to learn "normal" reconstruction
- Reconstruction error = anomaly score for unseen traffic
- Compare Autoencoder scores vs Isolation Forest scores

#### 3.3 Anomaly Score Validation
- Plot distribution of `ds_anomaly_score` per true label class
- Verify that known attacks score higher than Benign traffic
- Compute AUROC of anomaly score vs binary (Benign / Attack) label
- Export final `ds_anomaly_score` as a column in `features_with_ds.csv`

---

## Deliverables

| File | Description |
|---|---|
| `eda_report.ipynb` | Full EDA notebook with all plots |
| `feature_rankings.csv` | Mutual info + ANOVA ranked features for Part G |
| `class_imbalance_report.md` | Class counts, ratios, recommended strategy |
| `features_with_ds.csv` | Original features + `ds_anomaly_score` column |
| `isolation_forest_model.pkl` | Saved Isolation Forest model |
| `eda_summary.md` | Key findings in plain text for Part G and Part R to read |

---

## Handoff to Other Parts

- **→ Part R**: Share `eda_report.ipynb` and `eda_summary.md` so Part R can understand the data before writing CS rules
- **→ Part G**: Share `feature_rankings.csv` and `features_with_ds.csv` — these are direct inputs to the fusion model

---

## Tech Stack

```
pandas, numpy, matplotlib, seaborn
scikit-learn (IsolationForest, mutual_info_classif, f_classif)
statsmodels (VIF)
tensorflow/keras or pytorch (Autoencoder — optional)
jupyter notebook
```

---

## Key Questions This Part Must Answer

1. Which features have the highest predictive signal before any model is trained?
2. How severe is class imbalance and what does it mean for model training?
3. Which traffic patterns are statistically unusual purely from a data perspective?
4. Does the `ds_anomaly_score` correctly flag known attacks even without labels?
