# Part G — Cybersecurity Rules & ML Fusion Model

## Overview

Part G covers **Step 2 and Step 4** of the project pipeline: writing explicit cybersecurity (CS) detection rules inspired by real-world IDS systems (like Snort), and building the **fusion ML model** that combines CS rule outputs with DS anomaly scores into a final multi-class classifier.

This part sits at the intersection of cybersecurity domain knowledge and machine learning engineering.

---

## Datasets & Inputs

| Source | File | Description |
|---|---|---|
| Raw | `Data.csv` | Raw network flow features |
| Raw | `Label.csv` | Ground-truth labels (0–9) |
| Raw | `CICFlowMeter_out` | Additional CICFlowMeter features |
| Part S | `features_with_ds.csv` | Features + `ds_anomaly_score` (required before Step 4) |
| Part S | `feature_rankings.csv` | Top features by mutual info / ANOVA (for model input selection) |

### Label Reference (from `Readme.txt`)
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

### Step 2 — Rule-Based Detection Layer (CS Rules)

> *"Using known attack signatures (Snort-style rules), write explicit CS rules."*

These rules are **hand-coded domain knowledge** — not learned from data. They simulate what a real-world IDS (Intrusion Detection System) would fire on known attack patterns.

#### 2.1 Rule Design

Implement the following CS rules as Python functions applied row-wise to the dataframe:

**CS Rule 1 — DDoS Check**
```
IF SYN rate > 1000 packets/sec THEN cs_ddos_flag = 1
```
- Map to CICFlowMeter columns: `SYN Flag Count`, `Flow Duration`, `Flow Packets/s`
- Primary target labels: DoS (3), Generic (6)

**CS Rule 2 — Port Scan Check**
```
IF unique destination ports > 100/sec THEN cs_portscan_flag = 1
```
- Map to CICFlowMeter columns: `Destination Port`, `Flow Duration`
- Derive: unique dst ports contacted per source IP per time window
- Primary target labels: Reconnaissance (7), Analysis (1)

**CS Rule 3 — Small Packet Flood Check**
```
IF packet size < 60 bytes AND rate > 500 packets/sec THEN cs_flood_flag = 1
```
- Map to CICFlowMeter columns: `Average Packet Size`, `Flow Packets/s`
- Primary target labels: Fuzzers (5), DoS (3)

#### 2.2 Additional Rules (Recommended)

Extend coverage for remaining attack classes:

| Rule | Condition | Target Class |
|---|---|---|
| `cs_exploit_flag` | High payload size + unusual protocol flags | Exploits (4) |
| `cs_backdoor_flag` | Long-duration low-rate flows on unusual ports | Backdoor (2) |
| `cs_shellcode_flag` | Very small flow byte count + non-HTTP port | Shellcode (8) |
| `cs_worm_flag` | High unique destination IPs from same source | Worms (9) |

#### 2.3 Rule Evaluation (Pre-Fusion)
- Compute per-rule precision, recall, F1 against true labels
- Identify false positive rate for each rule on Benign traffic
- Document which attack classes have **no CS rule coverage** (these will rely on DS layer)
- Export: `cs_rule_evaluation.csv`

#### 2.4 Feature Engineering: CS Flag Columns
After applying all rules, the dataframe gains new binary columns:
```
cs_ddos_flag | cs_portscan_flag | cs_flood_flag | cs_exploit_flag | cs_backdoor_flag | ...
```
Export full dataframe as `features_with_cs.csv` — this is the input to Step 4.

---

### Step 4 — Fusion ML Model

> *"XGBoost / Random Forest takes CS rule outputs + DS anomaly scores as features. ML learns which combination is most predictive — neither domain is hardcoded as better."*

#### 4.1 Feature Matrix Construction

Combine all signal sets into one feature matrix:

```
[Original features from Data.csv]
+ [ds_anomaly_score from Part S]
+ [cs_ddos_flag, cs_portscan_flag, cs_flood_flag, ... from Step 2]
= Final feature matrix X
```

- Use `feature_rankings.csv` from Part S to optionally restrict to top-N original features
- Target vector `y` = labels from `Label.csv` (10 classes)

#### 4.2 Train/Test Split
- Stratified split: 70% train / 15% validation / 15% test
- Preserve class distribution across splits (critical due to imbalance)
- Apply SMOTE or class_weight balancing as recommended by Part S

#### 4.3 Model Training

**Primary: XGBoost Classifier**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    scale_pos_weight=...  # handle imbalance
)
```

**Secondary: Random Forest Classifier**
```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_features='sqrt'
)
```

- Train both, compare on validation set, select best for SHAP (Part R)
- Use `cross_val_score` (5-fold stratified) for robust evaluation

#### 4.4 Hyperparameter Tuning
- `GridSearchCV` or `Optuna` for XGBoost hyperparameter search
- Key parameters to tune: `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`

#### 4.5 Model Evaluation
Produce the following on the test set:
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap (10×10)
- Macro and weighted F1 scores
- ROC-AUC per class (one-vs-rest)
- Compare: Original features only vs CS-only vs DS-only vs Fusion (ablation study)

#### 4.6 Prediction Export
- Export `predictions.csv` with columns: `true_label`, `predicted_label`, `confidence`, all CS flags, `ds_anomaly_score`
- Save trained model: `fusion_model.pkl`

---

## Deliverables

| File | Description |
|---|---|
| `cs_rules.py` | Python module with all CS rule functions |
| `cs_rule_evaluation.csv` | Per-rule precision/recall/F1 on test data |
| `features_with_cs.csv` | Data.csv + all CS flag columns |
| `fusion_model.ipynb` | Training notebook with all ablation experiments |
| `fusion_model.pkl` | Saved final model (for Part R SHAP) |
| `predictions.csv` | Per-row predictions + all flag columns |
| `confusion_matrix.png` | 10×10 confusion matrix heatmap |
| `model_evaluation.md` | Written model performance summary |

---

## Handoff / Dependencies

**Part G depends on:**
- ✅ Part S: `features_with_ds.csv` and `feature_rankings.csv` (needed before Step 4)

**Part G delivers to:**
- → Part R: `fusion_model.pkl`, `features_with_cs.csv`, `predictions.csv`

**Step 2 (CS Rules) can start immediately** — no dependency on Part S.  
**Step 4 (Fusion Model) starts after** Part S delivers `features_with_ds.csv`.

---

## Tech Stack

```
pandas, numpy
scikit-learn (RandomForestClassifier, train_test_split, StratifiedKFold, SMOTE)
xgboost (XGBClassifier)
imbalanced-learn (SMOTE, RandomUnderSampler)
optuna or sklearn GridSearchCV (hyperparameter tuning)
matplotlib, seaborn (confusion matrix, evaluation plots)
joblib (model serialisation)
jupyter notebook
```

---

## Key Questions This Part Must Answer

1. Which CS rules have the highest precision and which have the highest false positive rate on Benign traffic?
2. Which attack classes have zero CS rule coverage (fully reliant on DS anomaly score)?
3. Does the fusion model (CS + DS + features) outperform a model trained on features alone?
4. In the ablation study, what is the marginal gain of adding CS flags vs DS score individually?
5. Which classes does the model confuse most often and why (from the confusion matrix)?
