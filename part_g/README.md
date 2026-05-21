# Part G — Cybersecurity Rules & ML Fusion Model

## Overview

Part G owns **Step 2 and Step 4** of the project pipeline.

| Step | Task | File |
|---|---|---|
| Step 2 | CS Rule Layer — Snort-style detection rules | `src/cs_rules/rules.py`, `src/cs_rules/evaluate.py` |
| Step 4 | Fusion ML Model — XGBoost/RF on all signals | `src/fusion_model/train.py` |

**Step 2 can start on Day 1** — no dependencies on any other part.
Read `part_r/outputs/overlap_report.md` when Part R delivers it before finalising rule thresholds.

**Step 4 requires:**
- `features_with_ds.csv` from Part S (Step 3)
- `feature_rankings.csv` from Part S (Step 1)
- `balanced_data.csv` from Part S (Phase 0c)

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

## Step 2 — CS Rule Layer ← Can Start Day 1

> **Run:** Day 1 — no dependencies
> **Files:** `src/cs_rules/rules.py`, `src/cs_rules/evaluate.py`
> **Output:** `shared/outputs/features_with_cs.csv`, `outputs/cs_rule_evaluation.csv`

These rules are **hand-coded domain knowledge — NOT learned from data.**
They simulate what a real-world IDS (like Snort) fires on known attack patterns.

### Before writing rules — read Part R's overlap report

When Part R delivers `part_r/outputs/overlap_report.md`, read it before finalising
thresholds. The overlap analysis tells you which class pairs share similar feature
distributions so you can write rules that better differentiate them.

### The 7 CS Rules

| Rule | Flag Column | Condition | Target Classes |
|---|---|---|---|
| Rule 1 | `cs_ddos_flag` | SYN rate > 1000 pkt/sec | DoS (3), Generic (6) |
| Rule 2 | `cs_portscan_flag` | unique dst ports > 100/sec | Reconnaissance (7), Analysis (1) |
| Rule 3 | `cs_flood_flag` | avg pkt size < 60 bytes AND rate > 500/sec | Fuzzers (5), DoS (3) |
| Rule 4 | `cs_exploit_flag` | payload near MTU OR URG flags set | Exploits (4) |
| Rule 5 | `cs_backdoor_flag` | long duration + low rate + non-standard port | Backdoor (2) |
| Rule 6 | `cs_shellcode_flag` | very small total bytes + non-HTTP port | Shellcode (8) |
| Rule 7 | `cs_worm_flag` | high rate + tiny packets (scanning proxy) | Worms (9) |

`cs_any_flag` is added automatically — it equals the max of all 7 flags per row.

```python
from src.cs_rules.rules import apply_all_rules

df = apply_all_rules(df)
# Adds: cs_ddos_flag, cs_portscan_flag, cs_flood_flag,
#       cs_exploit_flag, cs_backdoor_flag, cs_shellcode_flag,
#       cs_worm_flag, cs_any_flag
```

### Rule Evaluation

```python
from src.cs_rules.evaluate import evaluate_rules

report = evaluate_rules(df, label_col="label")
# Per-rule: precision, recall, F1, false positive count on Benign
report.to_csv("outputs/cs_rule_evaluation.csv", index=False)
```

**Document which attack classes have zero CS rule coverage** — those classes
will rely entirely on the DS anomaly score in the fusion model.

### Export

```python
df.to_csv("../../shared/outputs/features_with_cs.csv", index=False)
```

---

## Step 4 — Fusion ML Model

> **Run after:** Part S delivers `features_with_ds.csv` AND `feature_rankings.csv`
> **File:** `src/fusion_model/train.py`
> **Output:** `shared/outputs/fusion_model.pkl`, `shared/outputs/predictions.csv`, `outputs/confusion_matrix.png`

Trains XGBoost and Random Forest on the combined signal set:
original features + cs_* flags + ds_anomaly_score.
The model learns which combination is most predictive — neither domain is hardcoded as better.

### 4.1 Feature Matrix

```
[Original features from balanced_data.csv]
+ [ds_anomaly_score]          ← from Part S Step 3
+ [cs_ddos_flag, cs_portscan_flag, cs_flood_flag, ...]   ← from Step 2
= Final feature matrix X
```

Use `feature_rankings.csv` from Part S to optionally keep only top-N original features.

### 4.2 Train/Test Split

- Stratified 70% train / 15% validation / 15% test
- Preserves class distribution across all splits (critical given imbalance)

### 4.3 Models

**Primary — XGBoost:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
```

**Secondary — Random Forest:**
```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
```

Train both, compare on validation set, use the better one for SHAP in Part R.

### 4.4 Hyperparameter Tuning

Use Optuna or `GridSearchCV` to tune `max_depth`, `n_estimators`, `subsample`,
`colsample_bytree` on the validation set.

### 4.5 Ablation Study

Train 4 versions and compare macro F1 scores:

| Version | Features used |
|---|---|
| (a) Features only | Original features — no CS flags, no DS score |
| (b) CS only | CS flags only |
| (c) DS only | ds_anomaly_score only |
| (d) Full fusion | Original features + CS flags + DS score |

This directly answers: what does each domain contribute to the final accuracy?

### 4.6 Evaluation

On the test set, produce:
- Classification report (precision, recall, F1 per class)
- 10×10 confusion matrix heatmap
- ROC-AUC per class (one-vs-rest)

### 4.7 Export

```python
# Save model for Part R SHAP analysis
joblib.dump(model, "../../shared/outputs/fusion_model.pkl")

# predictions.csv must include these columns for Part R:
# true_label, predicted_label, confidence,
# all cs_* flag columns, ds_anomaly_score
predictions.to_csv("../../shared/outputs/predictions.csv", index=False)
```

---

## Notebook Run Order

| Notebook | Step | What to do |
|---|---|---|
| `01_cs_rule_development.ipynb` | Step 2 | Apply each rule, check how many rows each fires on, check false positive rate on Benign |
| `02_rule_evaluation.ipynb` | Step 2 | `evaluate_rules()` — precision/recall/F1 table, document gaps |
| `03_fusion_model_training.ipynb` | Step 4 | Build feature matrix, train XGBoost + RF, compare on validation set |
| `04_ablation_study.ipynb` | Step 4 | Train 4 versions, bar chart of macro F1 for each |
| `05_confusion_matrix.ipynb` | Step 4 | 10×10 confusion matrix heatmap + ROC-AUC per class |

---

## Inputs Required from Other Parts

| File | From | Needed for |
|---|---|---|
| `overlap_report.md` | Part R — Phase 0d | Step 2 — read before finalising rule thresholds |
| `balanced_data.csv` | Part S — Phase 0c | Step 4 — training data |
| `feature_rankings.csv` | Part S — Step 1 | Step 4 — feature selection |
| `features_with_ds.csv` | Part S — Step 3 | Step 4 — DS anomaly score column |

## Deliverables

| File | Saved to | Consumed by |
|---|---|---|
| `features_with_cs.csv` | `shared/outputs/` | Part G Step 4, Part R Step 5 |
| `cs_rule_evaluation.csv` | `outputs/` | Final report |
| `fusion_model.pkl` | `shared/outputs/` | Part R Step 5 (SHAP) |
| `predictions.csv` | `shared/outputs/` | Part R Step 5 |
| `confusion_matrix.png` | `outputs/` | Final report |
| `model_evaluation.md` | `outputs/` | Final report |

---

## Tech Stack

```
pandas, numpy
scikit-learn (train_test_split, StratifiedKFold, classification_report)
xgboost (XGBClassifier)
imbalanced-learn (SMOTE — if additional balancing needed)
optuna or sklearn GridSearchCV
matplotlib, seaborn (confusion matrix, evaluation plots)
joblib (model serialisation)
jupyter notebook
```

---

## Key Questions Part G Must Answer

1. Which CS rules have the highest precision and which have the highest false positive rate on Benign?
2. Which attack classes have zero CS rule coverage (fully reliant on DS anomaly score)?
3. Does the fusion model outperform a model trained on original features alone?
4. In the ablation study, what is the marginal gain of adding CS flags vs DS score individually?
5. Which classes does the model confuse most often (from the confusion matrix)?
