# Hybrid IDS — CS Rules + DS Anomaly Detection + ML Fusion

## Project Goal

Build a hybrid Intrusion Detection System (IDS) on the CICIDS / CICFlowMeter dataset that combines:
- **Cybersecurity domain rules** (explicit Snort-style signatures)
- **Statistical anomaly detection** (Isolation Forest — unsupervised, data-driven)
- **ML fusion model** (XGBoost/RF learns which signal to trust per attack type)
- **SHAP explainability** (quantifies CS vs DS contribution per attack class)

**Core research question:** *For each attack type, does the ML model rely more on CS rules or the DS anomaly score — and why?*

---

## Dataset

| File | Description |
|---|---|
| `Data.csv` | Network flow features (CICFlowMeter format) |
| `Label.csv` | Ground-truth labels (10 classes, see Readme.txt) |
| `CICFlowMeter_out/` | Extended CICFlowMeter output |

Place all raw data in `shared/data/raw/` before running any part.

### Attack Classes

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | Benign | 5 | Fuzzers |
| 1 | Analysis | 6 | Generic |
| 2 | Backdoor | 7 | Reconnaissance |
| 3 | DoS | 8 | Shellcode |
| 4 | Exploits | 9 | Worms |

---

## Team Structure

| Folder | Part | Role | Pipeline Steps |
|---|---|---|---|
| `part_s/` | Part S | Data Science — EDA & Anomaly Detection | Steps 1 + 3 |
| `part_r/` | Part R | Data Science — SHAP & Evaluation | Step 5 |
| `part_g/` | Part G | Cybersecurity Rules & ML Fusion | Steps 2 + 4 |

---

## Pipeline

```
Step 1  [Part S]  Deep EDA on CICIDS + CICFlowMeter
Step 2  [Part G]  Write Snort-style CS detection rules     ← runs parallel to Step 1
Step 3  [Part S]  Train Isolation Forest → ds_anomaly_score
Step 4  [Part G]  Train XGBoost/RF fusion model (CS flags + DS score + features)
Step 5  [Part R]  SHAP analysis + Venn diagram + comparative evaluation
```

## Dependency Graph

```
Part S (Steps 1+3)
    ├── feature_rankings.csv  ──────────────────────┐
    └── features_with_ds.csv  ──────────────────────┤
                                                    ▼
Part G (Step 2) → features_with_cs.csv ──→  Part G (Step 4) → fusion_model.pkl ──→ Part R (Step 5)
                                                                predictions.csv  ──→ Part R (Step 5)
Part S (eda_summary.md) ────────────────────────────────────────────────────────→ Part R (context)
```

**Parallel start:** Part S Step 1 (EDA) and Part G Step 2 (CS Rules) can begin simultaneously.

---

## Shared Outputs (place in `shared/outputs/`)

| File | Produced By | Consumed By |
|---|---|---|
| `feature_rankings.csv` | Part S | Part G |
| `features_with_ds.csv` | Part S | Part G, Part R |
| `eda_summary.md` | Part S | Part R |
| `features_with_cs.csv` | Part G | Part R |
| `fusion_model.pkl` | Part G | Part R |
| `predictions.csv` | Part G | Part R |

---

## Project Context

See `project_context/` for detailed phase documents:
- `PHASE_1_EDA.md` — Step 1 plan (Part S)
- `PHASE_2_CS_Rules.md` — Step 2 plan (Part G)
- `PHASE_3_Anomaly.md` — Step 3 plan (Part S)
- `PHASE_4_Fusion.md` — Step 4 plan (Part G)
- `PHASE_5_SHAP.md` — Step 5 plan (Part R)
- `architecture.md` — Full system architecture
- `decisions.md` — Key design decisions and rationale

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd hybrid_ids

# Each part has its own requirements
pip install -r part_s/requirements.txt
pip install -r part_r/requirements.txt
pip install -r part_g/requirements.txt
```

Each part's README has detailed run instructions.
