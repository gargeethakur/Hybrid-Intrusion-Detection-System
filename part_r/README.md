# Part R — Data Science: SHAP Analysis & Comparative Evaluation

## Overview

Part R is responsible for **Step 5** of the project pipeline: post-model explainability, SHAP-based domain attribution, and producing the final comparative evaluation between the CS rule layer (Part G) and the DS anomaly layer (Part S). This is where the project's core research question is answered quantitatively.

---

## Datasets & Inputs

Part R receives outputs from **all other parts** before starting:

| Source | File | Description |
|---|---|---|
| Part S | `features_with_ds.csv` | Original features + `ds_anomaly_score` |
| Part G | `features_with_cs.csv` | Features + CS rule flags (`cs_ddos_flag`, `cs_portscan_flag`, `cs_flood_flag`) |
| Part G | `fusion_model.pkl` | Trained XGBoost / Random Forest fusion model |
| Part G | `predictions.csv` | ML model predictions per row |
| Part S | `eda_summary.md` | Context about the data for interpreting results |

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

### Step 5 — SHAP Analysis & Comparative Evaluation

> *"Use SHAP to show which domain's signals the ML model relied on more per attack type. Produce a Venn diagram: CS-only detections, DS-only detections, both detected."*

---

#### 5.1 SHAP Value Computation

- Load `fusion_model.pkl` and the full feature matrix (CS flags + DS score + original features)
- Use `shap.TreeExplainer` (for XGBoost/RF) to compute SHAP values for every prediction
- Produce:
  - **Global SHAP summary plot** (beeswarm) — all features ranked by mean |SHAP|
  - **SHAP bar chart** — top 20 most impactful features
  - **Per-class SHAP breakdown** — for each of the 10 attack classes, which features drove the prediction

#### 5.2 Domain Attribution per Attack Type

This is the key research output. For each attack class, compute the **relative weight of CS signals vs DS signals**:

```
CS weight (%) = sum(|SHAP| for cs_* columns) / sum(|SHAP| all columns) × 100
DS weight (%) = sum(|SHAP| for ds_anomaly_score) / sum(|SHAP| all columns) × 100
```

Expected outputs (matching the project design):

| Attack Type | CS Weight | DS Weight |
|---|---|---|
| DDoS | ~72% | ~18% |
| Zero-day / Novel | ~8% | ~81% |
| Port Scan | ~65% | ~24% |

- Produce a **grouped bar chart** showing CS vs DS weight for every class
- Identify which attacks the ML model treats as "rule-driven" vs "anomaly-driven"

#### 5.3 Venn Diagram of Detections

Compute three detection sets **before the ML model** (purely from flags):

| Region | Definition |
|---|---|
| **CS Only** | `cs_any_flag == 1` AND `ds_anomaly_score < threshold` |
| **DS Only** | `cs_any_flag == 0` AND `ds_anomaly_score >= threshold` |
| **Both CS + DS** | `cs_any_flag == 1` AND `ds_anomaly_score >= threshold` |
| **Neither** | `cs_any_flag == 0` AND `ds_anomaly_score < threshold` |

- Use `matplotlib_venn` or `matplotlib patches` to draw a proper Venn diagram
- Annotate each region with sample count and most common true attack class
- **CS Only → Known attacks** (rules fire, DS score low)
- **DS Only → Novel / zero-day** (no rule fires, but statistically unusual)
- **Both → High confidence** (rules fired AND statistically unusual)

#### 5.4 Quantified Comparison: What Does Each Domain Catch That the Other Misses?

> *"Exactly which attack types does CS catch that DS misses — and why."*

- For each attack class, compute:
  - Detection rate by CS rules alone (at 0 DS threshold)
  - Detection rate by DS anomaly alone (at chosen threshold)
  - Detection rate by fusion model
- Produce a **detection rate table** and heatmap
- Identify specific attack types where:
  - CS-only misses (zero-day, novel variants) → DS score saves the day
  - DS-only misses (high-volume known attacks) → CS rules save the day

#### 5.5 Final Summary Report

Produce a written `evaluation_report.md` covering:
1. Which domain does the ML model trust more overall?
2. Which attack types are best covered by CS rules vs DS anomaly?
3. What is the incremental value of combining both (fusion) over using either alone?
4. Recommendations: for a real IDS deployment, how should thresholds be set?

---

## Deliverables

| File | Description |
|---|---|
| `shap_analysis.ipynb` | Full SHAP notebook with all plots |
| `domain_attribution.csv` | CS% vs DS% weight per attack class |
| `venn_diagram.png` | Detection Venn diagram (CS-only / DS-only / Both) |
| `detection_rate_table.csv` | Per-class detection rates for CS, DS, Fusion |
| `shap_summary_plot.png` | Global SHAP beeswarm plot |
| `evaluation_report.md` | Written final findings and recommendations |

---

## Handoff / Dependencies

Part R **cannot start** until:
- ✅ Part G has delivered `fusion_model.pkl`, `features_with_cs.csv`, `predictions.csv`
- ✅ Part S has delivered `features_with_ds.csv`, `eda_summary.md`

Part R's outputs feed the **final project report / presentation** — no downstream code dependency.

---

## Tech Stack

```
shap (TreeExplainer, summary_plot, bar_plot)
matplotlib, seaborn
matplotlib_venn (pip install matplotlib-venn)
pandas, numpy
scikit-learn (classification_report, roc_auc_score)
jupyter notebook
```

---

## Key Questions This Part Must Answer

1. For each attack type, does the ML model rely more on CS rule flags or the DS anomaly score?
2. Which attacks are caught exclusively by CS rules that the anomaly detector misses?
3. Which attacks are caught exclusively by the DS anomaly score (novel/zero-day) that CS rules miss?
4. What is the quantified accuracy improvement of the fusion model over either layer alone?
5. Where should a real-world IDS set the DS anomaly threshold to minimise false positives while catching zero-days?
