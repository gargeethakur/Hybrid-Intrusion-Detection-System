# Phase 1: Deep EDA on CICIDS / CICFlowMeter

**Owner**: Part S  
**Status**: 🔲 TODO  
**Prerequisite**: Raw data placed in `shared/data/raw/`  
**Outputs**: `feature_rankings.csv`, `eda_summary.md`, cleaned dataframe in `shared/data/processed/`

---

## Goal
Understand the data deeply before any model or rule is written.
Produce feature rankings and class imbalance analysis that feed directly into Part G.

## Tasks
- [ ] Load and merge Data.csv + Label.csv
- [ ] Standardise column names; handle nulls and infinities
- [ ] Distribution plots for all features
- [ ] Class imbalance analysis (counts + ratios per class)
- [ ] Correlation matrix + VIF multicollinearity check
- [ ] Mutual information + ANOVA feature rankings → `feature_rankings.csv`
- [ ] CICFlowMeter_out integration: align columns, document new features
- [ ] Write `eda_summary.md` with key findings for Part G and Part R

## Completion Criteria
- `feature_rankings.csv` exists in `shared/outputs/`
- `eda_summary.md` documents top 20 features, class imbalance ratios, recommended SMOTE strategy
