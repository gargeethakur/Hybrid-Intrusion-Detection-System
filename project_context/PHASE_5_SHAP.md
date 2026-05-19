# Phase 5: SHAP Analysis & Comparative Evaluation

**Owner**: Part R  
**Status**: 🔲 TODO  
**Prerequisite**: Phase 4 complete (fusion_model.pkl + predictions.csv)  
**Outputs**: `domain_attribution.csv`, `venn_diagram.png`, `evaluation_report.md`

---

## Goal
Use SHAP to quantify how much each domain (CS rules vs DS anomaly) the ML model relied on per attack type.
Produce the Venn diagram and the final comparative evaluation report.

## Tasks
- [ ] Load fusion_model.pkl + predictions.csv
- [ ] Compute SHAP values via TreeExplainer
- [ ] Global SHAP summary plot (beeswarm + bar chart)
- [ ] Per-class domain attribution: CS% vs DS% weight → `domain_attribution.csv`
- [ ] Bar chart: CS vs DS weight for every attack class
- [ ] Venn diagram: CS-only / DS-only / Both / Neither detection regions
- [ ] Detection rate table: CS, DS, Fusion per class → `detection_rate_table.csv`
- [ ] Write `evaluation_report.md` with final findings and deployment recommendations

## Completion Criteria
- `domain_attribution.csv` shows CS% and DS% for all 10 classes
- `venn_diagram.png` clearly shows CS-only, DS-only, and overlap regions
- `evaluation_report.md` answers all 5 key research questions
