# Phase 3: Statistical Anomaly Detection Layer

**Owner**: Part S  
**Status**: 🔲 TODO  
**Prerequisite**: Phase 1 complete (cleaned dataframe available)  
**Outputs**: `features_with_ds.csv`, `isolation_forest_model.pkl`

---

## Goal
Train an Isolation Forest on the full feature set (unsupervised).
Generate `ds_anomaly_score` for every row. This score is a direct input to the fusion model.

## Tasks
- [ ] Train IsolationForest with contamination = class imbalance ratio from Phase 1
- [ ] Generate anomaly scores, normalise to [0, 1]
- [ ] Validate: plot score distribution per true label class
- [ ] Compute AUROC of ds_anomaly_score vs binary (Benign / Attack) label
- [ ] Export `features_with_ds.csv` to `shared/outputs/`
- [ ] (Optional) Train Autoencoder on Benign-only traffic; compare reconstruction error vs IF scores

## Completion Criteria
- `features_with_ds.csv` exists in `shared/outputs/` with `ds_anomaly_score` column
- AUROC documented in `outputs/anomaly_validation.md`
