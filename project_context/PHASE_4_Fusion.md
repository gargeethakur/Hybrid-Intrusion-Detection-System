# Phase 4: ML Fusion Model

**Owner**: Part G  
**Status**: 🔲 TODO  
**Prerequisite**: Phase 1 (feature_rankings.csv) + Phase 3 (features_with_ds.csv)  
**Outputs**: `fusion_model.pkl`, `predictions.csv`, `confusion_matrix.png`

---

## Goal
Train XGBoost + Random Forest on CS flags + DS anomaly score + original features.
Run ablation study to quantify each domain's contribution.

## Tasks
- [ ] Build feature matrix: original features + ds_anomaly_score + cs_* flags
- [ ] Stratified 70/15/15 train/val/test split
- [ ] Apply SMOTE or class_weight balancing (per Part S recommendation)
- [ ] Train XGBoost (primary)
- [ ] Train Random Forest (secondary comparison)
- [ ] Hyperparameter tuning (Optuna or GridSearchCV)
- [ ] Ablation study: features only vs CS only vs DS only vs Fusion
- [ ] Export classification report, confusion matrix, ROC-AUC per class
- [ ] Save `fusion_model.pkl` and `predictions.csv` to `shared/outputs/`

## Completion Criteria
- `fusion_model.pkl` saved and loadable by Part R's SHAP code
- `predictions.csv` contains: true_label, predicted_label, confidence, all cs_* flags, ds_anomaly_score
- Ablation study results documented in `outputs/model_evaluation.md`
