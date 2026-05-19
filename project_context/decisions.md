# Key Design Decisions

| # | Decision | Rationale | Alternative Considered |
|---|---|---|---|
| 1 | Use Isolation Forest for anomaly detection | Unsupervised, no label leakage, well-suited for tabular network data | Autoencoder (higher complexity, similar performance on tabular) |
| 2 | XGBoost as primary fusion model | Handles mixed feature types (binary flags + continuous), natively supports multiclass, fast SHAP support via TreeExplainer | Neural network (harder to explain, slower to train) |
| 3 | SMOTE for class imbalance | Prevents majority class domination without discarding minority samples | Class weights only (weaker; SMOTE preferred for severe imbalance like Worms/Shellcode) |
| 4 | DS anomaly score normalised to [0,1] | Makes it directly comparable to binary CS flags as a model feature | Raw decision function output (negative values confuse some models) |
| 5 | Venn diagram computed before ML | Shows raw signal coverage independent of model decisions — cleaner research design | Post-prediction Venn (confounds model errors with signal gaps) |
| 6 | Separate requirements.txt per part | Each team member installs only what they need; avoids version conflicts | Monolithic requirements (harder to manage across team) |
