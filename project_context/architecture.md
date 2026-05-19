# System Architecture

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAW DATA                                │
│   Data.csv + Label.csv + CICFlowMeter_out                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼                               ▼
┌────────────────┐             ┌─────────────────────┐
│  PART S        │             │  PART G              │
│  Step 1: EDA   │             │  Step 2: CS Rules    │
│                │             │                      │
│  Step 3:       │             │  Snort-style rules:  │
│  Isolation     │             │  cs_ddos_flag        │
│  Forest        │             │  cs_portscan_flag    │
│                │             │  cs_flood_flag       │
│  ds_anomaly_   │             │  cs_exploit_flag     │
│  score ────────┼────────┐   │  cs_backdoor_flag    │
└────────────────┘        │   │  cs_shellcode_flag   │
                          │   │  cs_worm_flag ───────┼──┐
                          │   └──────────────────────┘  │
                          │                              │
                          └──────────────┬───────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  PART G              │
                              │  Step 4: Fusion ML   │
                              │                      │
                              │  XGBoost / RF        │
                              │  features +          │
                              │  cs_* flags +        │
                              │  ds_anomaly_score    │
                              │                      │
                              │  → fusion_model.pkl  │
                              │  → predictions.csv   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  PART R              │
                              │  Step 5: SHAP        │
                              │                      │
                              │  Domain attribution  │
                              │  CS% vs DS% per      │
                              │  attack class        │
                              │                      │
                              │  Venn diagram        │
                              │  Evaluation report   │
                              └─────────────────────┘
```

## Key Design Decisions

1. **DS anomaly layer is fully unsupervised** — Isolation Forest never sees labels.
2. **CS rules are hand-coded** — no learning from data, pure domain knowledge.
3. **Fusion model is the only supervised learner** — it decides which signal to trust.
4. **SHAP runs post-hoc** — attribution is not baked into the model architecture.
5. **Venn diagram is computed before ML runs** — purely from flags, not from predictions.
