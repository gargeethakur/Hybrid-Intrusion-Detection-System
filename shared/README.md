# Shared — Data & Intermediate Outputs

This folder is the **handoff zone** between all three parts. No code lives here — only data and shared outputs.

## Structure

```
shared/
├── data/
│   ├── raw/          ← Place original dataset files here
│   │   ├── Data.csv
│   │   ├── Label.csv
│   │   ├── CICFlowMeter_out/
│   │   └── Readme.txt
│   └── processed/    ← Cleaned/merged data produced by Part S Step 1
└── outputs/          ← Intermediate files shared between parts
    ├── feature_rankings.csv     (Part S → Part G)
    ├── features_with_ds.csv     (Part S → Part G, Part R)
    ├── eda_summary.md           (Part S → Part R)
    ├── features_with_cs.csv     (Part G → Part R)
    ├── fusion_model.pkl         (Part G → Part R)
    └── predictions.csv          (Part G → Part R)
```

## Rules

- **Do not commit large CSV/pkl files** — they are in `.gitignore`
- Share files via Google Drive / OneDrive / a shared network path and document the link below
- Each part should read from `shared/` and write its outputs back to `shared/outputs/`

## Shared Data Location

> 📁 **Google Drive link:** `[INSERT SHARED DRIVE LINK HERE]`
