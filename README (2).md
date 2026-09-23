# Breast Ultrasound Classification with CNNs

Leakage-aware deep learning pipeline for benign vs malignant classification of breast ultrasound images (BUSI dataset), with explainability, external validation, and cloud-portable training.

> ⚠️ Research project only — not a diagnostic tool.

## Status
- [x] Step 0 — Project setup
- [ ] Step 1 — Data audit & leak-free split
- [ ] Step 2 — Dataset & preprocessing
- [ ] Step 3 — Baseline CNN
- [ ] Step 4 — 5-fold CV & architecture comparison
- [ ] Step 5 — Ablations
- [ ] Step 6 — Held-out test evaluation
- [ ] Step 7 — Explainability (Grad-CAM vs lesion masks)
- [ ] Step 8 — External validation
- [ ] Step 9 — Cloud training
- [ ] Step 10 — Demo deployment
- [ ] Step 11 — Report & slides

## Dataset
BUSI — Breast Ultrasound Images Dataset (Al-Dhabyani et al., *Data in Brief*, 2020).

## Repository structure
```
configs/     experiment configs (YAML)
src/         reusable code (data, models, training, evaluation)
scripts/     entry-point scripts
notebooks/   Kaggle notebooks
results/     figures and tables
```

## Results
_Coming soon._

## Reproducibility
All experiments use fixed seeds and group-aware splits. Code runs unchanged on Kaggle, local machines, and AWS SageMaker.
