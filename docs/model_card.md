# Model Card — LightGBM Baseline (Sprint 1)

## Model overview
- Type: LightGBM binary classifier
- Objective: predict probability of default-like outcome (`target=1`)
- Use case: baseline credit decisioning (approve/review/reject)

## Data
- Source: LendingClub accepted loans (Kaggle dataset)
- Processed dataset: `data/processed/train.parquet`
- Label construction:
  - `target=1` for default-like outcomes (e.g., Charged Off, Default, Late)
  - `target=0` for Fully Paid
  - `loan_status` dropped after label creation to prevent leakage

## Features (Sprint 1)
- Core credit/application variables (subset)
- Categorical: one-hot encoding (baseline)

## Training setup
- Split: stratified train/validation (80/20)
- Random seed: 42
- Hyperparameters: baseline (not tuned)

## Metrics (validation)
- AUC: ~0.726
- Brier: ~0.150

## Intended use
- Demonstration of an end-to-end decisioning pipeline.
- Foundation for Sprint 2 (tuning + calibration) and Sprint 3 (champion/challenger).

## Limitations
- Baseline preprocessing (one-hot) may be suboptimal for high-cardinality features.
- No calibration yet (planned in next sprint).
- No segment-aware policy yet (planned in next sprint).

## Ethical / governance notes
- Avoid leakage: label column not used as feature.
- Future work: fairness checks and feature exclusion policies if needed.
