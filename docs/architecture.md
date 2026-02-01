# Architecture (Sprint 1)

This project is a minimal **Decision Intelligence** baseline for credit decisioning.

## High-level flow

Raw data (LendingClub)
→ Loader/Normalization
→ Processed dataset (Parquet)
→ Data quality report
→ Baseline model training (LightGBM)
→ Model artifacts
→ Scoring API (FastAPI)

## Components

### 1) Data ingestion & normalization
- Script: `src/credit_decisioning/data/load_lendingclub.py`
- Input: `data/raw/lending-club/accepted_*.csv(.gz)`
- Output: `data/processed/train.parquet`
- Notes:
  - Builds `target` from `loan_status` and **drops `loan_status`** to prevent leakage.

### 2) Data quality report
- Script: `scripts/data_quality.py`
- Output: `artifacts/reports/data_quality.json`
- Checks:
  - rows/cols, duplicates, null rate, target distribution

### 3) Baseline training
- Script: `scripts/train_lgbm.py`
- Output:
  - `artifacts/models/lgbm_model.joblib` (model + feature_names)
  - `artifacts/models/lgbm_model_info.json`
  - `artifacts/reports/metrics.json`

### 4) Orchestrator
- Script: `scripts/train_all.py`
- Runs loader → quality → training in order.

### 5) Scoring API
- Module: `src/credit_decisioning/app/main.py`
- Endpoints:
  - `GET /health`
  - `POST /score`

## Diagram
data/raw/
└── lending-club/
    └── accepted_*.csv.gz
          |
          v
src/credit_decisioning/data/load_lendingclub.py
          |
          v
data/processed/train.parquet
      |                    |
      |                    +--> scripts/train_lgbm.py
      |                         -> artifacts/models/lgbm_model.joblib
      |                         -> artifacts/reports/metrics.json
      |
      +--> scripts/data_quality.py
           -> artifacts/reports/data_quality.json

FastAPI:
POST /score loads artifacts/models/lgbm_model.joblib

