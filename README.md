# Credit Decisioning Platform (Decision Intelligence)

End-to-end decisioning system (data → model → decision engine → API → monitoring → AI copilot).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run (Sprint 1)

### 1) Download dataset (optional if you already have it)
```bash
python scripts/download_data.py
```

### 2) Run full pipeline (loader → quality → training)
```bash
python scripts/train_all.py
```

### 3) Start API
```bash
uvicorn credit_decisioning.app.main:app --reload
```

Health check:
```bash
curl -s http://127.0.0.1:8000/health
```

Example scoring request (recommended approach: JSON file)
```bash
cat > payload.json << 'EOF'
{
  "features": {
    "loan_amnt": 10000,
    "term": 36,
    "int_rate": 13.5,
    "installment": 340.0,
    "grade": "B",
    "sub_grade": "B3",
    "emp_length": "10+ years",
    "home_ownership": "RENT",
    "annual_inc": 55000,
    "verification_status": "Verified",
    "purpose": "credit_card",
    "addr_state": "CA",
    "dti": 18.2,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 55.0,
    "total_acc": 22,
    "application_type": "Individual"
  }
}
EOF

curl -s -X POST "http://127.0.0.1:8000/score" \
  -H "Content-Type: application/json" \
  --data-binary @payload.json
```

## Tests

```bash
pip install pytest
pytest -q
```

## Outputs (not tracked in git)

- Processed dataset:
  - data/processed/train.parquet
- Reports:
  - artifacts/reports/data_quality.json
  - artifacts/reports/metrics.json
- Model artifact:
  - artifacts/models/lgbm_model.joblib

## Notes

- loan_status is used to build the label and is dropped afterward to prevent target leakage.
- data/ and artifacts/ are intentionally ignored by git.

