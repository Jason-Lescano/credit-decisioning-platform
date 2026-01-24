# Credit Decisioning Platform (Decision Intelligence)

End-to-end decisioning system (data -> model -> decision engine -> API -> monitoring -> AI copilot).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run (Sprint 1)
- Download data: `python scripts/download_data.py`
- Train: `python -m scripts.train_all`
- API: `uvicorn credit_decisioning.app.main:app --reload`
