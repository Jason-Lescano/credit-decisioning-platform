"""
Train LightGBM baseline (Sprint 1)

What this does (high level):
- Reads the processed dataset (data/processed/train.parquet)
- Builds a simple baseline preprocessing:
  - numeric features are kept as-is
  - categorical features are one-hot encoded (pd.get_dummies)
- Splits the data into train/validation (stratified)
- Trains a LightGBM classifier
- Evaluates with AUC and Brier score
- Saves artifacts for later use (API, monitoring, etc.)

Artifacts:
- artifacts/models/lgbm_model.joblib  (model + feature list)
- artifacts/models/lgbm_model_info.json
- artifacts/reports/metrics.json
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import re


def sanitize_feature_names(columns: list[str]) -> list[str]:
    """
    LightGBM does not allow some special JSON characters in feature names.
    We replace any non-alphanumeric or underscore characters with underscore.
    """
    cleaned = []
    for c in columns:
        c2 = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        cleaned.append(c2)
    return cleaned


def main() -> None:
    # ---- Paths (project conventions) ----
    data_path = Path("data/processed/train.parquet")
    model_path = Path("artifacts/models/lgbm_model.joblib")
    info_path = Path("artifacts/models/lgbm_model_info.json")
    metrics_path = Path("artifacts/reports/metrics.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    # ---- Load data ----
    df = pd.read_parquet(data_path)

    if "target" not in df.columns:
        raise ValueError("Missing required column: target")

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    # ---- Baseline preprocessing ----
    # One-hot encoding is a simple, transparent baseline.
    # Later (Sprint 2+) we can switch to LightGBM native categorical handling or better encodings.
    X = pd.get_dummies(X, dummy_na=True)
    X.columns = sanitize_feature_names(list(X.columns))
    X = X.loc[:, ~X.columns.duplicated()]  # in case cleaning creates duplicates

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # preserves target proportions
    )

    # ---- Model ----
    # Reasonable defaults for a baseline (not tuned).
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ---- Evaluate ----
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    brier = brier_score_loss(y_val, val_proba)

    # ---- Save artifacts ----
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Store model + feature names (important for consistent scoring later)
    joblib.dump({"model": model, "feature_names": list(X.columns)}, model_path)

    info = {
        "model_type": "lightgbm",
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
        "random_state": 42,
        "model_path": str(model_path),
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    metrics = {
        "val_auc": float(auc),
        "val_brier": float(brier),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved model info -> {info_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"VAL AUC={auc:.4f} | Brier={brier:.4f}")


if __name__ == "__main__":
    main()
