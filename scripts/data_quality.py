"""
Data Quality Report (Sprint 1)

Reads the processed dataset (Parquet) and generates a lightweight quality report:
- rows/cols
- duplicates
- null rate per column
- target distribution
Saves the report to artifacts/reports/data_quality.json
"""

import json
from pathlib import Path
import pandas as pd


def main() -> None:
    data_path = Path("data/processed/train.parquet")
    report_path = Path("artifacts/reports/data_quality.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    df = pd.read_parquet(data_path)

    if "target" not in df.columns:
        raise ValueError("Missing required column: target")

    report = {
        "dataset_path": str(data_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_duplicates": int(df.duplicated().sum()),
        "null_rate_by_col": (df.isna().mean().round(6)).to_dict(),
        "target_distribution": df["target"].value_counts(dropna=False).to_dict(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved quality report -> {report_path}")


if __name__ == "__main__":
    main()
