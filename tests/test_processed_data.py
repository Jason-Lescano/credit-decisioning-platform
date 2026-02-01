from pathlib import Path
import pandas as pd


def test_processed_dataset_exists_and_has_target():
    data_path = Path("data/processed/train.parquet")
    assert data_path.exists(), "Expected processed dataset at data/processed/train.parquet"

    df = pd.read_parquet(data_path, columns=["target"])
    assert "target" in df.columns

    unique_vals = set(df["target"].dropna().unique().tolist())
    assert unique_vals.issubset({0, 1}), f"Unexpected target values: {unique_vals}"
