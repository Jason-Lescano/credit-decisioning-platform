"""
LendingClub loader (Sprint 1)

Goal:
- Read LendingClub "accepted" loans file (CSV or CSV.GZ)
- Normalize a small set of core features
- Build a binary target from loan_status (good vs bad outcomes)
- Save a clean dataset to data/processed/train.parquet

Why:
- This creates a stable internal "data contract" so the rest of the pipeline
  (features -> models -> decisioning -> monitoring) does not depend on raw file quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class LendingClubPaths:
    """File locations for raw and processed data."""

    raw_dir: Path
    processed_path: Path


def find_accepted_file(raw_dir: Path) -> Path:
    """
    Pick the accepted loans file.

    We prefer .csv.gz to save disk and because the dataset often comes compressed.
    """
    gz = sorted(raw_dir.glob("accepted_*.csv.gz"))
    if gz:
        return gz[0]
    csv = sorted(raw_dir.glob("accepted_*.csv"))
    if csv:
        return csv[0]
    raise FileNotFoundError(f"No accepted_*.csv(.gz) found in {raw_dir}")


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Build binary target from loan_status.

    Convention:
    - 1 = "bad" outcomes (default-like)
    - 0 = "good" outcomes (fully paid)

    We drop ambiguous/in-progress statuses for Sprint 1.
    """
    status = df["loan_status"].astype(str)

    bad_statuses = {
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "Does not meet the credit policy. Status:Charged Off",
    }
    good_statuses = {
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    }

    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target.loc[status.isin(bad_statuses)] = 1
    target.loc[status.isin(good_statuses)] = 0
    return target


def load_and_normalize(paths: LendingClubPaths) -> pd.DataFrame:
    """
    Load raw LendingClub accepted loans and normalize to our internal schema.
    """
    accepted_path = find_accepted_file(paths.raw_dir)

    # Minimal "core" columns for Sprint 1.
    # We can expand later without breaking downstream code.
    USECOLS = [
        "issue_d",
        "loan_status",
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "purpose",
        "addr_state",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "application_type",
    ]

    DTYPES = {
        # numeric (use smaller dtypes)
        "loan_amnt": "float32",
        "int_rate": "float32",
        "installment": "float32",
        "annual_inc": "float32",
        "dti": "float32",
        "delinq_2yrs": "float32",
        "inq_last_6mths": "float32",
        "open_acc": "float32",
        "pub_rec": "float32",
        "revol_bal": "float32",
        "revol_util": "float32",
        "total_acc": "float32",
        # categorical (big memory saver)
        "term": "string",
        "grade": "category",
        "sub_grade": "category",
        "emp_length": "category",
        "home_ownership": "category",
        "verification_status": "category",
        "purpose": "category",
        "addr_state": "category",
        "application_type": "category",
        "loan_status": "category",
        # keep as string, parse later
        "issue_d": "string",
    }

    # Load only columns in usecols to reduce memory footprint.
    chunks = []
    for chunk in pd.read_csv(
        accepted_path,
        compression="infer",
        usecols=USECOLS,
        dtype=DTYPES,
        low_memory=False,
        chunksize=200_000,
    ):
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        chunk = chunk.dropna(subset=["issue_d"])
        chunk["issue_month"] = chunk["issue_d"].dt.to_period("M").astype(str)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Parse issue date safely (e.g. "Dec-2015")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])

    # Helpful for time split / grouping
    df["issue_month"] = df["issue_d"].dt.to_period("M").astype(str)

    # Create target and drop rows without label (in-progress / ambiguous)
    df["target"] = build_target(df)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Normalize percent fields like "13.56%" -> 13.56
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False).replace("nan", pd.NA)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize term like " 36 months" -> 36
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0]
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # Ensure numeric columns are numeric (best-effort)
    numeric_cols = [
        "loan_amnt",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "total_acc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prevent leakage: loan_status is used to build the label, so it must NOT be a feature
    if "loan_status" in df.columns:
        df = df.drop(columns=["loan_status"])

    # Keep issue_d for temporal validation (OOT)
    if "issue_d" not in df.columns:
        raise KeyError("issue_d not found in raw dataset. Needed for OOT validation.")

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    parse_rate = df["issue_d"].notna().mean()
    if parse_rate < 0.95:
        raise ValueError(f"issue_d parse_rate too low: {parse_rate:.3f}. Check raw format.")

    # Optional: month bucket for easy plots (YYYY-MM)
    df["issue_month"] = df["issue_d"].dt.to_period("M").astype(str)

    return df


def save_processed(df: pd.DataFrame, processed_path: Path) -> None:
    """Save normalized dataset to Parquet for faster downstream reads."""
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)


def main() -> None:
    # Standard project paths
    raw_dir = Path("data/raw/lending-club")
    processed_path = Path("data/processed/train.parquet")

    df = load_and_normalize(LendingClubPaths(raw_dir=raw_dir, processed_path=processed_path))
    save_processed(df, processed_path)

    print(f"Saved processed dataset -> {processed_path} | rows={len(df):,} cols={df.shape[1]}")


if __name__ == "__main__":
    main()
