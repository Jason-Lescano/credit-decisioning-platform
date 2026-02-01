"""
Train-all orchestrator (Sprint 1)

Runs the full Sprint 1 pipeline in order:
1) Build processed dataset (LendingClub loader)
2) Generate data quality report
3) Train LightGBM baseline + save artifacts

Usage:
  python scripts/train_all.py
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    try:
        run([sys.executable, "src/credit_decisioning/data/load_lendingclub.py"])
        run([sys.executable, "scripts/data_quality.py"])
        run([sys.executable, "scripts/train_lgbm.py"])
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Step failed with exit code {e.returncode}")
        return e.returncode

    print("\nDONE: Sprint 1 pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
