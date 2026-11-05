"""Compare two scored submissions to separate the cross-section from the level.

Two submissions can agree almost perfectly on which flat is dearer than which
and still score 75 RMSE apart, because they disagree on where the market level
sits over the test horizon. Our own pair does exactly that, which is the only
evidence we have on real test labels for the damped forecast of Section V.

    python src/compare_submissions.py                       # the default pair
    python src/compare_submissions.py a.csv b.csv           # any two files
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

DEFAULT = ("submissions/blend_lgbm_cat_blockhist_shrink07.csv",
           "submissions/ridge_raw_rent_optuna.csv")


def monthly_slope(level: pd.Series) -> float:
    """Percent per month, from a least-squares line through the log levels."""
    x = np.arange(len(level))
    return float(np.polyfit(x, np.log(level.values), 1)[0] * 100)


def main(a_path: str, b_path: str) -> None:
    base = pd.read_parquet("outputs/base.parquet")
    test = base[base._split == "test"].reset_index(drop=True)
    a = pd.read_csv(a_path).Predicted.to_numpy()
    b = pd.read_csv(b_path).Predicted.to_numpy()
    if not len(a) == len(b) == len(test):
        raise SystemExit("the two files and the test frame must have equal length")

    d = pd.DataFrame({"TM": test.TM.to_numpy(), "a": a, "b": b})

    # Within one test month the market level is common to both, so what is left
    # is how each ranks the flats against one another.
    within = d.groupby("TM").apply(
        lambda g: np.corrcoef(g.a, g.b)[0, 1], include_groups=False)
    level = d.groupby("TM")[["a", "b"]].mean()

    print(f"a = {a_path}\nb = {b_path}\n")
    print(f"within-month correlation  mean {within.mean():.4f}  "
          f"min {within.min():.4f}  max {within.max():.4f}")
    print(f"monthly level slope       a {monthly_slope(level.a):+.2f}%  "
          f"b {monthly_slope(level.b):+.2f}%")
    gap = level.b.iloc[-1] / level.a.iloc[-1] - 1
    print(f"gap in the last test month  {gap:+.1%}  "
          f"(S${level.b.iloc[-1] - level.a.iloc[-1]:+,.0f})")

    os.makedirs("outputs", exist_ok=True)
    out = level.round(1).assign(within_month_corr=within.round(4))
    out.to_csv("outputs/submission_comparison.csv")
    print("\nwrote outputs/submission_comparison.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args if len(args) == 2 else DEFAULT))
