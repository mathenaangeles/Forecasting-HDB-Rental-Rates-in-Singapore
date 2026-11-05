"""The market level implied by each index forecast, and by each flat type.

Every forecast in Section V scores differently on the rolling folds, but the
folds cannot say which one is right for the real test months, which lie 1 to 16
months beyond the last labelled one. This script asks a narrower question that
the folds do answer poorly: how far apart are these forecasts in dollars over
exactly the months we must predict? It reproduces Table IX of the report.

    python src/index_horizon.py        # writes outputs/index_test_horizon.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import HORIZON, TRAIN_END  # noqa: E402
from tindex import extrapolate, fit_index  # noqa: E402

# Each row is one forecast of the index past the last labelled month. The
# shipped one is the damped trend over a 12-month window with phi = 0.6.
VARIANTS = [
    ("Random walk", dict(method="flat")),
    ("Damped, 12 months, phi = 0.4", dict(method="damped", window=12, damp=0.4)),
    ("Damped, 12 months, phi = 0.6 (shipped)", dict(method="damped", window=12, damp=0.6)),
    ("Damped, 12 months, phi = 0.75", dict(method="damped", window=12, damp=0.75)),
    ("Damped, 12 months, phi = 0.85", dict(method="damped", window=12, damp=0.85)),
    ("Damped, 12 months, phi = 0.95", dict(method="damped", window=12, damp=0.95)),
    ("Damped, 18 months, phi = 0.85", dict(method="damped", window=18, damp=0.85)),
    ("Linear, 12 months", dict(method="linear", window=12)),
    ("Linear, 18 months", dict(method="linear", window=18)),
]
SHIPPED = "Damped, 12 months, phi = 0.6 (shipped)"


def main() -> None:
    base = pd.read_parquet("outputs/base.parquet")
    tr = base[base._split == "train"]
    index = fit_index(tr)

    # The index is normalised so that the last labelled month equals 1. The
    # mean is taken over the 50,000 test rows rather than over the 16 months,
    # so that it is weighted by how many rows actually fall at each horizon.
    counts = base[base._split == "test"].TM.value_counts().sort_index()
    counts = counts[counts.index > TRAIN_END]
    rent = tr.MONTHLY_RENT.mean()

    rows = []
    for name, kw in VARIANTS:
        full = extrapolate(index, horizon=HORIZON, **kw)
        level = (full.loc[counts.index] * counts).sum() / counts.sum()
        # The level at the far end of the horizon, which is what Section V
        # quotes when it says the chosen forecast is almost flat.
        rows.append(dict(forecast=name, mean_index=level,
                         final_month_rise=full.iloc[-1] - 1))
    out = pd.DataFrame(rows)

    # Expressed as the shift in predicted rent against the forecast we shipped,
    # which is what the difference between these rows actually costs.
    ship = out.loc[out.forecast == SHIPPED, "mean_index"].iloc[0]
    out["shift_sgd"] = (out.mean_index - ship) * rent
    out["mean_index"] = out.mean_index.round(4)
    out["shift_sgd"] = out.shift_sgd.round(1)
    out["final_month_rise"] = (out.final_month_rise * 100).round(2)
    out = out[["forecast", "mean_index", "shift_sgd", "final_month_rise"]]

    # The same index fitted per flat type, which is the segmentation we ship
    # and the basis for the claim in Section VIII that the surge was uneven.
    seg = []
    for ft, g in tr.groupby("FLAT_TYPE"):
        if len(g) < 200:
            continue
        own = fit_index(g)
        seg.append(dict(flat_type=ft, rows=len(g),
                        rise_over_labelled_period=round(1 / own.iloc[0] - 1, 3)))

    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/index_test_horizon.csv", index=False)
    pd.DataFrame(seg).to_csv("outputs/index_by_flat_type.csv", index=False)
    print(out.to_string(index=False))
    print("\nrise in the market level over the labelled period, by flat type")
    print(pd.DataFrame(seg).to_string(index=False))
    print(f"\nmean training rent S${rent:,.2f}")
    print(f"spread of the whole family  "
          f"{(out.mean_index.max() - out.mean_index.min()) * 100:.2f}% of level, "
          f"S${out.shift_sgd.max() - out.shift_sgd.min():.1f} of rent")
    print("wrote outputs/index_test_horizon.csv and outputs/index_by_flat_type.csv")


if __name__ == "__main__":
    main()
