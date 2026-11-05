"""How much the supplied MRT table costs us by not describing physical stations.

The file lists one row per line code, so an interchange appears two or three
times, and its STATUS column marks Changi Terminal 5 as open although the
station is not built. Nearest-station distance is unaffected, since duplicate
rows sit on identical coordinates, but the second-nearest distance and the
radius counts are not. This script measures both the feature change and the
score change, so the limitation is quantified rather than asserted.

    python src/mrt_check.py      # writes outputs/mrt_station_check.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import _dist_and_count  # noqa: E402
from experiment import Config, evaluate  # noqa: E402
from tuned_params import BEST_LGBM  # noqa: E402

OUT = "outputs/mrt_station_check.csv"
COLS = ["MRT_NEAR1", "MRT_NEAR2", "MRT_CNT0.5", "MRT_CNT1.0"]
# Announced for the 2030s, so it cannot have been open during 2021 to 2026.
NOT_YET_BUILT = "changi terminal 5 mrt station"


def corrected_mrt(df, data_dir="data"):
    """Rebuild the MRT columns from physical stations rather than line codes."""
    mrt = pd.read_csv(f"{data_dir}/auxiliary/sg-mrt-stations.csv")
    opened = mrt[(mrt.STATUS == "open") & (mrt.NAME != NOT_YET_BUILT)]
    one_per_station = opened.drop_duplicates(["LATITUDE", "LONGITUDE"])
    feats = _dist_and_count(df, one_per_station.LATITUDE.values,
                            one_per_station.LONGITUDE.values)
    out = df.copy()
    for suffix, vals in feats.items():
        if f"MRT_{suffix}" in COLS:
            out[f"MRT_{suffix}"] = vals
    return out, len(mrt), len(opened), len(one_per_station)


def main():
    base = pd.read_parquet("outputs/base.parquet")
    fixed, n_rows, n_open, n_stations = corrected_mrt(base)
    print(f"supplied table: {n_rows} rows, {n_open} marked open after dropping "
          f"{NOT_YET_BUILT}, {n_stations} physical stations")

    for c in COLS:
        changed = ~np.isclose(base[c].to_numpy(float), fixed[c].to_numpy(float))
        gap = np.abs(base[c].to_numpy(float) - fixed[c].to_numpy(float))
        print(f"  {c:<11s} rows changed {changed.mean():6.1%}   "
              f"mean change {gap.mean():6.3f}   max {gap.max():6.3f}")

    rows = []
    for name, frame in [("supplied", base), ("one row per station", fixed)]:
        t0 = time.time()
        cfg = Config(name=name, model="lgbm", params=dict(BEST_LGBM))
        mean, _ = evaluate(cfg, frame, verbose=False)
        rows.append(dict(variant=name, rmse=round(mean, 2)))
        print(f"{name:<20s} {mean:8.2f}  ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    d["delta"] = (d.rmse - d.rmse.iloc[0]).round(2)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
