"""Alternative representations of location and of the numeric feature block.

Two techniques from the course are not part of our pipeline, and this script
records why. K-means clustering offers a data-driven partition of the island in
place of the administrative TOWN, PLANNING AREA, REGION and SUBZONE columns.
Principal component analysis offers a compressed numeric block in place of the
25 raw attributes.

Every variant is scored under the protocol of src/experiment.py, so the numbers
are directly comparable with the rest of the ablation study.

    python src/altreps.py            # run everything, write outputs/ablation_altreps.csv
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, evaluate  # noqa: E402
from tuned_params import BEST_LGBM  # noqa: E402

OUT = "outputs/ablation_altreps.csv"

VARIANTS = [
    ("default", "Administrative location columns, all 25 numeric attributes", {}),
    ("geo_k8", "K-means location clusters (k=8) replacing the admin columns", dict(geo_k=8)),
    ("geo_k25", "K-means location clusters (k=25) replacing the admin columns", dict(geo_k=25)),
    ("geo_k60", "K-means location clusters (k=60) replacing the admin columns", dict(geo_k=60)),
    ("pca_8", "First 8 principal components replacing the numeric block", dict(pca_k=8)),
    ("pca_16", "First 16 principal components replacing the numeric block", dict(pca_k=16)),
    ("pca_24", "First 24 principal components replacing the numeric block", dict(pca_k=24)),
]


def main():
    base = pd.read_parquet("outputs/base.parquet")
    rows = []
    for name, note, kw in VARIANTS:
        t0 = time.time()
        cfg = Config(name=name, model="lgbm", params=dict(BEST_LGBM), **kw)
        mean, _ = evaluate(cfg, base, verbose=False)
        rows.append(dict(variant=name, rmse=round(mean, 2), note=note))
        print(f"{name:<10s} {mean:8.2f}   ({time.time() - t0:.0f}s)  {note}", flush=True)
    d = pd.DataFrame(rows)
    d["delta"] = (d.rmse - d.rmse.iloc[0]).round(2)
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
