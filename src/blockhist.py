"""What block rent history adds to the notebook model, on the shared splits.

Fits the notebook's HistGB (raw rent, same search grid) with and without the
two history columns, at the three 16-month rolling cuts and the 12-month
holdout, and reports how much of the validation set has any history at all.

    python src/blockhist.py

Writes outputs/ablation_blockhist.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(__file__))
from experiment import rmse, clip_rent, HORIZON, FOLDS  # noqa: E402
from h2h import GRID, HIST, notebook_frame, build_X  # noqa: E402


def fit_predict(a, b, use_hist):
    a = a.sort_values("TM")
    Xa, Xb = build_X(a, a, False), build_X(b, a, True)
    if not use_hist:
        Xa, Xb = Xa.drop(columns=HIST), Xb.drop(columns=HIST)
    s = RandomizedSearchCV(HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42),
                           GRID, n_iter=8, cv=TimeSeriesSplit(n_splits=4),
                           scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
    s.fit(Xa, a["MONTHLY_RENT"])
    return clip_rent(s.best_estimator_.predict(Xb)), Xb


def main():
    nb = notebook_frame(pd.read_parquet("outputs/base.parquet"))
    tr = nb[nb._split == "train"]
    rows = []
    for cut in FOLDS + [39]:
        a, b = tr[tr.TM < cut], tr[(tr.TM >= cut) & (tr.TM < cut + HORIZON)]
        y = b["MONTHLY_RENT"].to_numpy()
        p1, Xb = fit_predict(a, b, True)
        p0, _ = fit_predict(a, b, False)
        has = Xb["HIST_BLK"].notna().to_numpy()
        rows.append(dict(
            cut=cut, n=len(b), with_hist=rmse(y, p1), without_hist=rmse(y, p0),
            bias_with=float(np.mean(p1 - y)), bias_without=float(np.mean(p0 - y)),
            covered=has.mean(), median_n=float(np.nanmedian(Xb["HIST_BLK_N"])),
            with_hist_covered=rmse(y[has], p1[has]), without_hist_covered=rmse(y[has], p0[has]),
            with_hist_new=rmse(y[~has], p1[~has]), without_hist_new=rmse(y[~has], p0[~has])))
        print({k: round(v, 2) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).round(3).to_csv("outputs/ablation_blockhist.csv", index=False)


if __name__ == "__main__":
    main()
