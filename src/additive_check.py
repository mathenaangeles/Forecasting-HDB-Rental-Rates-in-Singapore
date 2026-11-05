"""Matched additive versus multiplicative market-index comparison.

Run from the repository root:

    python src/additive_check.py

Both treatments use the same structural feature matrices, seven block-register
features, frozen BEST_LGBM settings, seed 0, and training-only market indices.
No target or spatial encoding is included. Both indices use twelve two-way
updates and a twelve-month slope window with geometric damping of 0.6. The
additive treatment fits rent minus a dollar-valued month effect; the existing
multiplicative treatment fits rent divided by its dimensionless month index.

The settings were previously selected for the multiplicative pipeline. This
finite sensitivity check does not independently tune the additive alternative.
The three rolling windows overlap; the fourth cut has twelve labeled months.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from configs import BLOCK_INFO  # noqa: E402
from experiment import (Config, FOLDS, HORIZON, _align, _feature_frame,
                        clip_rent, make_model, rmse)  # noqa: E402
from tindex import GRAIN, extrapolate, fit_index  # noqa: E402
from tuned_params import BEST_LGBM  # noqa: E402


def fit_additive_index(frame: pd.DataFrame, n_iter: int = 12) -> pd.Series:
    """Fit rent = group base + month effect by alternating mean residuals.

    The month effect is measured in dollars and normalized to zero in the last
    training month. A group is the same street/block/flat-type unit used by the
    multiplicative estimator. All input records must belong to the training
    side of the relevant fold.
    """
    y = frame["MONTHLY_RENT"].to_numpy(float)
    t = frame["TM"].to_numpy(int)
    g = frame.groupby(GRAIN, observed=True).ngroup().to_numpy()
    n_t, n_g = int(t.max()) + 1, int(g.max()) + 1
    nt = np.bincount(t, minlength=n_t)
    ng = np.bincount(g, minlength=n_g)
    if not np.all(nt > 0):
        raise ValueError("The comparison requires every training month to occur.")
    effect = np.zeros(n_t)
    for _ in range(n_iter):
        base = np.bincount(g, weights=y - effect[t], minlength=n_g) / ng
        effect = np.bincount(t, weights=y - base[g], minlength=n_t) / nt
        effect -= effect[-1]
    return pd.Series(effect, index=np.arange(n_t), name="MONTH_EFFECT")


def main(threads: int = 4) -> pd.DataFrame:
    base = pd.read_parquet("outputs/base.parquet")
    train = base[base["_split"] == "train"]
    cfg = Config(name="additive_check", model="lgbm", detrend=False,
                 spatial=False, te_mode="none", extra_num=list(BLOCK_INFO),
                 params={**BEST_LGBM, "random_state": 0, "n_jobs": threads})
    rows, cached = [], {}
    for cut in FOLDS + [39]:
        a = train[train.TM < cut].copy()
        b = train[(train.TM >= cut) & (train.TM < cut + HORIZON)].copy()
        # Construct the feature matrices once so the two targets use identical
        # inputs. Encodings based on the target are deliberately disabled.
        xa = _feature_frame(cfg, a, a, "expanding", ref_t=cut - 1)
        xb = _feature_frame(cfg, a, b, "future", ref_t=cut - 1)
        xa, xb = _align(xa, xb)
        ya = a.MONTHLY_RENT.to_numpy(float)
        yb = b.MONTHLY_RENT.to_numpy(float)
        cached[cut] = {"y": yb, "TM": b.TM.to_numpy()}
        for treatment in ("multiplicative", "additive"):
            start = time.monotonic()
            index = fit_index(a) if treatment == "multiplicative" else fit_additive_index(a)
            forecast = extrapolate(index, HORIZON + 2, window=12, damp=0.6,
                                   method="damped")
            ia = index.reindex(a.TM).to_numpy()
            ib = forecast.reindex(b.TM).to_numpy()
            if treatment == "multiplicative":
                if not np.all(ia > 0) or not np.all(ib > 0):
                    raise ValueError("The multiplicative index must be positive.")
                target = ya / ia
            else:
                target = ya - ia
            model = make_model(cfg)
            model.fit(xa, target)
            adjusted = model.predict(xb)
            prediction = adjusted * ib if treatment == "multiplicative" else adjusted + ib
            clipped = clip_rent(prediction)
            cached[cut][treatment] = prediction
            row = dict(treatment=treatment, cut=cut,
                       protocol="rolling16" if cut in FOLDS else "confirmation12",
                       n=len(yb), rmse=rmse(yb, clipped),
                       bias=float(np.mean(clipped - yb)),
                       prediction_min=float(prediction.min()),
                       prediction_max=float(prediction.max()),
                       n_clipped=int(np.count_nonzero(prediction != clipped)))
            rows.append(row)
            print({**row, "seconds": round(time.monotonic() - start, 1)}, flush=True)
    out = pd.DataFrame(rows)
    for treatment in ("multiplicative", "additive"):
        r = out[(out.treatment == treatment) & (out.protocol == "rolling16")]
        rows.append(dict(treatment=treatment, cut="mean", protocol="rolling16_mean",
                         n=int(r.n.sum()), rmse=float(r.rmse.mean()),
                         bias=float(r.bias.mean()),
                         prediction_min=float(r.prediction_min.min()),
                         prediction_max=float(r.prediction_max.max()),
                         n_clipped=int(r.n_clipped.sum())))
    out = pd.DataFrame(rows)
    out.to_csv("outputs/additive_index_check.csv", index=False)
    pd.to_pickle(cached, "outputs/additive_index_check.pkl")
    print(out.to_string(index=False), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    main(threads=args.threads)
