"""Out-of-selection confirmation of an LGBM ablation winner.

A config chosen because it won on tuned LGBM is re-scored with CatBoost and
as the shipped 50/50 blend, against the reference trained the same way, on
the three rolling origins plus the 12-month holdout (cut 39). Only a gain
that survives in the blend is adopted. The variants to confirm are named on the
command line.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import fit_predict, rmse, clip_rent, HORIZON, FOLDS  # noqa: E402
from tuned_params import BEST_CAT  # noqa: E402
import configs  # noqa: E402

CUTS = FOLDS + [39]


def score(cfg, tr):
    out = {}
    for cut in CUTS:
        a, b = tr[tr.TM < cut], tr[(tr.TM >= cut) & (tr.TM < cut + HORIZON)]
        lg = fit_predict(cfg, a, b, ref_t=cut - 1)
        ca = fit_predict(replace(cfg, model="cat", params=dict(BEST_CAT)), a, b, ref_t=cut - 1)
        out[cut] = dict(y=b["MONTHLY_RENT"].to_numpy(), lgbm=lg, cat=ca)
    return out


def summarise(name, p):
    row = dict(name=name)
    for k in ("lgbm", "cat", "blend"):
        r = {c: rmse(v["y"], clip_rent(v[k] if k != "blend" else 0.5 * (v["lgbm"] + v["cat"])))
             for c, v in p.items()}
        row[f"{k}_roll16"] = round(np.mean([r[c] for c in FOLDS]), 2)
        row[f"{k}_hold12"] = round(r[39], 2)
    return row


def main(names):
    tr = pd.read_parquet("outputs/base.parquet").query("_split == 'train'")
    rows = []
    # J0_ref is the tuned-LightGBM reference that every variant is measured
    # against, so it is always scored alongside them.
    for n in ["J0_ref"] + names:
        path = f"outputs/confirm_{n}.pkl"
        p = pd.read_pickle(path) if os.path.exists(path) else score(configs.get(n), tr)
        pd.to_pickle(p, path)
        rows.append(summarise(n, p))
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    prev = pd.read_csv("outputs/ablation_confirmation_cut.csv") if os.path.exists("outputs/ablation_confirmation_cut.csv") else None
    if prev is not None:
        res = pd.concat([prev[~prev.name.isin(res.name)], res], ignore_index=True)
    res.to_csv("outputs/ablation_confirmation_cut.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
