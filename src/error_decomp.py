"""Error budget for the shipped LightGBM: noise, market forecast, remainder.

The squared error of a fold is split into three additive parts.

    noise      Spread among rows that are identical in every feature (same
               block, flat type and month). Estimated as the pooled within-group variance. Interpreting this
               as conditional noise requires independent errors around a common
               group mean; it is not a fitted lower bound for the hidden test set.
    forecast   What is lost by not knowing the future market level: the MSE
               with the forecast index minus the MSE with an oracle index
               fitted on the validation months themselves.
    remainder  Oracle MSE minus the estimated noise variance. This algebraic
               remainder can be negative because the noise term is estimated.

The split is reported on the rows that have a feature-identical twin, because
only there is the noise term measured, and the forecast term is also given on
all rows. The oracle sees validation rents and changes the transformed training target.
Its improvement is an optimistic diagnostic, not a formal bound on gains from a
better index forecast. The three terms are not independent causal components.

    python src/error_decomp.py

Writes outputs/error_budget.csv, outputs/error_slices.csv and the per-row
predictions to outputs/error_preds.parquet.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import experiment as ex  # noqa: E402
from experiment import fit_predict, clip_rent, HORIZON, FOLDS  # noqa: E402
from configs import get  # noqa: E402

CUTS = FOLDS + [39]
TWIN = ["STREET", "BLOCK", "FLAT_TYPE", "TM"]
SHIPPED = get("Jc_ft05_blockinfo")


def _oracle(cfg, a, b, cut):
    """Fit and predict with the index estimated on train and validation together."""
    full, orig = pd.concat([a, b]), ex._index_lookup
    ex._index_lookup = lambda c, tr: orig(c, full)
    try:
        return clip_rent(fit_predict(cfg, a, b, ref_t=cut - 1))
    finally:
        ex._index_lookup = orig


def run():
    base = pd.read_parquet("outputs/base.parquet")
    tr = base[base._split == "train"]
    glob = replace(SHIPPED, index_by=None)
    parts = []
    for cut in CUTS:
        a = tr[tr.TM < cut]
        b = tr[(tr.TM >= cut) & (tr.TM < cut + HORIZON)]
        out = b[TWIN + ["TOWN"]].assign(
            cut=cut, y=b.MONTHLY_RENT,
            fc=clip_rent(fit_predict(SHIPPED, a, b, ref_t=cut - 1)),
            fc_global=clip_rent(fit_predict(glob, a, b, ref_t=cut - 1)),
            oracle=_oracle(SHIPPED, a, b, cut),
            oracle_global=_oracle(glob, a, b, cut))
        out["n_twin"] = out.groupby(TWIN).y.transform("size")
        parts.append(out)
        print(f"cut {cut} done", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet("outputs/error_preds.parquet")
    return d


def _noise_var(g):
    """Pooled within-group variance of rent among feature-identical rows."""
    # Folds overlap, so the cut is part of the key: a row must not count as
    # its own twin when several folds are pooled.
    g, keys = g[g.n_twin >= 2], TWIN + ["cut"]
    dev = g.y - g.groupby(keys).y.transform("mean")
    return float((dev ** 2).sum() / (len(g) - g.groupby(keys).ngroups))


def _mse(g, col):
    return float(np.mean((g[col] - g.y) ** 2))


def budget(d):
    rows = []
    for cut, g in d.groupby("cut"):
        t = g[g.n_twin >= 2]
        noise = _noise_var(g)
        month_bias = g.assign(e=g.fc - g.y).groupby("TM").e.transform("mean")
        rows.append(dict(
            cut=cut, n=len(g), twin_share=len(t) / len(g),
            rmse=np.sqrt(_mse(g, "fc")), rmse_oracle=np.sqrt(_mse(g, "oracle")),
            rmse_oracle_global=np.sqrt(_mse(g, "oracle_global")),
            bias=float(np.mean(g.fc - g.y)),
            # The share of the MSE that is a common miss within a month, which
            # needs no oracle. It is descriptive, not a bound on oracle gains.
            month_level_share=float(np.mean(month_bias ** 2)) / _mse(g, "fc"),
            forecast_share_all=1 - _mse(g, "oracle") / _mse(g, "fc"),
            twin_rmse=np.sqrt(_mse(t, "fc")), twin_noise_sd=np.sqrt(noise),
            noise_share=noise / _mse(t, "fc"),
            forecast_share=(_mse(t, "fc") - _mse(t, "oracle")) / _mse(t, "fc"),
            remainder_share=(_mse(t, "oracle") - noise) / _mse(t, "fc"),
        ))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("outputs/error_budget.csv", index=False)
    print(out.T.to_string())


def slices(d):
    d = d[d.cut.isin(FOLDS)].assign(
        year=2021 + d.TM // 12,
        level=pd.qcut(d.fc, 4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (dearest)"]))
    rows = []
    for by in ("FLAT_TYPE", "year", "level"):
        for k, g in d.groupby(by, observed=True):
            e = g.fc - g.y
            twins = g[g.n_twin >= 2]
            rows.append(dict(
                by=by, value=k, n=len(g), mean_rent=g.y.mean(),
                rmse=np.sqrt(np.mean(e ** 2)), bias=e.mean(),
                rel_rmse=np.sqrt(np.mean(e ** 2)) / g.y.mean(),
                noise_sd=np.sqrt(_noise_var(g)) if len(twins) > 50 else np.nan))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("outputs/error_slices.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    d = (pd.read_parquet("outputs/error_preds.parquet")
         if "--score" in sys.argv else run())
    budget(d)
    slices(d)
