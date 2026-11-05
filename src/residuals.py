"""Residual analysis for the shipped blend, and the one correction it suggests.

The error budget in `src/error_decomp.py` shows that almost all of the squared
error sits in the noise term, so there is very little left for a better
cross-section to recover. This script looks at what remains, which is the way
the bias moves with the forecast horizon, and tests the only correction that
the diagnosis actually motivates.

The bias of the shipped configuration is close to zero when it is averaged over
a whole validation window, but that average hides a pattern: the first months
after the cut are over-predicted and the last months are under-predicted. A
uniform change to the index cannot fix both ends at once, which is why the
window and damping sweep in `outputs/ablation_index_window_damping.csv` could not find it.
What it needs is a correction that is shaped by months ahead, tilting the index
path without moving its mean.

The tilt is fitted on the three rolling folds and applied to the month-39
holdout. Those windows overlap, because the labelled data stops at month 50, so
the holdout figure is optimistic rather than a clean out-of-sample test.

    python src/residuals.py            # fit, score and write the tables
    python src/residuals.py --score    # re-score from the saved predictions

Writes outputs/residual_by_horizon.csv, outputs/residual_slices.csv and
outputs/residual_tilt.csv, and the per-row predictions to
outputs/residual_preds.parquet.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import fit_predict, clip_rent, rmse, HORIZON, FOLDS  # noqa: E402
from configs import get  # noqa: E402

CUTS = FOLDS + [39]
SHIPPED = get("Jc_ft05_blockinfo")
PREDS = "outputs/residual_preds.parquet"
KEEP = ["TM", "TOWN", "FLAT_TYPE", "STREET", "BLOCK"]
# The buckets match the ones used for the ladder, so the two tables can be read
# side by side.
EDGES = [0, 4, 8, 12, 16]
LABELS = ["1-4", "5-8", "9-12", "13-16"]


def run() -> pd.DataFrame:
    base = pd.read_parquet("outputs/base.parquet")
    tr = base[base._split == "train"]
    parts = []
    for cut in CUTS:
        a = tr[tr.TM < cut]
        b = tr[(tr.TM >= cut) & (tr.TM < cut + HORIZON)]
        p = clip_rent(fit_predict(SHIPPED, a, b, ref_t=cut - 1))
        parts.append(b[KEEP].assign(cut=cut, h=b.TM - (cut - 1),
                                    y=b.MONTHLY_RENT, pred=p))
        print(f"cut {cut}: rmse {rmse(b.MONTHLY_RENT.to_numpy(), p):.2f}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(PREDS)
    return d


def _bucket(h: pd.Series) -> pd.Series:
    return pd.cut(h, EDGES, labels=LABELS)


def by_horizon(d: pd.DataFrame) -> None:
    """How the error and the bias change with months ahead, per fold and pooled."""
    d = d.assign(bucket=_bucket(d.h), e=d.pred - d.y)
    rows = []
    for (cut, bucket), g in d.groupby(["cut", "bucket"], observed=True):
        rows.append(dict(cut=cut, horizon=bucket, n=len(g),
                         rmse=np.sqrt(np.mean(g.e ** 2)), bias=g.e.mean(),
                         # The mean ratio is what the tilt below corrects, so it
                         # is reported next to the bias in dollars.
                         ratio=float((g.y / g.pred).mean())))
    for bucket, g in d[d.cut.isin(FOLDS)].groupby("bucket", observed=True):
        rows.append(dict(cut="rolling", horizon=bucket, n=len(g),
                         rmse=np.sqrt(np.mean(g.e ** 2)), bias=g.e.mean(),
                         ratio=float((g.y / g.pred).mean())))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("outputs/residual_by_horizon.csv", index=False)
    print("\n=== Error and bias by months ahead ===")
    print(out.pivot(index="cut", columns="horizon", values="bias").round(1).to_string())


def slices(d: pd.DataFrame) -> None:
    """Where the error is concentrated, and whether any slice is above its own floor.

    A slice is only worth attacking if its error is clearly larger than the
    spread among rows that are identical in every feature, because that spread
    is what no model can remove.
    """
    d = d[d.cut.isin(FOLDS)].copy()
    d["e"] = d.pred - d.y
    twin_keys = ["STREET", "BLOCK", "FLAT_TYPE", "TM", "cut"]
    d["n_twin"] = d.groupby(twin_keys).y.transform("size")
    rows = []
    for by in ("FLAT_TYPE", "TOWN"):
        for k, g in d.groupby(by, observed=True):
            t = g[g.n_twin >= 2]
            if len(t) > 50:
                dev = t.y - t.groupby(twin_keys).y.transform("mean")
                floor = np.sqrt((dev ** 2).sum() / (len(t) - t.groupby(twin_keys).ngroups))
            else:
                floor = np.nan
            r = np.sqrt(np.mean(g.e ** 2))
            rows.append(dict(by=by, value=k, n=len(g), mean_rent=g.y.mean(),
                             rmse=r, bias=g.e.mean(), noise_sd=floor,
                             # Above one means the slice has error the features
                             # could in principle still explain.
                             rmse_over_floor=r / floor if floor == floor else np.nan))
    out = pd.DataFrame(rows).round(3).sort_values(["by", "rmse"], ascending=[True, False])
    out.to_csv("outputs/residual_slices.csv", index=False)
    print("\n=== Slices against their own noise floor ===")
    print(out.to_string(index=False))


def tilt(d: pd.DataFrame) -> None:
    """Fit a per-horizon multiplier on the rolling folds and apply it to the holdout.

    The multiplier for a bucket is the mean ratio of actual to predicted rent on
    the rolling folds, rescaled so that the multipliers average to one. That
    keeps the overall level the model already gets right and only changes the
    slope of the index path across the horizon.
    """
    d = d.assign(bucket=_bucket(d.h))
    roll, hold = d[d.cut.isin(FOLDS)], d[d.cut == 39]
    f = roll.groupby("bucket", observed=True).apply(
        lambda g: float((g.y / g.pred).mean()), include_groups=False)
    f = f / f.mean()
    rows = []
    for name, g in [("rolling (fitted here)", roll), ("holdout 39", hold)]:
        m = g.bucket.map(f).to_numpy(float)
        y = g.y.to_numpy()
        rows.append(dict(split=name, n=len(g), rmse=rmse(y, g.pred.to_numpy()),
                         rmse_tilted=rmse(y, g.pred.to_numpy() * m),
                         bias=float((g.pred - g.y).mean()),
                         bias_tilted=float((g.pred.to_numpy() * m - y).mean())))
    out = pd.DataFrame(rows).round(3)
    out["gain"] = (out.rmse - out.rmse_tilted).round(3)
    out.to_csv("outputs/residual_tilt.csv", index=False)
    print("\n=== Horizon tilt ===")
    print("multipliers: " + ", ".join(f"{k} {v:.4f}" for k, v in f.items()))
    print(out.to_string(index=False))


# Every column other than these is constant within a block, so a group keyed on
# them really is a set of rows that no function of the features can tell apart.
IDENTITY = ["TOWN", "STREET", "BLOCK", "FLAT_TYPE", "FLAT_MODEL",
            "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "TM", "FURNISHED", "FEE"]


def _identity_frame(path: str) -> pd.DataFrame:
    """Read a competition file and build the identity keys `data.py` would build."""
    d = pd.read_csv(path)
    for c in ("TOWN", "BLOCK", "STREET", "FLAT_MODEL", "FLAT_TYPE"):
        d[c] = d[c].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    d["FLAT_TYPE"] = d["FLAT_TYPE"].str.replace(" ", "-", regex=False)
    t = pd.to_datetime(d.RENT_APPROVAL_DATE, format="%Y-%m")
    d["TM"] = (t.dt.year - 2021) * 12 + t.dt.month - 1
    return d


def floor() -> None:
    """The lowest RMSE any function of the given features could reach.

    Rows that agree on every feature cannot be told apart, so the best a model
    can do on them is predict their mean, and the spread around that mean is
    paid in full. Counting that spread on the rows where it can be measured, and
    charging nothing at all for every other row, gives a bound that no model
    reading only these columns can beat. It is a very generous bound, because it
    assumes that every row outside such a group is predicted exactly.

    The keys are normalised exactly as `data.py` normalises them. Without that
    step the two spellings of a street name split a block in two, which hides a
    third of the groups and makes the bound look weaker than it is.
    """
    d = _identity_frame("data/train.csv")
    rows = []
    years = [(y, d[(d.TM >= (y - 2021) * 12) & (d.TM < (y - 2020) * 12)])
             for y in (2021, 2022, 2023, 2024)]
    for label, sub in ([("all training data", d)] + [(str(y), s) for y, s in years]
                       + [("2025, January to March", d[d.TM >= 48]),
                          ("2024 onward", d[d.TM >= 36])]):
        g = sub.groupby(IDENTITY)["MONTHLY_RENT"]
        n, dev = g.transform("size"), sub.MONTHLY_RENT - g.transform("mean")
        m = n > 1
        within = (dev[m] ** 2).sum() / (m.sum() - sub[m].groupby(IDENTITY).ngroups)
        rows.append(dict(period=label, n=len(sub), twin_share=m.mean(),
                         within_rmse=np.sqrt(within),
                         rmse_lower_bound=np.sqrt(m.mean() * within)))
    # The bound is only useful if it also holds on the test window, so the twin
    # share is recomputed there. It needs no labels, and it comes out higher
    # than in training, which makes the test-side bound slightly stronger.
    te = _identity_frame("data/test.csv")
    share_te = (te.groupby(IDENTITY, dropna=False)["TOWN"].transform("size") > 1).mean()
    by_period = {r["period"]: r["within_rmse"] for r in rows}
    for w, lab in [(by_period["all training data"], "test window, noise from all training data"),
                   (by_period["2025, January to March"], "test window, noise from 2025 only")]:
        rows.append(dict(period=lab, n=len(te), twin_share=share_te, within_rmse=w,
                         rmse_lower_bound=np.sqrt(share_te) * w))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("outputs/rmse_floor.csv", index=False)
    print("\n=== Lower bound on RMSE for any model using only these features ===")
    print(out.to_string(index=False))


def feature_floor() -> None:
    """Whether any engineered feature separates rows the raw columns cannot.

    The floor above is a bound for the eleven competition columns. It would stop
    being a bound if some feature we build varied inside one of those groups,
    because the model could then tell those rows apart after all. Every source
    in data/auxiliary joins on block or on month, and a twin group is one block
    in one month, so in principle none of them can. This checks it rather than
    assuming it, by regrouping on all engineered columns at once and seeing
    whether the groups get any finer.
    """
    df = pd.read_parquet("outputs/base.parquet")
    tr = df[df._split == "train"]
    feats = [c for c in tr.columns if c != "MONTHLY_RENT" and not c.startswith("_")]
    raw = [c for c in IDENTITY if c in tr.columns]
    rows = []
    for keys, label in [(raw, "raw competition columns"),
                        (feats, "every engineered feature")]:
        g = tr.groupby(keys, dropna=False)["MONTHLY_RENT"]
        n, dev = g.transform("size"), tr.MONTHLY_RENT - g.transform("mean")
        m = n > 1
        within = (dev[m] ** 2).sum() / (m.sum() - tr[m].groupby(keys, dropna=False).ngroups)
        rows.append(dict(grouped_on=label, n_columns=len(keys), twin_share=m.mean(),
                         within_rmse=np.sqrt(within)))
    out = pd.DataFrame(rows).round(3)
    out.to_csv("outputs/feature_floor.csv", index=False)
    print("\n=== Does any engineered feature split a group of identical rows? ===")
    print(out.to_string(index=False))
    same = np.isclose(rows[0]["within_rmse"], rows[1]["within_rmse"])
    print(f"\n{len(feats)} engineered features move the floor by "
          f"{rows[0]['within_rmse'] - rows[1]['within_rmse']:.4f} RMSE. "
          f"{'No feature separates identical rows.' if same else 'Some feature does separate them.'}")


if __name__ == "__main__":
    d = pd.read_parquet(PREDS) if "--score" in sys.argv else run()
    by_horizon(d)
    slices(d)
    tilt(d)
    floor()
    feature_floor()
