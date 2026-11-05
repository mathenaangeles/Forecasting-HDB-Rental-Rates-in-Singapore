"""Leak-free hierarchical target encoding of the *detrended* rent ratio.

Location is by far the strongest predictor, but BLOCK has ~2 700 levels and
STREET ~1 150, far too many for one-hot and too sparse for plain mean encoding.
We use a hierarchy that starts at the global level and narrows through TOWN,
STREET and BLOCK. The same hierarchy is also grouped by FLAT_TYPE. Each level
is shrunk towards its parent, so a block with three observations mostly follows
its street while a block with eighty observations has more influence.

Crucially the quantity encoded is the *ratio* y / INDEX[t], not raw rent.
Raw-rent block means mix 2021 rents (~S$2 000) with 2025 rents (~S$3 200) and
are therefore contaminated by how a block's transactions happen to be spread
over time. Ratios are comparable across the whole window.

Three leakage regimes are supported through `mode`, and all three are evaluated
in the ablation. In `future` mode the source rows are strictly earlier than the
target rows, which is how test and holdout encodings are built. In `expanding`
mode a row may only see strictly earlier months, which is the default when
fitting. In `loo_month` mode a row may see every month except its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Each level is (name, keys, parent), and a parent of None means the global root.
HIERARCHY = [
    ("TWN_FT", ["TOWN", "FLAT_TYPE"], None),
    ("STR_FT", ["STREET", "FLAT_TYPE"], "TWN_FT"),
    ("BLK_FT", ["STREET", "BLOCK", "FLAT_TYPE"], "STR_FT"),
    ("TWN", ["TOWN"], None),
    ("STR", ["STREET"], "TWN"),
    ("BLK", ["STREET", "BLOCK"], "STR"),
    ("SUBZ_FT", ["SUBZONE", "FLAT_TYPE"], "TWN_FT"),
]
SMOOTH_K = 12.0  # Prior strength, expressed in equivalent observations.


def _agg(src: pd.DataFrame, tgt: pd.DataFrame, keys, mode: str, w: np.ndarray):
    """Weighted (sum, count) of `src.RATIO` aligned to every row of `tgt`."""
    s = src[keys].copy()
    s["_w"] = w
    s["_ws"] = w * src["RATIO"].to_numpy()

    if mode == "future":
        g = s.groupby(keys, observed=True)[["_w", "_ws"]].sum().reset_index()
        out = tgt[keys].merge(g, on=keys, how="left")
        return out["_ws"].to_numpy(), out["_w"].to_numpy()

    s["TM"] = src["TM"].to_numpy()
    per = s.groupby(keys + ["TM"], observed=True)[["_w", "_ws"]].sum().reset_index()

    if mode == "expanding":
        per = per.sort_values("TM")
        cum = per.groupby(keys, observed=True)[["_w", "_ws"]].cumsum()
        # Shifting by one month within the group means a row never sees its own.
        per[["_w", "_ws"]] = cum.groupby(
            [per[k] for k in keys], observed=True).shift(1)
    elif mode == "loo_month":
        tot = per.groupby(keys, observed=True)[["_w", "_ws"]].transform("sum")
        per[["_w", "_ws"]] = tot - per[["_w", "_ws"]]
    elif mode == "lagged":
        # Each training row sees the history as it stood `LAG` months before
        # its own month (cumulative up to and including month TM-LAG), so the
        # encodings are exactly as stale in training as they are at test time.
        per = per.sort_values("TM")
        per[["_w", "_ws"]] = per.groupby(keys, observed=True)[["_w", "_ws"]].cumsum()
        q = tgt[keys].copy()
        q["TM"] = (tgt["TM"].to_numpy() - tgt["LAG"].to_numpy()).astype(np.int64)
        per["TM"] = per["TM"].astype(np.int64)
        q["_row"] = np.arange(len(q))
        out = pd.merge_asof(q.sort_values("TM"), per, on="TM", by=keys,
                            direction="backward").sort_values("_row")
        return out["_ws"].to_numpy(), out["_w"].to_numpy()
    else:
        raise ValueError(mode)

    out = tgt[keys + ["TM"]].merge(per, on=keys + ["TM"], how="left")
    return out["_ws"].to_numpy(), out["_w"].to_numpy()


def encode(src: pd.DataFrame, tgt: pd.DataFrame, mode: str,
           half_life: float | None = None, k: float = SMOOTH_K) -> pd.DataFrame:
    """Return one shrunk encoding column (+ count) per hierarchy level.

    `src` must contain RATIO and TM. `tgt` needs the key columns and TM for
    time-aware modes. `half_life` is measured in months and reduces the weight
    of older observations. This lets a block move away from its long-run
    average when its rents have genuinely changed.
    """
    if half_life is None:
        w = np.ones(len(src))
    else:
        age = src["TM"].max() - src["TM"].to_numpy()
        w = 0.5 ** (age / half_life)

    root = float(np.average(src["RATIO"], weights=w))
    res, parents = {}, {}
    for name, keys, parent in HIERARCHY:
        ws, wn = _agg(src, tgt, keys, mode, w)
        ws, wn = np.nan_to_num(ws), np.nan_to_num(wn)
        prior = parents[parent] if parent else np.full(len(tgt), root)
        res[f"TE_{name}"] = (ws + k * prior) / (wn + k)
        res[f"N_{name}"] = wn
        parents[name] = res[f"TE_{name}"]

    out = pd.DataFrame(res, index=tgt.index)
    # The deviation of the finest level from its parent answers whether a block
    # is dearer than its street, which is a scale-free signal the trees can
    # split on directly.
    out["TE_BLK_DEV"] = out["TE_BLK_FT"] - out["TE_STR_FT"]
    out["TE_STR_DEV"] = out["TE_STR_FT"] - out["TE_TWN_FT"]
    return out


def spatial_encode(src: pd.DataFrame, tgt: pd.DataFrame, ks=(10, 30)) -> pd.DataFrame:
    """Observation-weighted mean detrended ratio of the k nearest *other* blocks.

    Covers the ~0.8% of test blocks never seen in training and injects a
    smooth spatial prior that the discrete hierarchy above cannot express.
    The block itself is always excluded, so this stays leak-free even when
    `src` and `tgt` are the same rows.
    """
    from sklearn.neighbors import BallTree

    bl = (src.groupby(["STREET", "BLOCK"], observed=True)
             .agg(LAT=("LATITUDE", "first"), LON=("LONGITUDE", "first"),
                  R=("RATIO", "mean"), N=("RATIO", "size")).reset_index())
    tree = BallTree(np.radians(bl[["LAT", "LON"]].values), metric="haversine")
    _, nb = tree.query(np.radians(tgt[["LATITUDE", "LONGITUDE"]].values),
                       k=min(max(ks) + 1, len(bl)))

    pos = pd.Series(np.arange(len(bl)),
                    index=pd.MultiIndex.from_frame(bl[["STREET", "BLOCK"]]))
    self_pos = pos.reindex(pd.MultiIndex.from_frame(
        tgt[["STREET", "BLOCK"]])).to_numpy()  # NaN when the block is unseen.
    self_pos = np.where(np.isnan(self_pos), -1, self_pos).astype(int)

    r, n = bl["R"].to_numpy(), bl["N"].to_numpy()
    out = {}
    for kk in ks:
        sel = nb[:, :kk + 1]
        w = n[sel] * (sel != self_pos[:, None])  # Drop the block itself.
        out[f"TE_SPATIAL{kk}"] = (r[sel] * w).sum(1) / np.maximum(w.sum(1), 1e-9)
    return pd.DataFrame(out, index=tgt.index)
