"""Ablation K: Do extra members improve the shipped blend out of selection?

Weights for each candidate set are grid-searched (step 0.05) on the three
rolling origins only. The 12-month holdout (cut 39) is then read off as the
out-of-selection check. This needs three sets of cached predictions: the shipped
configuration from confirm.py (Jc_ft05_blockinfo, meaning one index per flat
type shrunk halfway to the global index, plus the block-composition features),
the ported notebook from h2h.py, and the ridge and nb_plus candidates from
members.py.

    python src/blend_members.py
    python src/blend_members.py --inner month

The --inner option names which inner tuning protocol of the notebook members to
read, and is described in src/h2h.py. The default is the one the reported table
was built from. The repaired protocol writes its own table beside it.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import rmse, clip_rent, FOLDS  # noqa: E402
from h2h import preds_path, suffix_for  # noqa: E402

# The candidate sets to weight against the shipped pair, where nb is the ported
# notebook, nbp is that notebook given our block and coordinate features, and
# ridge is a linear model on the shipped detrended target.
SETS = [("nb",), ("nbp",), ("nb", "ridge"), ("nbp", "ridge"), ("nb", "nbp"), ("nb", "nbp", "ridge")]


def needs(tag):
    """Each cache and the script that writes it, so a missing one names its fix."""
    inner = "rows" if tag == "" else "month"
    return {"outputs/confirm_Jc_ft05_blockinfo.pkl": "python src/confirm.py Jc_ft05_blockinfo",
            "outputs/member_ridge.pkl": "python src/members.py ridge",
            f"outputs/member_nb_plus{tag}.pkl": f"python src/members.py nb_plus --inner {inner}"}


def load(inner="rows"):
    tag = suffix_for(inner)
    absent = [f"  {k}  <- {v}" for k, v in needs(tag).items() if not os.path.exists(k)]
    if absent:
        raise SystemExit("missing cached fold predictions:\n" + "\n".join(absent))
    P = pd.read_pickle("outputs/confirm_Jc_ft05_blockinfo.pkl")
    N = pd.read_pickle(preds_path(inner))
    R = pd.read_pickle("outputs/member_ridge.pkl")
    B = pd.read_pickle(f"outputs/member_nb_plus{tag}.pkl")
    M, Y = {}, {}
    for c in P:
        Y[c] = P[c]["y"]
        for src in (N[c]["y"], R[c]["y"], B[c]["y"]):
            assert np.array_equal(Y[c], src)
        M[c] = dict(ours=0.5 * (P[c]["lgbm"] + P[c]["cat"]), nb=N[c]["notebook"],
                    ridge=R[c]["pred"], nbp=B[c]["pred"])
    return M, Y


def score(M, Y, w):
    r = {c: rmse(Y[c], clip_rent(sum(w.get(k, 0) * M[c][k] for k in M[c]))) for c in M}
    return float(np.mean([r[c] for c in FOLDS])), r[39]


def main(inner="rows"):
    M, Y = load(inner)
    grid = np.round(np.arange(0, 0.61, 0.05), 2)
    rows = [dict(members="nb (shipped, w=0.30)", weights="nb=0.30",
                 **dict(zip(("roll16", "hold12"), score(M, Y, dict(ours=0.7, nb=0.3)))))]
    for extras in SETS:
        best = None
        for ws in itertools.product(grid, repeat=len(extras)):
            if sum(ws) > 0.8:
                continue
            w = dict(zip(extras, map(float, ws)), ours=1 - float(sum(ws)))
            s = score(M, Y, w)
            if best is None or s[0] < best[0][0]:
                best = (s, w)
        (r16, h12), w = best
        rows.append(dict(members="+".join(extras), roll16=r16, hold12=h12,
                         weights=" ".join(f"{k}={v:.2f}" for k, v in w.items())))
    res = pd.DataFrame(rows).round(2)
    out = f"outputs/ablation_blend_members{suffix_for(inner)}.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv[argv.index("--inner") + 1] if "--inner" in argv else "rows")
