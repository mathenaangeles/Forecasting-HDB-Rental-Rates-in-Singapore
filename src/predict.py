"""Fit the final model on all labelled data and write a Kaggle submission.

    python src/predict.py                              # the shipped blend
    python src/predict.py blend_lgbm_cat_blockhist     # under a chosen name
    python src/predict.py shrink07 --shrink 0.7        # with a different index

The first argument names the file written under submissions/. The --shrink
option sets how far the per-flat-type index is pulled towards the global one,
which is the only difference between the two submissions we scored.

The --inner option selects the inner tuning protocol of the block-history
member, described in src/h2h.py. It defaults to rows, the protocol the files in
submissions/ were written under, so the default run reproduces a scored file
byte for byte. Asking for the repaired protocol with --inner month appends
_inner_month to the name, because those predictions were never scored and must
not land under a filename that carries a public score.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, fit_predict, TRAIN_END  # noqa: E402
from tuned_params import BEST_LGBM, BEST_CAT  # noqa: E402
from h2h import notebook_frame, notebook_fit_predict, suffix_for  # noqa: E402
from configs import BLOCK_INFO  # noqa: E402

# Ablation J settled on one index per flat type, pulled halfway towards the
# global index, together with the block composition features.
SHRINK = 0.5

# The weight search in outputs/blend_grid.json favored 45% LightGBM and 55%
# CatBoost. The best combinations were very close, so we use an even split.
# XGBoost received no weight near the optimum.
WEIGHTS = {"lgbm": 0.5, "cat": 0.5}


def members(shrink=SHRINK):
    """The two boosting members, with the number of seeds each is averaged over."""
    idx = dict(index_by="FLAT_TYPE", index_shrink=shrink, extra_num=list(BLOCK_INFO))
    return [
        (Config(name="lgbm", model="lgbm", params=dict(BEST_LGBM), **idx), 3),
        (Config(name="cat", model="cat", params=dict(BEST_CAT), **idx), 2),
    ]


# The third member is the team's notebook model, ported in h2h.py. On its own,
# it loses to the blend on every split, but its errors differ enough to help.
# A 30% weight lowers the rolling-origin RMSE from 518.56 to 516.66 and the
# holdout RMSE from 466.54 to 463.77 under the inner protocol the scored files
# were written with, and to 516.54 and 463.82 under the repaired one. Thirty
# percent is the best weight either way. See outputs/ablation_head_to_head.csv
# and outputs/ablation_head_to_head_repaired.csv.
NOTEBOOK_WEIGHT = 0.3

# Bounds match the observed training range. The cached repaired, single-seed
# validation blend lies inside them, so clipping is inactive in that check.
CLIP = (300.0, 7600.0)


def build_predictions(df, member_list=None, verbose=True, inner="rows"):
    tr = df[df._split == "train"]
    te = df[df._split == "test"]
    parts = {}
    for cfg, n_seeds in member_list or members():
        preds = []
        seed_key = "random_seed" if cfg.model == "cat" else "random_state"
        for s in range(n_seeds):
            c = replace(cfg, params={**cfg.params, seed_key: s})
            preds.append(fit_predict(c, tr, te, ref_t=TRAIN_END))
            if verbose:
                print(f"  {cfg.name} seed {s} done", flush=True)
        parts[cfg.name] = np.mean(preds, axis=0)
    blend = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS[k] for k in parts)
    if NOTEBOOK_WEIGHT:
        nb = notebook_frame(df)
        parts["notebook"], bp = notebook_fit_predict(nb[nb._split == "train"],
                                                     nb[nb._split == "test"], inner=inner)
        if verbose:
            print(f"  notebook member done  {bp}", flush=True)
        blend = (1 - NOTEBOOK_WEIGHT) * blend + NOTEBOOK_WEIGHT * parts["notebook"]
    return np.clip(blend, *CLIP), parts


def main(name="submission", shrink=SHRINK, inner="rows"):
    suffix_for(inner)
    df = pd.read_parquet("outputs/base.parquet")
    pred, parts = build_predictions(df, members(shrink), inner=inner)
    os.makedirs("submissions", exist_ok=True)
    out = pd.DataFrame({"Id": np.arange(len(pred)), "Predicted": pred})
    # A scored filename means the protocol it was scored under, so predictions
    # from the other one are written under a name of their own.
    path = f"submissions/{name}.csv" if inner == "rows" else f"submissions/{name}_inner_month.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print(out.Predicted.describe().round(1).to_string())
    for k, v in parts.items():
        print(f"  corr(blend, {k}) = {np.corrcoef(pred, v)[0,1]:.4f}   mean {v.mean():.1f}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    shrink, inner = SHRINK, "rows"
    if "--shrink" in args:
        i = args.index("--shrink")
        shrink = float(args[i + 1])
        del args[i:i + 2]
    if "--inner" in args:
        i = args.index("--inner")
        inner = args[i + 1]
        del args[i:i + 2]

    main(args[0] if args else "submission", shrink, inner)
