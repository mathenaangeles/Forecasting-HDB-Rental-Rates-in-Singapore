"""Model ladder: every model class on one simplified setup, same splits.

The question is how much model complexity the data actually rewards, so nothing
here is tuned. Each model runs at default hyper-parameters on the base feature
set, with no target or spatial encoding, under three treatments of time:

    index    rent / global index is modelled, then multiplied back
    raw      unadjusted rent, retaining calendar month and age-related features
    raw_tm   raw rent with the month counter TM as a plain feature

`raw_tm` tests whether an explicit month coordinate substitutes for index
adjustment under the evaluated model specifications.

    python src/ladder.py                  # run everything missing, then score
    python src/ladder.py knn rf           # only these models
    python src/ladder.py --score          # re-score from the saved predictions

Per-row predictions go to outputs/ladder_preds.parquet, so any further metric or
slice can be computed without refitting. Scores go to outputs/ladder_metrics.csv
and the per-horizon breakdown to outputs/ladder_by_horizon.csv.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, fit_predict, clip_rent, HORIZON, FOLDS  # noqa: E402

CUTS = FOLDS + [39]  # Three 16-month rolling folds and the 12-month holdout.
MODELS = ["median", "ridge", "knn", "tree", "bag", "rf", "ada", "mlp", "xgb",
          "lgbm", "cat"]
SETUPS = {
    "index": dict(detrend=True),
    "raw": dict(detrend=False),
    "raw_tm": dict(detrend=False, extra_num=["TM"]),
}
PREDS = "outputs/ladder_preds.parquet"
KEEP = ["TM", "TOWN", "FLAT_TYPE"]  # Carried along for slicing the residuals.


def run(models):
    base = pd.read_parquet("outputs/base.parquet")
    tr = base[base._split == "train"]
    done = pd.read_parquet(PREDS) if os.path.exists(PREDS) else pd.DataFrame()
    for model in models:
        for setup, kw in SETUPS.items():
            if model == "median" and setup == "raw_tm":
                continue  # The lookup has no use for a numeric feature.
            if len(done) and ((done.model == model) & (done.setup == setup)).any():
                continue
            parts, t0 = [], time.time()
            for cut in CUTS:
                a = tr[tr.TM < cut]
                b = tr[(tr.TM >= cut) & (tr.TM < cut + HORIZON)]
                cfg = Config(name=f"{model}/{setup}", model=model, spatial=False, **kw)
                p = clip_rent(fit_predict(cfg, a, b, ref_t=cut - 1))
                parts.append(b[KEEP].assign(model=model, setup=setup, cut=cut,
                                            h=b.TM - (cut - 1),
                                            y=b.MONTHLY_RENT, pred=p))
            new = pd.concat(parts)
            done = pd.concat([done, new], ignore_index=True)
            done.to_parquet(PREDS)
            r = new.groupby("cut")[["y", "pred"]].apply(_rmse)
            print(f"{model:>6s}/{setup:<6s} roll16 {r[FOLDS].mean():7.2f}  "
                  f"hold12 {r[39]:7.2f}  ({time.time() - t0:.0f}s)", flush=True)


def _rmse(g):
    return float(np.sqrt(np.mean((g.pred - g.y) ** 2)))


def _metrics(g):
    e = g.pred - g.y
    return pd.Series(dict(
        rmse=np.sqrt(np.mean(e ** 2)),
        mae=np.mean(np.abs(e)),
        bias=np.mean(e),
        # Standard deviation of prediction errors after subtracting their mean.
        sd=np.std(e),
        r2=1 - np.sum(e ** 2) / np.sum((g.y - g.y.mean()) ** 2),
    ))


def score():
    d = pd.read_parquet(PREDS)
    per_cut = d.groupby(["model", "setup", "cut"])[["y", "pred"]].apply(_metrics)
    roll = per_cut[per_cut.index.get_level_values("cut").isin(FOLDS)]
    out = roll.groupby(["model", "setup"]).mean().add_prefix("roll16_")
    hold = per_cut.xs(39, level="cut")[["rmse", "bias"]].add_prefix("hold12_")
    out = out.join(hold).reset_index()
    out["model"] = pd.Categorical(out.model, MODELS)
    out = out.sort_values(["setup", "model"]).round(2)
    out.to_csv("outputs/ladder_metrics.csv", index=False)
    print(out.to_string(index=False))

    d = d[d.cut.isin(FOLDS)].assign(horizon=pd.cut(
        d.h, [0, 4, 8, 12, 16], labels=["1-4", "5-8", "9-12", "13-16"]))
    by_h = (d.groupby(["model", "setup", "horizon"], observed=True)[["y", "pred"]]
            .apply(_metrics)[["rmse", "bias"]].round(2).reset_index())
    by_h.to_csv("outputs/ladder_by_horizon.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--score" not in args:
        run(args or MODELS)
    score()
