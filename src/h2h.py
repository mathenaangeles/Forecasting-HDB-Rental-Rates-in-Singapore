"""Head-to-head against the team's leaderboard notebook (HistGB + block
history, LB 436.05), scored on identical splits.

The two CV numbers were never comparable: the notebook reports a single
12-month holdout (months 39-50), we report three 16-month rolling origins.
This script runs the notebook on both protocols with the same features, search
grid and block-history rule. It also evaluates our models on both protocols.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, TimeSeriesSplit

sys.path.insert(0, os.path.dirname(__file__))
from experiment import Config, fit_predict, rmse, clip_rent, HORIZON, FOLDS  # noqa: E402
from tuned_params import BEST_LGBM, BEST_CAT  # noqa: E402

NUM = ["FLOOR_AREA_SQM", "LEASE_AGE_YR", "MTH_SINCE_START", "MTH_OF_YR", "DIST_MRT_KM",
       "DIST_MALL_KM", "DIST_SCH_KM", "COE_PREMIUM_AVG", "STOCK_CLOSE_AVG"]
CAT = ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
HIST = ["HIST_BLK", "HIST_BLK_N"]
KEYS = ["STREET", "BLOCK", "FLAT_TYPE"]
GRID = {"learning_rate": [0.03, 0.05, 0.1, 0.15], "max_leaf_nodes": [15, 31, 63],
        "max_iter": [100, 200, 300], "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0]}


def notebook_frame(df):
    """The notebook's feature set, rebuilt from our (identically normalised) base."""
    d = df.copy()
    d["MTH_SINCE_START"] = d["TM"] + 1
    d["MTH_OF_YR"] = d["MONTH_OF_YEAR"]
    d["LEASE_AGE_YR"] = d["TM"] // 12 + 2021 - d["LEASE_COMMENCE_DATE"]
    # The notebook measured to every station in the file, open or planned.
    d["DIST_MRT_KM"] = np.minimum(d["MRT_NEAR1"], d["MRTPLAN_NEAR1"])
    d["DIST_MALL_KM"] = d["MALL_NEAR1"]
    d["DIST_SCH_KM"] = d["PRISCH_NEAR1"]

    mth = dict(jan=1, feb=2, mar=3, apr=4, may=5, jun=6, jul=7, aug=8, sep=9, oct=10, nov=11, dec=12)
    coe = pd.read_csv("data/auxiliary/sg-coe-prices.csv")
    coe["P"] = coe["QUOTA_PREMIUM"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
    coe["MTH_SINCE_START"] = (coe["YEAR"] - 2021) * 12 + coe["MONTH"].map(mth)
    st = pd.read_csv("data/auxiliary/sg-stock-prices.csv")
    st["DATE"] = pd.to_datetime(st["DATE"])
    st["MTH_SINCE_START"] = (st["DATE"].dt.year - 2021) * 12 + st["DATE"].dt.month
    d = d.merge(coe.groupby("MTH_SINCE_START")["P"].mean().rename("COE_PREMIUM_AVG"),
                on="MTH_SINCE_START", how="left")
    d = d.merge(st.dropna(subset=["CLOSE"]).groupby("MTH_SINCE_START")["CLOSE"].mean()
                  .rename("STOCK_CLOSE_AVG"), on="MTH_SINCE_START", how="left")
    for c in CAT:
        d[c] = pd.Categorical(d[c])
    return d


def block_history(source, target, is_future):
    if is_future:
        s = source.groupby(KEYS, observed=True)["MONTHLY_RENT"].agg(HIST_BLK="mean", HIST_BLK_N="size").reset_index()
        out = target[KEYS].merge(s, on=KEYS, how="left")
    else:
        m = (source.groupby(KEYS + ["TM"], observed=True)["MONTHLY_RENT"]
                   .agg(s="sum", c="size").reset_index().sort_values("TM"))
        m[["cs", "cc"]] = m.groupby(KEYS, observed=True)[["s", "c"]].cumsum()
        m[["cs", "cc"]] = m.groupby(KEYS, observed=True)[["cs", "cc"]].shift(1)
        m["HIST_BLK"], m["HIST_BLK_N"] = m["cs"] / m["cc"], m["cc"]
        out = target[KEYS + ["TM"]].merge(m[KEYS + ["TM"] + HIST], on=KEYS + ["TM"], how="left")
    return out[HIST].to_numpy()


def build_X(target, source, is_future, num=NUM):
    X = target[num + CAT].reset_index(drop=True).copy()
    X[HIST] = block_history(source, target, is_future)
    return X


# The two inner tuning protocols, and the filename tag each one writes under.
# Every artifact is named for the protocol that produced it, so a run cannot
# overwrite the other protocol's results and a reader cannot mistake them.
INNER = {"rows": "", "month": "_repaired"}


def suffix_for(inner: str) -> str:
    """The filename tag for one inner tuning protocol, checking the name."""
    if inner not in INNER:
        raise SystemExit(f"unknown inner protocol {inner!r}. Pass one of "
                         f"{', '.join(sorted(INNER))}")
    return INNER[inner]


def preds_path(inner: str = "rows") -> str:
    """The cached fold predictions written under one inner tuning protocol.

    The protocol is named by the caller rather than guessed from which caches
    happen to exist, because the two differ by about two tenths and mixing them
    across a table would leave its rows incomparable.
    """
    path = f"outputs/h2h_preds{suffix_for(inner)}.pkl"
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}. Write it with\n"
                         f"  python src/h2h.py --inner {inner}")
    return path


def month_cv(tm, horizon=HORIZON, n_splits=3, min_train=12):
    """Inner origins for the hyperparameter search, as whole months.

    Returns a list of (origin, length) pairs. Each fold trains on the months
    before its origin and is scored on the months from the origin onward, which
    is the outer protocol applied one level down. The horizon is held at the
    outer one wherever the span allows it, and is shortened only when holding
    it would leave fewer than two usable origins.
    """
    span = int(np.asarray(tm).max()) + 1
    for h in range(min(horizon, span - min_train), 2, -1):
        for spacing in (4, 2, 1):
            origins = sorted({span - h - spacing * k for k in range(n_splits)})
            folds = [(o, h) for o in origins if o >= min_train]
            if len(folds) >= 2:
                return folds
    raise ValueError(f"an inner span of {span} months is too short to split")


def _search_month_folds(a, num, n_iter=8, seed=42):
    """Random search over GRID, scored on month-aligned inner origins.

    sklearn's TimeSeriesSplit cuts on row position rather than on month, so a
    fold boundary lands inside a month and the search is scored partly on
    months it has already trained on. The month counter is a feature, so that
    rewards settings which fit a month level instead of carrying one forward,
    which is the opposite of what the test window asks for. Each fold here also
    rebuilds the block history the way deployment does, from the fold's own
    training months only, so the feature the search is scored on is the feature
    the final model will be given.

    The candidates are drawn by the same sampler, with the same grid, count and
    seed as the positional search, so the two protocols consider an identical
    set of settings and the comparison isolates how they are scored.
    """
    cands = list(ParameterSampler(GRID, n_iter=n_iter, random_state=seed))
    folds = []
    for o, h in month_cv(a["TM"].to_numpy()):
        trn, val = a[a.TM < o], a[(a.TM >= o) & (a.TM < o + h)]
        folds.append((build_X(trn, trn, False, num), trn["MONTHLY_RENT"].to_numpy(),
                      build_X(val, trn, True, num), val["MONTHLY_RENT"].to_numpy()))
    best, best_score = None, np.inf
    for p in cands:
        scores = []
        for Xt, yt, Xv, yv in folds:
            m = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                              random_state=42, **p)
            m.fit(Xt, yt)
            scores.append(rmse(yv, clip_rent(m.predict(Xv))))
        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = p, score
    return best


def notebook_fit_predict(a, b, num=NUM, inner="rows"):
    """Fit the notebook model on `a` and predict `b`.

    The inner argument selects the tuning protocol. "month" is the repaired one
    described in _search_month_folds. "rows" is sklearn's positional
    TimeSeriesSplit, which is what produced the scored submissions and the
    shipped head-to-head table. It is the default so that every script
    reproduces the numbers the report quotes unless asked for the other one.
    """
    suffix_for(inner)
    a = a.sort_values("TM")
    Xa = build_X(a, a, False, num)
    if inner == "rows":
        s = RandomizedSearchCV(HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42),
                               GRID, n_iter=8, cv=TimeSeriesSplit(n_splits=4),
                               scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
        s.fit(Xa, a["MONTHLY_RENT"])
        return s.best_estimator_.predict(build_X(b, a, True, num)), s.best_params_
    bp = _search_month_folds(a, num)
    m = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42, **bp)
    m.fit(Xa, a["MONTHLY_RENT"].to_numpy())
    return m.predict(build_X(b, a, True, num)), bp


def ours(a, b, cut):
    lg = fit_predict(Config(model="lgbm", params=dict(BEST_LGBM)), a, b, ref_t=cut - 1)
    ca = fit_predict(Config(model="cat", params=dict(BEST_CAT)), a, b, ref_t=cut - 1)
    return lg, 0.5 * lg + 0.5 * ca


def main(inner="rows"):
    base = pd.read_parquet("outputs/base.parquet")
    nb = notebook_frame(base)
    tr_b, tr_n = base[base._split == "train"], nb[nb._split == "train"]
    rows, preds = [], {}
    # Protocol 1 is our three 16-month origins, protocol 2 the notebook's
    # 12-month holdout.
    for proto, cut in [("rolling16", c) for c in FOLDS] + [("holdout12", 39)]:
        t0 = time.time()
        m_b = (tr_b.TM >= cut) & (tr_b.TM < cut + HORIZON)
        m_n = (tr_n.TM >= cut) & (tr_n.TM < cut + HORIZON)
        y = tr_n.loc[m_n, "MONTHLY_RENT"].to_numpy()
        p_nb, bp = notebook_fit_predict(tr_n[tr_n.TM < cut], tr_n[m_n], inner=inner)
        yb = tr_b.loc[m_b, "MONTHLY_RENT"].to_numpy()
        lg, bl = ours(tr_b[tr_b.TM < cut], tr_b[m_b], cut)
        assert np.array_equal(y, yb)  # The same rows, in the same order.
        preds[cut] = dict(y=y, notebook=p_nb, lgbm=lg, blend=bl)
        r = dict(protocol=proto, cut=cut, n=len(y),
                 notebook=rmse(y, clip_rent(p_nb)), notebook_bias=float(np.mean(p_nb - y)),
                 ours_lgbm=rmse(yb, clip_rent(lg)), ours_blend=rmse(yb, clip_rent(bl)),
                 ours_bias=float(np.mean(bl - yb)), notebook_params=str(bp),
                 secs=round(time.time() - t0))
        rows.append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    # The shipped table was produced under the positional inner split. The
    # repaired protocol writes beside it rather than over it.
    suffix = suffix_for(inner)
    res.to_csv(f"outputs/ablation_head_to_head{suffix}.csv", index=False)
    pd.to_pickle(preds, f"outputs/h2h_preds{suffix}.pkl")
    # Does the notebook add anything on top of our blend? The weight shown is
    # the one given to the notebook.
    for w in (0.0, 0.1, 0.2, 0.3, 0.5):
        r = {c: rmse(v["y"], clip_rent((1 - w) * v["blend"] + w * v["notebook"]))
             for c, v in preds.items()}
        print(f"  w_notebook={w:.1f}  rolling16={np.mean([r[c] for c in FOLDS]):7.2f}  holdout12={r[39]:7.2f}")
    print(res.groupby("protocol")[["notebook", "ours_lgbm", "ours_blend"]].mean().round(2))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[args.index("--inner") + 1] if "--inner" in args else "rows")
