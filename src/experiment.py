"""Rolling-origin evaluation harness.

Each fold matches the actual forecasting setup. It trains on months before the
cutoff and predicts the next HORIZON months. The market index is fitted and
forecast again inside each fold, so the RMSE includes errors in the future
price level. Random K-fold validation would hide those errors.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import encode as enc
import tindex

warnings.filterwarnings("ignore")

HORIZON = 16  # test.csv spans the 16 months after the last training month.
TRAIN_END = 50  # The last labelled month, 2025-03.
FOLDS = [27, 31, 35]  # Cut months, each giving a validation window of HORIZON.
RENT_CLIP = (300.0, 8000.0)  # This safety limit does not affect the folds.

BASE_NUM = [
    "FLOOR_AREA_SQM", "N_ROOMS", "AREA_PER_ROOM", "AREA_VS_TYPE",
    "LEASE_AGE", "LEASE_REMAIN", "BLOCK_AGE", "MAX_FLOOR", "IS_TOP_BLOCK",
    "LATITUDE", "LONGITUDE", "DIST_CBD",
    "MRT_NEAR1", "MRT_NEAR2", "MRT_CNT0.5", "MRT_CNT1.0",
    "MRTPLAN_NEAR1", "MALL_NEAR1", "MALL_CNT1.0",
    "PRISCH_NEAR1", "PRISCH_CNT1.0", "SECSCH_NEAR1",
    "MONTH_OF_YEAR", "HORIZON_M",
]
BASE_CAT = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "PLANNING_AREA", "REGION", "SUBZONE"]


@dataclass
class Config:
    name: str = "default"
    detrend: bool = True  # Model y / INDEX instead of raw y.
    te_mode: str = "none"  # One of expanding, loo_month, lagged or none.
    te_half_life: float | None = None
    te_k: float = 12.0
    spatial: bool = True
    spatial_ks: tuple = (10, 30)
    index_method: str = "damped"
    index_window: int = 12
    index_damp: float = 0.6
    log_target: bool = False
    model: str = "lgbm"
    params: dict = field(default_factory=dict)
    drop: list = field(default_factory=lambda: ["HORIZON_M"])
    sample_half_life: float | None = None  # Recency weighting of the training rows.
    train_window: int = 0  # Keep only the last N months, or all of them when 0.
    extra_num: list = field(default_factory=list)
    extra_cat: list = field(default_factory=list)
    index_grain: tuple = ("STREET", "BLOCK", "FLAT_TYPE")  # The unit of the ratio index.
    index_by: str | None = None  # A separate index per segment, e.g. per FLAT_TYPE.
    index_shrink: float = 1.0  # Segment index = global * (segment / global) ** shrink.
    te_levels: tuple | None = None  # Subset of encode.HIERARCHY names, or all when None.
    te_age: bool = False  # Expose encoding staleness to the model as a feature.
    geo_k: int = 0  # Replace the administrative columns with k k-means clusters on coordinates.
    pca_k: int = 0  # Replace the numeric columns with their first k principal components.


def _feature_frame(cfg, src, tgt, mode, ref_t):
    """Assemble the model matrix for `tgt`, with encodings drawn from `src`."""
    cats = BASE_CAT + cfg.extra_cat
    X = tgt[BASE_NUM[:-1] + cfg.extra_num + cats].copy()
    X["HORIZON_M"] = tgt["TM"].to_numpy() - ref_t  # Months ahead of the train end.
    if cfg.te_mode != "none":
        E = enc.encode(src, tgt, mode, cfg.te_half_life, cfg.te_k)
        if cfg.te_levels is not None:
            keep = [c for c in E if any(c.endswith("_" + l) for l in cfg.te_levels)]
            E = E[keep]
        X = pd.concat([X, E], axis=1)
        if cfg.te_age:
            X["TE_AGE"] = tgt["LAG"].to_numpy() if mode == "lagged" else X["HORIZON_M"]
    if cfg.spatial:
        X = pd.concat([X, enc.spatial_encode(src, tgt, cfg.spatial_ks)], axis=1)
    if cfg.geo_k:
        X["GEO_CLUSTER"] = _geo_cluster(cfg.geo_k, src, tgt)
        X = X.drop(columns=["TOWN", "PLANNING_AREA", "REGION", "SUBZONE"],
                   errors="ignore")
        cats = [c for c in cats if c in X] + ["GEO_CLUSTER"]
    for c in cfg.drop:
        X = X.drop(columns=[c], errors="ignore")
    for c in cats:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def _geo_cluster(k, src, tgt):
    """Partition the island into k regions by k-means on block coordinates.

    This is the data-driven alternative to the administrative TOWN, PLANNING
    AREA, REGION and SUBZONE columns. The centroids come from the training side
    of the fold alone, and blocks without coordinates fall into their own level.
    """
    from sklearn.cluster import KMeans
    cols = ["LATITUDE", "LONGITUDE"]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(src[cols].dropna())
    out = np.full(len(tgt), -1)
    ok = tgt[cols].notna().all(axis=1).to_numpy()
    if ok.any():
        out[ok] = km.predict(tgt.loc[ok, cols])
    return pd.Categorical(out)


def _pca(k, Xtr, Xte):
    """Replace the numeric columns with their first k principal components.

    The components are fitted on the training side only. Categorical columns
    pass through untouched, since PCA is defined on the numeric block.
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    num = [c for c in Xtr.columns if str(Xtr[c].dtype) != "category"]
    cat = [c for c in Xtr.columns if c not in num]
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         PCA(n_components=k, random_state=0)).fit(Xtr[num])
    names = [f"PC{i + 1}" for i in range(k)]
    A = pd.DataFrame(pipe.transform(Xtr[num]), columns=names, index=Xtr.index)
    B = pd.DataFrame(pipe.transform(Xte[num]), columns=names, index=Xte.index)
    return (pd.concat([A, Xtr[cat]], axis=1), pd.concat([B, Xte[cat]], axis=1))


def _align(Xa, Xb):
    """Give train and predict frames identical category dictionaries."""
    for c in Xa.columns:
        if str(Xa[c].dtype) == "category":
            cats = Xa[c].cat.categories.union(Xb[c].cat.categories)
            Xa[c] = Xa[c].cat.set_categories(cats)
            Xb[c] = Xb[c].cat.set_categories(cats)
    return Xa, Xb[Xa.columns]


def make_model(cfg):
    p = dict(cfg.params)
    if cfg.model == "lgbm":
        import lightgbm as lgb
        d = dict(n_estimators=1200, learning_rate=0.04, num_leaves=127,
                 min_child_samples=40, subsample=0.85, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=5.0, n_jobs=-1,
                 random_state=0, verbose=-1)
        d.update(p)
        return lgb.LGBMRegressor(**d)
    if cfg.model == "cat":
        from catboost import CatBoostRegressor
        d = dict(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=4.0,
                 random_seed=0, verbose=0, allow_writing_files=False)
        d.update(p)
        return CatBoostRegressor(**d)
    if cfg.model == "xgb":
        import xgboost as xgb
        d = dict(n_estimators=1500, learning_rate=0.04, max_depth=8,
                 min_child_weight=20, subsample=0.85, colsample_bytree=0.7,
                 reg_lambda=5.0, n_jobs=-1, random_state=0,
                 enable_categorical=True, tree_method="hist")
        d.update(p)
        return xgb.XGBRegressor(**d)
    if cfg.model == "median":
        return GroupMedian(**p)
    if cfg.model in ("ridge", "knn", "mlp"):
        # Distance- and gradient-based models need one-hot categories and
        # standardised numerics, or the widest-ranging column decides everything.
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        num = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        cat = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
        ct = ColumnTransformer([("c", cat, _cat_cols)], remainder=num)
        if cfg.model == "ridge":
            from sklearn.linear_model import Ridge
            est = Ridge(alpha=p.get("alpha", 10.0))
        elif cfg.model == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            d = dict(n_neighbors=20, weights="distance", n_jobs=-1)
            d.update(p)
            est = KNeighborsRegressor(**d)
        else:
            from sklearn.neural_network import MLPRegressor
            d = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                     max_iter=200, random_state=0)
            d.update(p)
            est = MLPRegressor(**d)
        return make_pipeline(ct, est)
    if cfg.model in ("tree", "rf", "bag", "ada"):
        # A tree only needs an ordering to split on, so integer codes are enough.
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ct = ColumnTransformer([("c", cat, _cat_cols)], remainder="passthrough")
        if cfg.model == "tree":
            from sklearn.tree import DecisionTreeRegressor
            d = dict(min_samples_leaf=50, random_state=0)
            d.update(p)
            est = DecisionTreeRegressor(**d)
        elif cfg.model == "bag":
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor
            d = dict(n_estimators=50, n_jobs=-1, random_state=0)
            d.update(p)
            est = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), **d)
        elif cfg.model == "ada":
            from sklearn.ensemble import AdaBoostRegressor
            from sklearn.tree import DecisionTreeRegressor
            d = dict(n_estimators=100, learning_rate=0.1, loss="square",
                     random_state=0)
            d.update(p)
            est = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), **d)
        else:
            from sklearn.ensemble import RandomForestRegressor
            d = dict(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                     n_jobs=-1, random_state=0)
            d.update(p)
            est = RandomForestRegressor(**d)
        return make_pipeline(ct, est)
    raise ValueError(cfg.model)


def _cat_cols(d):
    return d.select_dtypes("category").columns.tolist()


class GroupMedian:
    """Predict the training median of a row's cell, e.g. town x flat type.

    Cells unseen in training fall back to the median of the last key alone,
    then to the global median.
    """

    def __init__(self, keys=("TOWN", "FLAT_TYPE")):
        self.keys = list(keys)

    def fit(self, X, y):
        d = X[self.keys].astype(str)
        d["y"] = np.asarray(y)
        self.cell_ = d.groupby(self.keys)["y"].median()
        self.last_ = d.groupby(self.keys[-1])["y"].median()
        self.all_ = float(d["y"].median())
        return self

    def predict(self, X):
        d = X[self.keys].astype(str)
        p = pd.MultiIndex.from_frame(d).map(self.cell_.to_dict())
        p = pd.Series(p, dtype=float).fillna(d[self.keys[-1]].map(self.last_).reset_index(drop=True))
        return p.fillna(self.all_).to_numpy()


def _index_lookup(cfg, tr):
    """Build a function that looks up the index value for each row.

    With `index_by`, each segment gets its own ratio index and damped forecast,
    shrunk towards the global one. Segments with too little data, or no data in
    the last month, use the global index instead.
    """
    def one(part):
        i = tindex.fit_index(part, grain=list(cfg.index_grain))
        return tindex.extrapolate(i, HORIZON + 2, cfg.index_window,
                                  cfg.index_damp, cfg.index_method)

    if not cfg.detrend:
        g = pd.Series(1.0, index=np.arange(int(tr["TM"].max()) + HORIZON + 3))
        return lambda f: g.reindex(f["TM"]).to_numpy()
    g = one(tr)
    segs = {}
    if cfg.index_by:
        n_t = tr["TM"].max() + 1
        for s, part in tr.groupby(cfg.index_by, observed=True):
            if len(part) < 3000 or part["TM"].nunique() < n_t:
                continue
            segs[s] = g * (one(part) / g) ** cfg.index_shrink

    def lookup(f):
        out = g.reindex(f["TM"]).to_numpy().copy()
        for s, si in segs.items():
            m = (f[cfg.index_by] == s).to_numpy()
            out[m] = si.reindex(f["TM"][m]).to_numpy()
        return out
    return lookup


def fit_predict(cfg, tr, te, ref_t, fit_mode=None, return_model=False):
    """Train on `tr`, predict `te`. `ref_t` = last training month."""
    idx = _index_lookup(cfg, tr)

    tr = tr.copy()
    tr["RATIO"] = tr["MONTHLY_RENT"] / idx(tr)
    y = np.log(tr["RATIO"]) if cfg.log_target else tr["RATIO"]

    mode = fit_mode or cfg.te_mode
    if mode == "lagged":
        tr["LAG"] = np.random.default_rng(0).integers(1, HORIZON + 1, len(tr))
    Xtr = _feature_frame(cfg, tr, tr, mode if mode != "none" else "expanding", ref_t)
    Xte = _feature_frame(cfg, tr, te, "future", ref_t)
    Xtr, Xte = _align(Xtr, Xte)
    if cfg.pca_k:
        Xtr, Xte = _pca(cfg.pca_k, Xtr, Xte)

    # Keep the full history for encoding so each block has more observations.
    # Training windows and weights apply only to the rows used to fit the model.
    keep = np.ones(len(tr), bool)
    if cfg.train_window:
        keep = (tr["TM"].to_numpy() > ref_t - cfg.train_window)
        Xtr, y = Xtr[keep], y[keep]

    m = make_model(cfg)
    fit_kw = {}
    if cfg.model == "cat":
        # CatBoost will not accept the pandas category dtype implicitly. The
        # columns have to be listed, and it wants them as plain strings.
        cat_cols = Xtr.select_dtypes("category").columns.tolist()
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype(str)
            Xte[c] = Xte[c].astype(str)
        fit_kw["cat_features"] = cat_cols
    if cfg.sample_half_life:
        # Detrending removes the drift in the price level, but the
        # cross-sectional structure may itself re-rate. Down-weighting old rows
        # lets the model track that without discarding them outright.
        age = ref_t - tr["TM"].to_numpy()[keep]
        m.fit(Xtr, y, sample_weight=0.5 ** (age / cfg.sample_half_life), **fit_kw)
    else:
        m.fit(Xtr, y, **fit_kw)
    p = m.predict(Xte)
    if cfg.log_target:
        p = np.exp(p)
    pred = p * idx(te)
    return (pred, m, Xtr) if return_model else pred


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def clip_rent(p):
    return np.clip(p, *RENT_CLIP)


def evaluate(cfg, df, folds=FOLDS, verbose=True):
    tr_all = df[df["_split"] == "train"]
    rows = []
    for cut in folds:
        a = tr_all[tr_all["TM"] < cut]
        b = tr_all[(tr_all["TM"] >= cut) & (tr_all["TM"] < cut + HORIZON)]
        p = fit_predict(cfg, a, b, ref_t=cut - 1)
        p = clip_rent(p)
        r = rmse(b["MONTHLY_RENT"], p)
        h = b["TM"].to_numpy() - (cut - 1)
        early = rmse(b["MONTHLY_RENT"][h <= 8], p[h <= 8])
        late = rmse(b["MONTHLY_RENT"][h > 8], p[h > 8])
        rows.append(dict(cut=cut, rmse=r, rmse_h1_8=early, rmse_h9_16=late,
                         bias=float(np.mean(p - b["MONTHLY_RENT"]))))
        if verbose:
            print(f"  cut={cut:>2}  RMSE={r:7.2f}  (h1-8 {early:7.2f} | h9-16 {late:7.2f})"
                  f"  bias={rows[-1]['bias']:+7.1f}")
    res = pd.DataFrame(rows)
    mean = res["rmse"].mean()
    if verbose:
        print(f"  {cfg.name:<28s} MEAN RMSE = {mean:7.2f}   (last fold {res.rmse.iloc[-1]:7.2f})")
    return mean, res
