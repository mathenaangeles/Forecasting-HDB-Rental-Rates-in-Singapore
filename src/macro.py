"""Monthly macro indicators from the auxiliary COE and stock-price files.

Both auxiliary series extend to 2026-07, i.e. they *cover the test window*.
They are therefore contemporaneous information legitimately available at
prediction time, and are the only in-competition signal about where the
rental market went after the last labelled month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = "data"
# SGX-listed property developers and S-REITs: the closest available proxy for
# the Singapore real-estate cycle.
PROPERTY = [
    "Ascendas Reit", "CapitaLand Ascott Trust", "CapitaLand Investment Limited",
    "CapitaLand Mall Trust", "City Developments", "Frasers Logistics & Industrial Trust",
    "Hotel Properties Limited", "Keppel REIT", "Mapletree Commercial Trust",
    "Mapletree Industrial Trust", "Mapletree Logistics Trust", "Suntec REIT",
    "Singapore Land", "UOL Group", "Yanlord",
]
BANKS = ["DBS Group", "OCBC Bank", "UOB"]
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _basket(st: pd.DataFrame, names) -> pd.Series:
    """Equal-weight index of log adjusted close, rebased to its first month."""
    sub = st[st.NAME.isin(names)].dropna(subset=["ADJUSTED_CLOSE"])
    m = sub.groupby(["NAME", "TM"])["ADJUSTED_CLOSE"].mean()
    lg = np.log(m).unstack(0)
    lg = lg - lg.iloc[0]  # Rebase each name before averaging them.
    return lg.mean(axis=1)


def monthly(data_dir: str = DATA) -> pd.DataFrame:
    st = pd.read_csv(f"{data_dir}/auxiliary/sg-stock-prices.csv")
    d = pd.to_datetime(st["DATE"])
    st["TM"] = (d.dt.year - 2021) * 12 + d.dt.month - 1

    out = pd.DataFrame({
        "MACRO_PROP": _basket(st, PROPERTY),
        "MACRO_BANK": _basket(st, BANKS),
        "MACRO_ALL": _basket(st, st.NAME.unique()),
    })

    coe = pd.read_csv(f"{data_dir}/auxiliary/sg-coe-prices.csv")
    coe["QP"] = coe["QUOTA_PREMIUM"].replace(r"[\$,]", "", regex=True).astype(float)
    coe["TM"] = (coe["YEAR"] - 2021) * 12 + coe["MONTH"].map(MONTHS) - 1
    out["MACRO_COE"] = np.log(coe.groupby("TM")["QP"].mean())
    out["MACRO_COE"] -= out["MACRO_COE"].iloc[0]
    out = out.join(coe_by_category(data_dir))

    return out.sort_index().ffill()


def coe_by_category(data_dir: str = DATA) -> pd.DataFrame:
    """Log COE premium for each bidding category, rebased to its first month.

    The categories are a for cars up to 1600cc, b for larger cars, c for goods
    vehicles and buses, and e for open bids. Categories a and b track private
    car demand most directly, so they are worth keeping apart from the pooled
    MACRO_COE series. The file has no category d.
    """
    coe = pd.read_csv(f"{data_dir}/auxiliary/sg-coe-prices.csv")
    coe["QP"] = coe["QUOTA_PREMIUM"].replace(r"[\$,]", "", regex=True).astype(float)
    coe["TM"] = (coe["YEAR"] - 2021) * 12 + coe["MONTH"].map(MONTHS) - 1
    out = np.log(coe.pivot_table(index="TM", columns="CATEGORY", values="QP", aggfunc="mean"))
    out.columns = [f"MACRO_COE_{c.upper()}" for c in out.columns]
    return (out - out.iloc[0]).sort_index().ffill()


# Each method is a forecaster of the rent index. The file has no category D, so
# the per-category split uses A, B, C and E.
METHODS = {
    "flat": None,
    "damped": None,
    "lvl:MACRO_PROP": ["MACRO_PROP"],
    "lvl:MACRO_COE": ["MACRO_COE"],
    "lvl:MACRO_COE_A": ["MACRO_COE_A"],
    "lvl:MACRO_COE_by_cat": ["MACRO_COE_A", "MACRO_COE_B", "MACRO_COE_C", "MACRO_COE_E"],
    "d12:MACRO_COE": ["MACRO_COE"],
}


def backtest(tr: pd.DataFrame, cuts=(27, 31, 35, 39), horizon: int = 16,
             data_dir: str = DATA) -> pd.DataFrame:
    """Test whether the macro series forecast the rent index better than extrapolation.

    At every cut the index is fitted on the months before the cut and then
    forecast `horizon` months ahead. The forecast either uses the index alone
    (flat, or the damped trend we ship) or a ridge regression on the macro
    series, whose values are known for the forecast months. The `lvl` methods
    regress the log index on the log series, and the `d12` method does the same
    in 12-month differences.

    Every forecaster, the macro regressions included, is fitted on the index
    estimated from months before its own cut, so no method sees a rent approved
    after its origin. Only the realized index it is scored against comes from a
    fit on all of `tr`, which is what actually happened and is the target rather
    than an input. Errors are mean absolute, in percent of the level.
    """
    import tindex
    from sklearn.linear_model import Ridge

    M = monthly(data_dir)
    li = np.log(tindex.fit_index(tr))
    rows = []
    for cut in cuts:
        h = np.arange(cut, cut + horizon)
        h = h[h <= li.index.max()]
        true = li.loc[h].to_numpy() - li.loc[cut - 1]
        own = tindex.fit_index(tr[tr.TM < cut])
        lown = np.log(own)
        for name, cols in METHODS.items():
            kind = name.split(":")[0]
            if kind == "flat":
                pred = np.zeros(len(h))
            elif kind == "damped":
                e = np.log(tindex.extrapolate(own, horizon, 12, 0.6, "damped"))
                pred = e.loc[h].to_numpy() - e.loc[cut - 1]
            elif kind == "lvl":
                r = Ridge(alpha=1.0).fit(M.loc[:cut - 1, cols].values, lown.loc[:cut - 1].values)
                pred = r.predict(M.loc[h, cols].values) - r.predict(M.loc[[cut - 1], cols].values)[0]
            else:
                dm, dl = M[cols].diff(12), lown.diff(12)
                r = Ridge(alpha=1.0).fit(dm.loc[12:cut - 1].values, dl.loc[12:cut - 1].values)
                # The forecast is chained, so each month adds its predicted
                # 12-month change to the index a year earlier. Past horizon 12
                # that anchor is itself a forecast, because the pre-cut fit stops
                # at cut - 1, so the chain is carried recursively rather than
                # read off the observed index. Months after the cut therefore
                # enter only through the macro series, which are published
                # without rents.
                step = dict(zip(h, r.predict(dm.loc[h].values)))
                chain: dict[int, float] = {}
                for t in h:
                    anchor = lown.get(t - 12, chain.get(t - 12))
                    chain[t] = anchor + step[t]
                pred = np.array([chain[t] for t in h]) - lown.loc[cut - 1]
            rows.append(dict(cut=cut, method=name,
                             err_pct=float(np.abs(np.exp(pred - true) - 1).mean() * 100)))
    return pd.DataFrame(rows).pivot(index="method", columns="cut", values="err_pct")[list(cuts)]


if __name__ == "__main__":
    import tindex

    df = pd.read_parquet("outputs/base.parquet")
    tr = df[df._split == "train"]
    res = backtest(tr)
    # Month 39 is the holdout, so it is reported but left out of the mean.
    res["rolling_mean"] = res[[27, 31, 35]].mean(axis=1)
    res = res.sort_values("rolling_mean").round(2)
    res.to_csv("outputs/macro_backtest.csv")
    print(res.to_string())

    li = np.log(tindex.fit_index(tr))
    coe = monthly()["MACRO_COE"].loc[:li.index.max()]
    print(f"\ncorr(log COE, log index): levels {np.corrcoef(coe, li)[0, 1]:.3f}, "
          f"12-month differences {coe.diff(12).corr(li.diff(12)):.3f}")
