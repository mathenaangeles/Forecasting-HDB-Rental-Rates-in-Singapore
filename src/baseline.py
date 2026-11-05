"""The Section IV baselines, rebuilt so their reported numbers are measurable.

Both baselines predate the rest of this repository and were first run in the
notebook. They are kept here because the report quotes their holdout scores and
the permutation importance of the boosting one, and a quoted number should be
reproducible from a script rather than from a console session. The feature
engineering, the chronological split and the search below are ported from
commit 45e2524 of notebook_baseline.ipynb and are deliberately unchanged, so
this script reproduces the original baselines and not an improved version of
them.

The twelve features are the four attributes the linear model uses, the flat
model, the haversine distance to the nearest MRT station, mall and primary
school, the mean COE premium and share price of the month, a month index
counting from January 2021 and the calendar month.

    python src/baseline.py      # writes outputs/permutation_importance.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = "outputs/permutation_importance.csv"
EARTH_RADIUS_KM = 6371
SEED = 42
HOLDOUT_SHARE = 0.15

NUM_FEATS = ["FLOOR_AREA_SQM", "LEASE_AGE_YR", "MTH_SINCE_START", "MTH_OF_YR",
             "DIST_MRT_KM", "DIST_MALL_KM", "DIST_SCH_KM", "COE_PREMIUM_AVG",
             "STOCK_CLOSE_AVG"]
CAT_FEATS = ["TOWN", "FLAT_TYPE", "FLAT_MODEL"]
FEAT_COLS = NUM_FEATS + CAT_FEATS
# The linear model takes only these four, with the two categoricals one-hot
# encoded, because least squares cannot split on a category. It was validated
# on the months after the 85th percentile of approval dates, which is a
# slightly different window from the boosting holdout below, so the report
# states that the two scores are not comparable with each other.
LINEAR_NUM = ["FLOOR_AREA_SQM", "LEASE_AGE_YR"]
LINEAR_CAT = ["TOWN", "FLAT_TYPE"]
LINEAR_QUANTILE = 0.85

# The names the report prints, in place of the column names used here.
LABELS = {"TOWN": "Town", "FLAT_TYPE": "Flat type", "FLOOR_AREA_SQM": "Floor area",
          "LEASE_AGE_YR": "Lease age", "FLAT_MODEL": "Flat model",
          "DIST_MRT_KM": "Distance to nearest MRT station",
          "DIST_MALL_KM": "Distance to nearest mall",
          "DIST_SCH_KM": "Distance to nearest primary school",
          "MTH_OF_YR": "Calendar month", "MTH_SINCE_START": "Month index",
          "STOCK_CLOSE_AVG": "Mean stock closing price",
          "COE_PREMIUM_AVG": "Mean COE premium"}

MTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

GRID = {"learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_leaf_nodes": [15, 31, 63],
        "max_iter": [100, 200, 300],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0]}


def nearest_dist_km(lat, lon, ref_lat, ref_lon):
    """Haversine distance to the nearest reference point, in kilometres.

    Rows whose block did not match the block table have no coordinates and keep
    a missing distance, which the boosting model routes down a learned branch.
    """
    valid = lat.notna() & lon.notna()
    out = pd.Series(np.nan, index=lat.index)
    if valid.sum() == 0:
        return out
    tree = BallTree(np.radians(np.column_stack([ref_lat, ref_lon])), metric="haversine")
    dist, _ = tree.query(np.radians(np.column_stack([lat[valid], lon[valid]])), k=1)
    out.loc[valid] = dist[:, 0] * EARTH_RADIUS_KM
    return out


def build(data_dir="data"):
    """The twelve-feature training frame, in the order the notebook built it."""
    df = pd.read_csv(f"{data_dir}/train.csv").drop(columns=["FEE", "FURNISHED"])
    df["FLAT_TYPE"] = df["FLAT_TYPE"].str.replace(" ", "-", regex=False)
    df["RENT_APPROVAL_DATE"] = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="%Y-%m")
    df["MTH_SINCE_START"] = ((df.RENT_APPROVAL_DATE.dt.year - 2021) * 12
                             + df.RENT_APPROVAL_DATE.dt.month)
    df["MTH_OF_YR"] = df.RENT_APPROVAL_DATE.dt.month
    df["LEASE_AGE_YR"] = df.RENT_APPROVAL_DATE.dt.year - df.LEASE_COMMENCE_DATE

    block = pd.read_csv(f"{data_dir}/auxiliary/sg-hdb-block.csv")
    keys = ["TOWN", "BLOCK", "STREET"]
    df = df.merge(block[keys + ["LATITUDE", "LONGITUDE"]], on=keys, how="left")

    mrt = pd.read_csv(f"{data_dir}/auxiliary/sg-mrt-stations.csv")
    mall = pd.read_csv(f"{data_dir}/auxiliary/sg-shopping-malls.csv")
    school = pd.read_csv(f"{data_dir}/auxiliary/sg-schools.csv")
    # The project description asks for primary schools, so secondary and
    # tertiary institutions are dropped before the nearest one is found.
    primary = school[school.MAINLEVEL_CODE == "primary"]
    df["DIST_MRT_KM"] = nearest_dist_km(df.LATITUDE, df.LONGITUDE, mrt.LATITUDE, mrt.LONGITUDE)
    df["DIST_MALL_KM"] = nearest_dist_km(df.LATITUDE, df.LONGITUDE, mall.LATITUDE, mall.LONGITUDE)
    df["DIST_SCH_KM"] = nearest_dist_km(df.LATITUDE, df.LONGITUDE, primary.LATITUDE, primary.LONGITUDE)

    coe = pd.read_csv(f"{data_dir}/auxiliary/sg-coe-prices.csv")
    coe["QUOTA_PREMIUM"] = coe.QUOTA_PREMIUM.str.replace(r"[\$,]", "", regex=True).astype(float)
    coe["MTH_SINCE_START"] = (coe.YEAR - 2021) * 12 + coe.MONTH.map(MTH_NUM)
    monthly = coe.groupby("MTH_SINCE_START").QUOTA_PREMIUM.mean().reset_index(name="COE_PREMIUM_AVG")
    df = df.merge(monthly, on="MTH_SINCE_START", how="left")

    stock = pd.read_csv(f"{data_dir}/auxiliary/sg-stock-prices.csv")
    stock["DATE"] = pd.to_datetime(stock.DATE)
    stock["MTH_SINCE_START"] = (stock.DATE.dt.year - 2021) * 12 + stock.DATE.dt.month
    closes = (stock.dropna(subset=["CLOSE"]).groupby("MTH_SINCE_START")
              .CLOSE.mean().reset_index(name="STOCK_CLOSE_AVG"))
    df = df.merge(closes, on="MTH_SINCE_START", how="left")

    for col in CAT_FEATS:
        df[col] = pd.Categorical(df[col])
    return df.iloc[df.RENT_APPROVAL_DATE.argsort()].reset_index(drop=True)


def split(df):
    """The chronological holdout, being the latest 15 percent of rows by date.

    The competition test set is itself a future-only split of the training
    file, so a random split would place later rows in the training half and
    report a score the test set cannot match.
    """
    cut = int(len(df) * (1 - HOLDOUT_SHARE))
    return df.iloc[:cut], df.iloc[cut:]


def least_squares(df):
    """The linear baseline, scored on the months after its date cutoff."""
    cutoff = df.RENT_APPROVAL_DATE.quantile(LINEAR_QUANTILE)
    fit = df[df.RENT_APPROVAL_DATE <= cutoff]
    val = df[df.RENT_APPROVAL_DATE > cutoff]
    cols = LINEAR_NUM + LINEAR_CAT
    model = Pipeline([
        ("encode", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                                       LINEAR_CAT)], remainder="passthrough")),
        ("fit", LinearRegression())])
    model.fit(fit[cols], fit.MONTHLY_RENT)
    rmse = root_mean_squared_error(val.MONTHLY_RENT, model.predict(val[cols]))
    print(f"least squares cutoff {cutoff.date()} | fitting rows: {len(fit)} | "
          f"validation rows: {len(val)}")
    print(f"least squares validation RMSE: {rmse:.2f}")


def main(data_dir="data"):
    df = build(data_dir)
    least_squares(df)

    fit, hold = split(df)
    y_fit, y_hold = fit.MONTHLY_RENT, hold.MONTHLY_RENT
    print(f"boosting fitting rows: {len(fit)} | holdout rows: {len(hold)}")

    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED),
        GRID, n_iter=8, cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1)
    search.fit(fit[FEAT_COLS], y_fit)
    model = search.best_estimator_
    print(f"boosting params: {search.best_params_}")
    print(f"boosting holdout RMSE: "
          f"{root_mean_squared_error(y_hold, model.predict(hold[FEAT_COLS])):.2f}")

    # Permutation importance is the drop in the coefficient of determination
    # when one column is shuffled among the holdout rows, so it reports what
    # the fitted model actually uses on months it was not fitted on.
    perm = permutation_importance(model, hold[FEAT_COLS], y_hold, n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    res = (pd.DataFrame({"feature": [LABELS[c] for c in FEAT_COLS],
                         "column": FEAT_COLS,
                         "importance": perm.importances_mean,
                         "sd": perm.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    res.to_csv(OUT, index=False)
    print(res.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
