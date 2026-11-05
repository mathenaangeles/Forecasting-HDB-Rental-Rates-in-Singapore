"""Data loading, cleaning and feature engineering for HDB rental prediction.

* The combination of STREET, BLOCK and FLAT_TYPE determines FLOOR_AREA_SQM,
  FLAT_MODEL and LEASE_COMMENCE_DATE. The EDA confirmed this relationship.
* FEE and FURNISHED are constant in both datasets, so we drop them.
* "4 room" and "4-room" mean the same thing. We normalize this difference
  from the 2021 source file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import macro
from sklearn.neighbors import BallTree

EARTH_R_KM = 6371.0
# Raffles Place, taken as the CBD centroid.
CBD = (1.28384, 103.85154)
DATA = "data"


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


def load_raw(data_dir: str = DATA):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    tr["_split"], te["_split"] = "train", "test"
    te["MONTHLY_RENT"] = np.nan
    df = pd.concat([tr, te], ignore_index=True)

    for c in ("TOWN", "BLOCK", "STREET", "FLAT_MODEL", "FLAT_TYPE"):
        df[c] = _norm(df[c])
    df["FLAT_TYPE"] = df["FLAT_TYPE"].str.replace(" ", "-", regex=False)

    d = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="%Y-%m")
    df["TM"] = (d.dt.year - 2021) * 12 + d.dt.month - 1  # Month index, 0 = 2021-01.
    df["MONTH_OF_YEAR"] = d.dt.month
    df["DATE"] = d
    return df.drop(columns=["FEE", "FURNISHED"])


def add_block_geo(df: pd.DataFrame, data_dir: str = DATA) -> pd.DataFrame:
    """Join the block gazetteer (100% coverage after normalisation)."""
    blk = pd.read_csv(f"{data_dir}/auxiliary/sg-hdb-block.csv")
    for c in ("TOWN", "BLOCK", "STREET", "SUBZONE", "PLANNING_AREA", "REGION"):
        blk[c] = _norm(blk[c])
    blk = blk.drop_duplicates(["TOWN", "BLOCK", "STREET"])
    cols = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
            "SUBZONE", "PLANNING_AREA", "REGION"]
    out = df.merge(blk[["TOWN", "BLOCK", "STREET"] + cols],
                   on=["TOWN", "BLOCK", "STREET"], how="left")
    assert len(out) == len(df)
    return out


_STREET_ABBR = {
    "AVE": "avenue", "DR": "drive", "CRES": "crescent", "ST": "street",
    "GDN": "garden", "GDNS": "gardens", "RD": "road", "STH": "south",
    "NTH": "north", "LOR": "lorong", "UPP": "upper", "CTRL": "central",
    "PK": "park", "HTS": "heights", "TER": "terrace", "PL": "place",
    "CL": "close", "BT": "bukit", "JLN": "jalan", "KG": "kampong",
    "TG": "tanjong", "C'WEALTH": "commonwealth", "MKT": "market",
    "SQ": "square", "CTR": "centre", "CT": "court", "PKWY": "parkway",
    "VW": "view", "MT": "mount", "ST.": "saint",
}


def add_property_info(df: pd.DataFrame, data_dir: str = DATA) -> pd.DataFrame:
    """Block composition from data.gov.sg HDB Property Information (99.6% match).

    This file has no rent data. It describes each block's size, flat mix,
    public rental share and facilities. Street abbreviations are expanded
    before the join.
    """
    p = pd.read_csv(f"{data_dir}/HDBPropertyInformation.csv")
    p["STREET"] = p["street"].map(
        lambda s: " ".join(_STREET_ABBR.get(w, w.lower()) for w in s.upper().split()))
    p["BLOCK"] = p["blk_no"].astype(str).str.lower()
    tot = p["total_dwelling_units"].clip(lower=1)
    rental = p[["1room_rental", "2room_rental", "3room_rental", "other_room_rental"]].sum(1)
    small = p[["1room_sold", "2room_sold", "studio_apartment_sold"]].sum(1)
    big = p[["5room_sold", "exec_sold", "multigen_sold"]].sum(1)
    feats = pd.DataFrame({
        "STREET": p["STREET"], "BLOCK": p["BLOCK"],
        "BLK_UNITS": p["total_dwelling_units"],
        "BLK_RENTAL_SHARE": rental / tot,
        "BLK_SMALL_SHARE": small / tot,
        "BLK_BIG_SHARE": big / tot,
        "BLK_COMMERCIAL": (p["commercial"] == "Y").astype(int),
        "BLK_HAWKER": (p["market_hawker"] == "Y").astype(int),
        "BLK_MSCP": (p["multistorey_carpark"] == "Y").astype(int),
    }).drop_duplicates(["STREET", "BLOCK"])
    out = df.merge(feats, on=["STREET", "BLOCK"], how="left")
    assert len(out) == len(df)
    return out


def _tree(lat, lon):
    return BallTree(np.radians(np.column_stack([lat, lon])), metric="haversine")


def _dist_and_count(df, ref_lat, ref_lon, radii=(0.5, 1.0, 2.0), k=3):
    """Nearest-k distances (km) and counts within radii for one POI table."""
    tree = _tree(np.asarray(ref_lat), np.asarray(ref_lon))
    q = np.radians(np.column_stack([df["LATITUDE"], df["LONGITUDE"]]))
    kk = min(k, len(ref_lat))
    d, _ = tree.query(q, k=kk)
    d = d * EARTH_R_KM
    res = {f"NEAR{i+1}": d[:, i] for i in range(kk)}
    for r in radii:
        res[f"CNT{r}"] = tree.query_radius(q, r=r / EARTH_R_KM, count_only=True)
    return res


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def add_poi_features(df: pd.DataFrame, data_dir: str = DATA) -> pd.DataFrame:
    df = df.copy()
    mrt = pd.read_csv(f"{data_dir}/auxiliary/sg-mrt-stations.csv")
    mall = pd.read_csv(f"{data_dir}/auxiliary/sg-shopping-malls.csv")
    sch = pd.read_csv(f"{data_dir}/auxiliary/sg-schools.csv")

    PREMIUM_TYPES = ["independent school", "specialised independent school", "specialised school"]

    sources = {
        # Open and planned stations are kept apart, because a planned line is a
        # future premium and behaves differently from an operating one.
        "MRT": mrt[mrt.STATUS == "open"],
        "MRTPLAN": mrt[mrt.STATUS == "planned"],
        "MALL": mall,
        "PRISCH": sch[sch.MAINLEVEL_CODE == "primary"],
        "SECSCH": sch[sch.MAINLEVEL_CODE.str.startswith("secondary")],
        "PREMSCH": sch[sch["TYPE_CODE"].isin(PREMIUM_TYPES)],

    }
    for name, ref in sources.items():
        feats = _dist_and_count(df, ref["LATITUDE"].values, ref["LONGITUDE"].values)
        for suffix, vals in feats.items():
            df[f"{name}_{suffix}"] = vals

    df["DIST_CBD"] = haversine_km(df["LATITUDE"], df["LONGITUDE"], *CBD)
    return df


def add_flat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LEASE_AGE"] = df["TM"] / 12 + 2021 - df["LEASE_COMMENCE_DATE"]
    df["LEASE_REMAIN"] = 99 - df["LEASE_AGE"]
    df["BLOCK_AGE"] = df["TM"] / 12 + 2021 - df["YEAR_COMPLETED"]
    # Room counts run from 1 to 5, with executive flats treated as 6.
    n = df["FLAT_TYPE"].str.extract(r"^(\d)")[0].astype(float)
    df["N_ROOMS"] = n.fillna(6.0)
    df["AREA_PER_ROOM"] = df["FLOOR_AREA_SQM"] / df["N_ROOMS"]
    # Compare each flat's area with the median for its flat type.
    df["AREA_VS_TYPE"] = df["FLOOR_AREA_SQM"] / df.groupby("FLAT_TYPE")["FLOOR_AREA_SQM"].transform("median")
    df["IS_TOP_BLOCK"] = (df["MAX_FLOOR"] >= 20).astype(int)
    df["REGION_FT"] = df["REGION"].astype(str) + "_" + df["FLAT_TYPE"].astype(str)
    
    return df


def build(data_dir: str = DATA) -> pd.DataFrame:
    df = load_raw(data_dir)
    df = add_block_geo(df, data_dir)
    df = add_poi_features(df, data_dir)
    df = add_flat_features(df)
    df = add_property_info(df, data_dir)
    m = macro.monthly(data_dir)
    df = df.merge(m, left_on="TM", right_index=True, how="left")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    d = build()
    d.to_parquet("outputs/base.parquet")
    print(f"built {d.shape} and saved outputs/base.parquet")
