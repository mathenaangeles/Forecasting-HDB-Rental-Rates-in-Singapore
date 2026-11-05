"""Named experiment registry.

Every ablation reported in the paper is defined here, so a run is reproducible
as `python src/run_ablation.py <name> ...` locally or as one task of a SLURM
job array on a CPU partition.
"""
from __future__ import annotations
from experiment import Config

REGISTRY: dict[str, Config] = {}


def reg(c: Config) -> Config:
    REGISTRY[c.name] = c
    return c


# Ablation B varies how the rent index is carried past the last training month.
reg(Config(name="idx_flat", index_method="flat"))
reg(Config(name="idx_damp_w6_d07", index_window=6, index_damp=0.7))
reg(Config(name="idx_damp_w12_d06"))  # The chosen default.
reg(Config(name="idx_damp_w18_d085", index_window=18, index_damp=0.85))
reg(Config(name="idx_linear", index_method="linear", index_window=18))
reg(Config(name="no_detrend", detrend=False))

# Ablations C and D vary the target encoding and the switches around it.
reg(Config(name="base"))  # The default model.
reg(Config(name="te_none", te_mode="none"))
reg(Config(name="te_expanding", te_mode="expanding"))
# These unpinned legacy entries inherit te_mode. Its current default is
# "none", so smoothing and recency parameters are inactive and they reproduce
# `base`. Historical names are retained for the ablation log. The te_loo_* and
# te_exp_* entries below specify the mode explicitly for actual comparisons.
reg(Config(name="te_k4", te_k=4.0))
reg(Config(name="te_k30", te_k=30.0))
reg(Config(name="te_hl24", te_half_life=24.0))
reg(Config(name="te_hl12", te_half_life=12.0))
# The batch was run with loo_month as the default mode, which is what the
# te_loo_* entries restore. The te_exp_* entries are the same sweep under
# expanding, kept because the two modes rank the smoothing settings differently.
reg(Config(name="te_loo_ref", te_mode="loo_month"))
reg(Config(name="te_loo_k4", te_mode="loo_month", te_k=4.0))
reg(Config(name="te_loo_k30", te_mode="loo_month", te_k=30.0))
reg(Config(name="te_loo_hl24", te_mode="loo_month", te_half_life=24.0))
reg(Config(name="te_loo_hl12", te_mode="loo_month", te_half_life=12.0))
# D5 and D7 of the batch varied one switch each while the encoding stayed on,
# so they need the mode pinned as well to reproduce their rows.
reg(Config(name="te_loo_log", te_mode="loo_month", log_target=True))
reg(Config(name="te_loo_nospatial", te_mode="loo_month", spatial=False))
reg(Config(name="te_exp_k4", te_mode="expanding", te_k=4.0))
reg(Config(name="te_exp_k30", te_mode="expanding", te_k=30.0))
reg(Config(name="te_exp_hl24", te_mode="expanding", te_half_life=24.0))
reg(Config(name="te_exp_hl12", te_mode="expanding", te_half_life=12.0))
reg(Config(name="no_spatial", spatial=False))
reg(Config(name="with_horizon", drop=[]))
reg(Config(name="no_month_of_year", drop=["HORIZON_M", "MONTH_OF_YEAR"]))
reg(Config(name="log_target", log_target=True))

# Ablation E compares location representations after hierarchical target
# encoding increased validation error. Available history becomes stale over
# the forecast horizon, but the experiments do not isolate staleness as the
# mechanism of the encoding penalty.
reg(Config(name="loc_spatial", te_mode="none"))  # The chosen default.
reg(Config(name="loc_none", te_mode="none", spatial=False))
reg(Config(name="loc_street_cat", te_mode="none", extra_cat=["STREET"]))
reg(Config(name="loc_street_block_cat", te_mode="none",
           extra_cat=["STREET", "BLOCK"]))
reg(Config(name="loc_spatial_ks", te_mode="none", spatial_ks=(5, 15, 40, 90)))
reg(Config(name="loc_sample_hl24", te_mode="none", sample_half_life=24.0))
reg(Config(name="loc_window36", te_mode="none", train_window=36))

# Ablation F varies the model class at default hyper-parameters.
reg(Config(name="lgbm", model="lgbm"))
reg(Config(name="catboost", model="cat"))
reg(Config(name="xgboost", model="xgb"))
reg(Config(name="ridge", model="ridge"))


# Ablation I removes one feature group at a time. The groups must also be tested
# jointly, because the POI groups substitute for one another and each therefore
# looks worthless on its own.
FEATURE_GROUPS = {
    "mrt": ["MRT_NEAR1", "MRT_NEAR2", "MRT_CNT0.5", "MRT_CNT1.0"],
    "mrtplan": ["MRTPLAN_NEAR1"],
    "mall": ["MALL_NEAR1", "MALL_CNT1.0"],
    "prisch": ["PRISCH_NEAR1", "PRISCH_CNT1.0"],
    "secsch": ["SECSCH_NEAR1"],
    "lease": ["LEASE_AGE", "LEASE_REMAIN", "BLOCK_AGE"],
    "size": ["FLOOR_AREA_SQM", "N_ROOMS", "AREA_PER_ROOM", "AREA_VS_TYPE"],
    "height": ["MAX_FLOOR", "IS_TOP_BLOCK"],
    "coords": ["LATITUDE", "LONGITUDE", "DIST_CBD"],
    "finegeo": ["PLANNING_AREA", "REGION", "SUBZONE"],
    "monthofyear": ["MONTH_OF_YEAR"],
}
_POI = sum((FEATURE_GROUPS[k] for k in
            ("mrt", "mrtplan", "mall", "prisch", "secsch")), [])
for _g, _cols in FEATURE_GROUPS.items():
    reg(Config(name=f"no_{_g}", drop=["HORIZON_M"] + _cols))
reg(Config(name="no_poi", drop=["HORIZON_M"] + _POI))
reg(Config(name="no_poi_no_coords",
           drop=["HORIZON_M"] + _POI + FEATURE_GROUPS["coords"]))

# Ablation H disentangles the index window from the damping factor, which batch B
# had confounded, by crossing the two properly.
for _w in (9, 12, 18):
    for _d in (0.6, 0.75, 0.85, 0.95):
        reg(Config(name=f"idx_w{_w}_d{_d}", index_window=_w, index_damp=_d))


# Batch J runs on top of the tuned LightGBM, so that the reference is the model
# we actually ship. Within it, Jg_* varies the grain of the ratio index, Js_*
# segments the index, Jte_* revisits target encoding, Jf_* adds features, and
# Jc_* combines the survivors. Jc_ft05_blockinfo is the shipped configuration:
# one index per flat type, shrunk halfway to the global index, plus the
# block-composition features.
from tuned_params import BEST_LGBM  # noqa: E402
_T = dict(BEST_LGBM)
reg(Config(name="J0_ref", params=_T))  # The reference every J variant is scored against.
# The grain of the index unit matters because coarser units mix flats whose rents
# move differently.
reg(Config(name="Jg_str_blk", params=_T, index_grain=("STREET", "BLOCK")))
reg(Config(name="Jg_str_ft", params=_T, index_grain=("STREET", "FLAT_TYPE")))
reg(Config(name="Jg_twn_ft", params=_T, index_grain=("TOWN", "FLAT_TYPE")))
reg(Config(name="Jg_ft", params=_T, index_grain=("FLAT_TYPE",)))
reg(Config(name="Jg_fine", params=_T,
           index_grain=("STREET", "BLOCK", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM")))
# One index per segment asks whether small flats, or the city fringe, re-rated
# differently from the market as a whole.
for _by in ("FLAT_TYPE", "REGION"):
    for _s in (1.0, 0.5):
        reg(Config(name=f"Js_{_by.lower()}_{_s}", params=_T, index_by=_by, index_shrink=_s))
# Own-block history, made as stale during training as it is at test time.
reg(Config(name="Jte_lag_all", params=_T, te_mode="lagged"))
reg(Config(name="Jte_lag_all_age", params=_T, te_mode="lagged", te_age=True))
reg(Config(name="Jte_lag_blkft", params=_T, te_mode="lagged",
           te_levels=("BLK_FT", "STR_FT", "TWN_FT")))
reg(Config(name="Jte_lag_blkft_age", params=_T, te_mode="lagged", te_age=True,
           te_levels=("BLK_FT", "STR_FT", "TWN_FT")))
reg(Config(name="Jte_exp_blkft", params=_T, te_mode="expanding",
           te_levels=("BLK_FT", "STR_FT", "TWN_FT")))
# Block composition, taken from HDB Property Information.
BLOCK_INFO = ["BLK_UNITS", "BLK_RENTAL_SHARE", "BLK_SMALL_SHARE", "BLK_BIG_SHARE",
              "BLK_COMMERCIAL", "BLK_HAWKER", "BLK_MSCP"]
reg(Config(name="Jf_blockinfo", params=_T, extra_num=BLOCK_INFO))
reg(Config(name="Jc_ft05_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="FLAT_TYPE", index_shrink=0.5))
reg(Config(name="Jc_region05_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="REGION", index_shrink=0.5))
COE_CATS_COLS = ["MACRO_COE_A", "MACRO_COE_B", "MACRO_COE_C", "MACRO_COE_E"]
reg(Config(name="Jf_coecats", params=_T, extra_num=COE_CATS_COLS))
reg(Config(name="Jc_coecats_blockinfo", params=_T, extra_num=BLOCK_INFO + COE_CATS_COLS,
           index_by="FLAT_TYPE", index_shrink=0.5))
reg(Config(name="Jc_regionft05_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="REGION_FT", index_shrink=0.5))
reg(Config(name="Jc_ft03_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="FLAT_TYPE", index_shrink=0.3))
reg(Config(name="Jc_ft07_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="FLAT_TYPE", index_shrink=0.7))
reg(Config(name="Jc_ft09_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="FLAT_TYPE", index_shrink=0.9))
reg(Config(name="Jc_town05_blockinfo", params=_T, extra_num=BLOCK_INFO,
           index_by="TOWN", index_shrink=0.5))

PREM_COLS = ["PREMSCH_NEAR1", "PREMSCH_NEAR2", "PREMSCH_NEAR3",
             "PREMSCH_CNT0.5", "PREMSCH_CNT1.0", "PREMSCH_CNT2.0"]
reg(Config(name="Jc_premsch_blockinfo", params=_T, extra_num=PREM_COLS + BLOCK_INFO,
           index_by="FLAT_TYPE", index_shrink=0.5))

def get(name: str) -> Config:
    if name not in REGISTRY:
        raise SystemExit(f"unknown config {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
