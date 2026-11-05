"""Which registry configs make up each named ablation table.

The tables under `outputs/` were cut by hand out of a running log while the
experiments were being done, so their rows carry the display labels used at the
time (`B1 index=flat`, `D3 halflife=24`) rather than the registry keys. That
made them impossible for an outsider to connect back to anything runnable. This
module writes the mapping down, so `python src/run_ablation.py --group
index_forecast` re-scores exactly the configurations a table reports.

What a rerun reproduces is the comparison, not the original absolute numbers.
Every config inherits the defaults in `experiment.py` as they stand today, and
those were refined after the early batches were scored, so a rerun of a 2024
batch is that batch judged by the current pipeline. Reruns therefore go to
`outputs/ablation_<group>_rerun.csv` and never overwrite the shipped table,
which stays as the record of what was actually decided and when.

Four tables are only partly covered, because some variants they contain were
run before the registry settled and their configs were never added to it. They
are listed in `PARTIAL` with the count that is still regenerable.
"""
from __future__ import annotations

# Each entry maps a table under outputs/ to the registry configs it reports, in
# the order the shipped table lists them.
GROUPS: dict[str, list[str]] = {
    # Batch B: how the index is carried past the last training month.
    "index_forecast": [
        "idx_flat", "idx_damp_w6_d07", "idx_damp_w12_d06", "idx_damp_w18_d085",
        "idx_linear",
    ],
    # Batch H: the window crossed properly with the damping factor.
    "index_window_damping": [
        f"idx_w{w}_d{d}" for w in (9, 12, 18) for d in (0.6, 0.75, 0.85, 0.95)
    ] + ["idx_flat"],
    # Batch D: the target encoding and the switches around it. The batch ran
    # while the default te_mode was "loo_month". The default is "none" today,
    # and te_k and te_half_life are read only when encoding is on, so the
    # originally named configs would rerun as copies of `base`. The te_loo_
    # entries pin the mode, which is what makes this a comparison rather than
    # one experiment repeated, and under them the batch reproduces its shipped
    # numbers rather than only their ordering.
    "target_encoding": [
        "te_loo_ref", "te_loo_k4", "te_loo_k30", "te_loo_hl24", "te_loo_hl12",
        "te_loo_log", "te_none", "te_loo_nospatial", "te_expanding",
    ],
    # Batch E: what should carry location once the target encodings were dropped.
    "location_representation": [
        "loc_spatial", "loc_none", "loc_street_cat", "loc_street_block_cat",
        "loc_spatial_ks", "loc_sample_hl24", "loc_window36",
    ],
    # Batch Jg: the grain of the unit the ratio index is fitted on.
    "index_grain": [
        "J0_ref", "Jg_str_blk", "Jg_str_ft", "Jg_twn_ft", "Jg_ft", "Jg_fine",
    ],
    # Batch Js: one index per segment, at two shrinkage levels.
    "index_segments": [
        "Jf_blockinfo", "Js_flat_type_1.0", "Js_flat_type_0.5",
        "Js_region_1.0", "Js_region_0.5",
    ],
    # Batch Jte: own-block history, made as stale in training as at test time.
    "lagged_block_history": [
        "Jte_lag_all", "Jte_lag_all_age", "Jte_lag_blkft",
        "Jte_lag_blkft_age", "Jte_exp_blkft",
    ],
}

# Tables whose shipped rows include variants that predate the registry. The
# value is the note printed when the group is rerun.
PARTIAL: dict[str, str] = {
    "target_encoding": "9 of 9 rows, under te_loo_ names rather than the "
                       "table's, because the originals stopped varying anything "
                       "once the default mode became none. This group reproduces "
                       "its shipped numbers exactly, not just their ordering",
    "feature_groups": "15 of 16 rows; the wider POI set was never registered",
    "model_family": "5 of 8 rows; the no-detrend CatBoost and the two shorter "
                    "training windows were never registered",
    "time_features": "2 of 5 rows; this batch was superseded by target_encoding "
                     "before the registry settled",
    "retune": "0 of 8 rows; the hand-set LightGBM variants were never registered",
}

GROUPS["feature_groups"] = [
    "base", "no_mrt", "no_mrtplan", "no_mall", "no_prisch", "no_secsch",
    "no_lease", "no_size", "no_height", "no_coords", "no_finegeo",
    "no_monthofyear", "no_poi", "no_poi_no_coords", "with_horizon",
]
GROUPS["model_family"] = ["lgbm", "catboost", "xgboost", "ridge", "no_detrend"]
GROUPS["time_features"] = ["base", "no_spatial"]
GROUPS["retune"] = []


def resolve(group: str) -> tuple[list[str], str]:
    """Return the config names for a group and the path its rerun is written to."""
    if group not in GROUPS:
        raise KeyError(f"unknown group {group!r}; known: {', '.join(sorted(GROUPS))}")
    return GROUPS[group], f"outputs/ablation_{group}_rerun.csv"
