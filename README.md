# HDB Rental Prediction (CS5228)

The task is to predict the monthly rent of a Singapore HDB flat from its attributes, location and transaction date. Predictions are scored by RMSE over the 16 months after the last labelled month.

## Getting Started

Everything below runs on CPU. Install the pinned environment and build the feature table once. Every other script reads it back from `outputs/base.parquet`, and it takes about five seconds to build.

```bash
uv sync --all-groups
uv run python src/data.py          # writes outputs/base.parquet
```

Run the steps in the order given. `src/data.py` must come first, because everything else reads `outputs/base.parquet`. Within step 3 the scripts are independent and may run in any order, but step 4 is not independent of step 3: `h2h.py`, `confirm.py` and `members.py` write the cached fold predictions that `blend_weights.py` and `blend_members.py` read, so the two blending scripts cannot run until those three have. Step 5 must run last, because the figures read the CSVs the earlier steps write rather than refitting anything, and it needs `ladder.py`, `error_decomp.py`, `residuals.py`, `h2h.py`, `blend_weights.py` and `blend_members.py` to have run.

Three scripts need arguments and do nothing useful without them. `members.py` fits only the members you name, `confirm.py` scores only the variants you name alongside its built-in reference, and `tune.py` exits with an error unless given `--index` or `--range`. The commands below pass what the reported results were produced with.

```bash
# 1. The shipped submission, under the inner tuning protocol it was scored
#    with. Add --inner month for the repaired protocol, which writes
#    blend_lgbm_cat_blockhist_inner_month.csv rather than over the scored file.
uv run python src/predict.py blend_lgbm_cat_blockhist

# 2. Every configuration in the registry (90 named entries), appended to
#    outputs/ablation_results.csv. This does NOT rewrite the named ablation
#    tables; see "Ablation Results" below for what does.
uv run python src/run_ablation.py --all
uv run python src/run_ablation.py --list-groups   # the eleven named tables

# 3. The experiments that are not registry entries. Any order.
uv run python src/eda.py           # every statistic Section II of the report quotes
uv run python src/baseline.py      # the two Section IV baselines and their importances
uv run python src/ladder.py        # eleven untuned model classes, three treatments of time
uv run python src/altreps.py       # k-means location clusters and PCA
uv run python src/index_horizon.py # the market level each index forecast implies
uv run python src/additive_check.py # matched additive and multiplicative global indices
uv run python src/compare_submissions.py  # cross-section against level, on two scored files
uv run python src/macro.py         # COE and SGX as forecasters of the index
uv run python src/blockhist.py     # the notebook with and without block rent history
uv run python src/error_decomp.py  # noise, market forecast and remainder
uv run python src/residuals.py     # residuals by horizon and slice, and twin-group noise estimates
uv run python src/mrt_check.py     # MRT columns rebuilt from one row per station
uv run python src/tune.py --range 0 40    # the hyper-parameter search. Slow, and the
                                   # winners are already frozen in tuned_params.py.
                                   # Bare `tune.py` is an error: it needs a range.
                                   # This regenerates 40 of the 50 saved draws.
                                   # Indices 100-109 predate the current sampler
                                   # and no longer regenerate from their seed.

# 4. The ensemble. These three write the cached fold predictions
#    (outputs/h2h_preds*.pkl, confirm_*.pkl, member_*.pkl) ...
uv run python src/h2h.py           # our splits against the notebook's
uv run python src/confirm.py Js_flat_type_0.5 Jf_blockinfo Jc_ft05_blockinfo
uv run python src/members.py ridge nb_plus
#    ... that these two read back. They refit nothing and fail without the caches.
uv run python src/blend_weights.py # the weight grid over the three libraries
uv run python src/blend_members.py # candidate extra members for the ensemble

#    h2h.py, members.py, blend_weights.py, and blend_members.py take --inner
#    rows or --inner month. The default is rows, which reproduces the quoted
#    tables. confirm.py has no inner search and takes configuration names only.
#    The repaired protocol writes its own caches and its own tables beside
#    them, never over them, so the two are never mixed inside one table.
uv run python src/h2h.py --inner month
uv run python src/members.py nb_plus --inner month
uv run python src/blend_weights.py --inner month
uv run python src/blend_members.py --inner month

# 5. The report figures, into figures/.
uv run python src/figs_eda.py
uv run python src/figs_ablation_summary.py
uv run python src/figs_ablation_detail.py
uv run python src/figs_ladder.py
uv run python src/figs_residuals.py
```

Step 2 takes a few hours on one machine. `scripts/slurm_ablation.sbatch` runs it as a job array instead, one task per registry name, and `scripts/slurm_tune.sbatch` does the same for `src/tune.py`. Submit the ablation array with the bound taken from the registry rather than from the directive in the file, so it cannot go stale as configs are added:

```bash
sbatch --array=0-$(($(uv run python src/run_ablation.py --count) - 1)) scripts/slurm_ablation.sbatch
uv run python src/run_ablation.py --merge        # once the array has finished

sbatch scripts/slurm_tune.sbatch                 # the 40 draws the report quotes
uv run python src/tune.py --merge
uv run python src/tune.py --write-best
```

Array tasks write one file each under `outputs/parts/` and `--merge` folds them into the shared table. They have to, because appending means reading the shared file and writing it back, and two tasks doing that at once lose a row. An index outside the registry now fails loudly instead of scoring the wrong config. The local sequential route in step 2 needs none of this and is the simpler option if the cluster is not already set up.

### Submissions

Every file in `submissions/` was submitted except the linear baseline, which is kept as the starting point the rest are measured against. The name says what the approach was. The first two are written by `src/predict.py`, with `--shrink 0.7` producing the first file and the default shrinkage 0.5 producing the second.

| File | Approach | Public RMSE |
|---|---|---|
| `blend_lgbm_cat_blockhist_shrink07.csv` | The shipped blend with index shrinkage 0.7 instead of 0.5 | **430.76** |
| `blend_lgbm_cat_blockhist.csv` | LightGBM + CatBoost on the detrended target, plus the block-history member. This is the configuration the report describes | 431.01 |
| `histgb_blockhist.csv` | The team's HistGB notebook with block rent history added | 436.05 |
| `histgb_geo_macro.csv` | The same notebook with geography and macro features, before block rent history | 458.01 |
| `ridge_raw_rent_optuna.csv` | A separate pipeline of ours: an Optuna-tuned ridge regression on raw rent over 71 features, with no index. Carries the level up 0.80% per month against our 0.02% | 505.60 |
| `baseline_linear_regression.csv` | Ordinary least squares on floor area, lease age, town and flat type | not scored |

Shrinkage 0.7 has the lower observed public RMSE by 0.25. The shrinkage grid spans 0.14 RMSE in rolling validation across 0.3 to 0.9. The statistical uncertainty of the public difference cannot be estimated from submission files alone because the hidden labels and paired prediction errors are unavailable.

`ridge_raw_rent_optuna.csv` is the most informative of the rest, but read it carefully. It ranks flats much as the shipped blend does, correlating 0.973 with it within a test month, and it ends 15% above us in the last test month because it carries the level up far faster. It is a different model class as well as a different level path, so the 75-point gap cannot be charged to the market-level projection alone, and it is a confounded comparison rather than an ablation of the index forecast. The clean version of that comparison is `src/ladder.py`, which holds everything but the treatment of time fixed. `src/compare_submissions.py` writes the per-month comparison to `outputs/submission_comparison.csv`.

### What Each Output File Holds

| File | Contents |
|---|---|
| `ablation_index_forecast.csv` | Flat, damped and linear index extrapolations |
| `ablation_index_window_damping.csv` | Slope window crossed with the damping factor |
| `ablation_index_grain.csv` | The unit the index is fitted on |
| `ablation_index_segments.csv` | One index per flat type or region, and the shrinkage towards the global one |
| `ablation_time_features.csv` | The horizon feature, the calendar month and the log target |
| `ablation_target_encoding.csv` | Prior strength and half-life for the hierarchical encodings |
| `ablation_lagged_block_history.csv` | Own-block history made as stale in training as it is at prediction time |
| `ablation_location_representation.csv` | Raw street and block categories against the spatial prior |
| `ablation_altreps.csv` | K-means location clusters and PCA of the numeric block |
| `eda_summary.csv` | Every exploratory statistic quoted in Section II, one row per number |
| `index_test_horizon.csv` | The test-horizon market level implied by each index forecast, and the rent shift between them |
| `index_by_flat_type.csv` | How far the market level rose over the labelled period, one index per flat type |
| `submission_comparison.csv` | Mean predicted level per test month for two scored submissions, and their within-month correlation |
| `ablation_feature_groups.csv` | Each feature group removed in turn, and the POI groups removed together |
| `ablation_model_family.csv` | LightGBM, CatBoost, XGBoost and ridge at default settings |
| `ablation_blend_weights.csv` | The weight grid over the three boosting libraries. A `_repaired` file holds the same grid under the repaired inner protocol |
| `ablation_blend_members.csv` | Candidate extra members, with the same `_repaired` convention |
| `ablation_confirmation_cut.csv` | Every survivor re-scored at the month-39 origin |
| `ablation_retune.csv` | The second hyper-parameter search |
| `ablation_head_to_head.csv` | Our splits against the notebook's, on identical rows. The `_repaired` file is the same comparison with the member's inner tuning fixed |
| `ablation_blockhist.csv` | The notebook with and without block rent history |
| `ablations_all.csv`, `ablation_results.csv` | Every registry run, pooled |
| `ladder_metrics.csv`, `ladder_by_horizon.csv` | The eleven untuned classes |
| `error_budget.csv` | Estimated conditional noise, oracle-index improvement, and algebraic remainder shares on repeated-predictor rows |
| `rmse_floor.csv`, `feature_floor.csv` | Twin-group noise estimates and illustrative all-row scales, using competition columns and all engineered columns; these do not establish hidden-test error floors |
| `residual_slices.csv`, `residual_by_horizon.csv`, `residual_tilt.csv` | Residuals by slice and horizon, and the horizon tilt they motivate |
| `macro_backtest.csv` | Each macro series as a forecaster of the index. Every forecaster is fitted on the index estimated from months before its own cut; only the realized index they are scored against uses all labelled months |
| `mrt_station_check.csv` | The MRT columns rebuilt from one row per operating station, against the supplied table as joined |
| `tuning.csv` | Every hyper-parameter draw |

`outputs/*.parquet` is gitignored because it is derived; `src/data.py` and `src/ladder.py` rebuild theirs. `blend_grid.json` and `error_slices.csv` are intermediates read by the figure scripts.

## Method

Rent is factored into two parts, `RENT_it = BASE_i * INDEX_t`.

`INDEX_t` is a mix-adjusted monthly market index, fitted by iterative two-way (unit by month) ratio updates on rent in levels and forecast 16 months ahead with a short-window damped trend. `BASE_i` is what remains once the market level is divided out, and that is what the trees learn from flat attributes, coordinates and POI geography.

Rents rose 54% over the training window, and 99.2% of test rows follow the last labelled month. Tree splits on a month feature cannot extend a trend beyond the training range when other inputs are held fixed. The ablations below quantify the prediction error associated with omitting the index in the evaluated forecast windows.

Two refinements sit on top of that. The index is estimated per flat type and shrunk halfway back towards the global index, because the flat types did not re-rate in proportion: 1.46x for 2-room against 1.61x for executive over 2021-25. Block composition from HDB Property Information (unit count, flat-type mix, rental share, whether the ground floor is commercial) then describes what a block is rather than where it is, which the POI distances cannot proxy.

### Categorical Features

Six columns are categorical: `TOWN`, `FLAT_TYPE`, `FLAT_MODEL`, `PLANNING_AREA`, `REGION` and `SUBZONE`. They are never one-hot encoded globally, because how they are represented depends on what the model can consume, and `src/experiment.py` branches on that in `make_model`.

| Model | Handling |
|---|---|
| LightGBM, CatBoost | Passed as pandas `category` columns and split on natively, so the model chooses its own grouping of levels |
| XGBoost | Same, via `enable_categorical=True` with the histogram tree method |
| Decision tree, random forest | `OrdinalEncoder` integer codes, with unseen levels mapped to `-1`. A tree only needs an ordering to split on, and the arbitrary order costs little once the tree is deep enough to isolate single levels |
| Ridge, k-nearest neighbours, MLP | `OneHotEncoder(handle_unknown="ignore", min_frequency=30)`, with numeric columns median-imputed and standardised in the same `ColumnTransformer`. Rare levels are pooled so that the one-hot block stays small, and standardising matters because these three are distance- or gradient-based |

`FLAT_TYPE` also plays a second role that has nothing to do with the feature matrix: the market index is estimated separately for each flat type and shrunk halfway back to the global index, because the flat types did not re-rate in proportion.

Target encoding was tried as an alternative for the fine-grained location columns and made things worse, which `outputs/ablation_target_encoding.csv` records.

## Residual Error and Conditional Noise Estimates

The following analyses describe the evaluated models and validation periods. Noise estimates are restricted to rows with repeated predictors and do not establish the minimum achievable RMSE on other rows or the hidden test set.

**Repeated-predictor rows permit a conditional noise estimate.** A deterministic model assigns the same prediction to records with identical predictors. Their group mean minimizes the observed within-group squared error. Such groups contain 26.8% of training rows, with an observed within-group RMSE of S$343. A degrees-of-freedom correction gives a pooled standard deviation estimate of S$464. Interpreting this as conditional noise requires independent deviations around a common group mean. Multiplying the estimated variance by the twin-row share and assuming zero error on other rows gives an illustrative all-row scale of S$240. The corresponding scales are S$249 for 2024 onward and S$230 for January to March 2025. Test predictors give a twin-row share of 27.9%, implying scales of S$245 using the full-training noise estimate and S$234 using the 2025 estimate. These test calculations additionally assume that the estimated conditional variance transfers to the test period. None is a hidden-test error bound. `src/residuals.py` writes the calculations to `outputs/rmse_floor.csv`.

The evaluated LightGBM configuration has RMSE S$544.9 on twin rows and S$558.4 on other rows at cut 27, with the same ordering at the other cuts. These results describe that model's errors; they do not identify the noise variance or achievable RMSE on rows without twins.

**The current engineered features preserve the repeated-predictor groups.** Grouping on all 78 base-table features, including coordinates, POI distances, block composition, and macro series, gives the same 26.8% twin-row share and S$464.02 corrected noise estimate as grouping on the competition predictors. The joined features vary by block or month and do not distinguish observations within a block, flat type, and month. Additional unit-level attributes, such as storey or condition, could distinguish these observations, but are absent from the supplied data. This finding concerns the features evaluated here. `src/residuals.py` writes the comparison to `outputs/feature_floor.csv`.

**Model differences are smaller after index adjustment.** `src/ladder.py` compares eleven untuned model classes on the same simplified setup and splits. With index adjustment, rolling RMSE ranges from S$554.6 for the town-by-flat-type median lookup to S$522.3 for CatBoost, a span of S$32.2. The 20-nearest-neighbours model scores S$523.8, compared with S$523.4 for LightGBM. Bagged trees score S$528.3, 2.2 above the random forest, and AdaBoost scores S$535.9. These configurations do not isolate the causes of the differences. Without index adjustment or a month counter, the model classes span S$361.4, from S$603.7 to S$965.0, and every model worsens by at least S$75 relative to its index-adjusted version. Temporal target adjustment therefore has a larger measured effect than model choice within this comparison.

**Estimated noise accounts for most error on twin rows.** `src/error_decomp.py` compares the conditional noise estimate with the squared error of one LightGBM configuration on repeated-predictor validation rows. These rows comprise 27.6% of rolling-validation observations. The pooled noise estimate accounts for 98.8% of their model MSE. Across the four individual cuts, estimated shares range from 97.9% to 100.4%; values above 100% reflect estimation uncertainty. Replacing the forecast index with an oracle fitted using validation rents reduces all-row MSE by 0.6% to 1.6%, while the improvement on twin rows ranges from 0.3% to 2.2%. The oracle changes the transformed training target and uses future labels. Its improvement is a diagnostic, not a formal bound on gains from better forecasting. The noise estimate also depends on which attributes are observed and does not quantify error on rows without twins.

**Slice-level RMSE is comparable to estimated within-group dispersion.** `src/residuals.py` compares each town's and flat type's RMSE with the noise estimate from its repeated-predictor rows. Among slices with an available estimate, RMSE exceeds the estimated noise standard deviation by no more than 11%. Bukit Timah has RMSE S$729 and estimated noise standard deviation S$746; executive flats have corresponding values S$624 and S$601. The slice RMSE uses all rows, whereas the noise estimate uses twin rows. The comparison therefore depends on how representative those twin rows are and cannot rule out recoverable error within a slice.

**A horizon correction does not transfer to the confirmation cut.** Pooled rolling-fold bias decreases from S$13 at horizons one to four months to S$-33 at horizons thirteen to sixteen. Within individual folds, fold 31 has positive bias and fold 27 has negative bias at every horizon group. This pattern indicates sensitivity to the forecast origin as well as the horizon. A horizon-specific multiplier fitted on the rolling folds improves their RMSE by 0.33 but worsens confirmation-cut RMSE by 0.79, moving bias from S$-40.1 to S$-49.9. This correction was therefore rejected.

The tested model, feature, and calibration changes gave limited additional gains under these validation windows. The analyses do not establish an optimal model or a general error floor. Additional unit-level attributes and validation across different market regimes remain potential improvements.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/data.py` | Loading, cleaning, geo join, POI distances and counts |
| `src/tindex.py` | The two-way ratio rent index and its forecast |
| `src/macro.py` | Monthly COE and SGX series, and the backtest of them as index forecasters |
| `src/encode.py` | Hierarchical and spatial target encoding of detrended rent |
| `src/experiment.py` | `Config`, the model zoo, rolling-origin evaluation |
| `src/configs.py` | Every experiment we report, by name |
| `src/groups.py` | Which registry configs make up each named ablation table, so `run_ablation.py --group` can re-score one |
| `src/run_ablation.py` | Ablation driver, local, one SLURM array task, or one named table |
| `src/mrt_check.py` | The MRT columns rebuilt from one row per operating station, against the supplied table as joined |
| `src/tune.py` | Random hyper-parameter search |
| `src/tuned_params.py` | The winning parameters, frozen |
| `src/h2h.py` | Port of `notebook_baseline.ipynb`, scored on our splits, and the third blend member. `--inner rows` is the default and reproduces the notebook's own inner `TimeSeriesSplit` tuning, which is what the scored submissions and the reported tables used. `--inner month` repairs it, as described below, and writes to `_repaired` filenames |
| `src/confirm.py` | Re-scores a LightGBM winner on CatBoost and the blend |
| `src/blend_weights.py` | Scores confirmed variants inside the three-member blend |
| `src/members.py` | Candidate extra members, per-fold predictions |
| `src/blend_members.py` | Weight search for those candidates |
| `src/ladder.py` | Eleven untuned model classes on one setup, with and without the index |
| `src/altreps.py` | K-means location clusters and PCA as alternatives to the administrative columns and the raw numeric block |
| `src/eda.py` | Recomputes every statistic the report quotes in Section II, from the raw CSVs, including the Tukey fence, the variance decomposition, the within-group noise and the street-name join. Reads nothing in `outputs/` |
| `src/baseline.py` | The two baselines of Section IV, ported unchanged from the first notebook so their quoted holdout scores and the permutation importance of Appendix D can be measured rather than recalled. Writes `outputs/permutation_importance.csv` |
| `src/index_horizon.py` | What each index forecast implies for the market level over the real test months, which the folds cannot measure, and how unevenly that level rose across flat types. Reproduces Appendix G |
| `src/additive_check.py` | Matched global-index comparison using identical structural and block-composition features, fixed LightGBM settings, and training-only indices. Writes `outputs/additive_index_check.csv`. Multiplicative versus additive rolling RMSE is 519.41 versus 525.56, with confirmation RMSE 467.04 versus 470.68. The additive alternative is not independently tuned |
| `src/compare_submissions.py` | Separates the cross-section from the market level in two scored submissions, which is how `ridge_raw_rent_optuna.csv` is read against the shipped blend |
| `src/error_decomp.py` | Compares estimated conditional noise, oracle-index improvement, and an algebraic remainder on repeated-predictor rows |
| `src/blockhist.py` | The notebook model with and without block rent history |
| `src/residuals.py` | Residuals by horizon and slice, and the horizon tilt they motivate |
| `src/predict.py` | Final fit and submission |
| `src/figs_eda.py`, `src/figs_ablation_summary.py` to `src/figs_residuals.py` | Figures for the report. All but the first only read `outputs/` |
| `notebook_baseline.py`, `notebook_baseline.ipynb` | The team's original HistGB notebook and its single-file port, kept as the starting point every later result is measured against. Also the source of the permutation importances in Appendix D of the report |
| `scripts/*.sbatch` | The two SLURM arrays. Each task writes its own file under `outputs/parts/`, which `--merge` collects |
| `docs/` | The progress report and the feedback we received on it, which Sections V, VII and VIII respond to |

`src/encode.py` holds two separate things. Its hierarchical target encoding is not used by the shipped model and stays in the tree because the encoding ablations are a reported result. Its spatial encoder is a different matter: `spatial` is on in the shipped config, so those columns are in the final model.

## Validation Protocol

Rolling origin, cutting at months 27, 31 and 35, each scored over a 16-month window so that the validation horizon matches the real one. The index is re-fitted and re-forecast inside every fold, so a fold pays for being wrong about the future price level. Month 39 gives a 12-month confirmation cut. We adopted a change only when it improved both the rolling folds and that later origin, so it is a secondary selection gate and not an independent test.

Where a configuration does use target encoding, the validation and test encodings are always built in `future` mode whatever the training-side regime, so no reported number can flatter itself through encoding leakage.

## Ablation Results

All figures are three-fold rolling-origin means, and the raw tables are in `outputs/`.

The tables were cut out of a running log while the experiments were being done, so their rows carry the display labels used at the time (`B1 index=flat`) rather than registry keys. `src/groups.py` maps each named table to the registry configurations it reports, and `src/run_ablation.py --group <name>` re-scores one. Run `--list-groups` to see the eleven groups and their coverage.

What a rerun reproduces depends on the batch. Every configuration inherits today's defaults, and those were refined after the early batches were scored, so a rerun is the batch judged by the finished pipeline rather than a replay of it. Reruns therefore go to `outputs/ablation_<name>_rerun.csv` and never overwrite the shipped table, which stays as the record of what was decided and when. The two groups we re-ran bracket the range. `index_forecast` returns its ranking under shifted levels. `target_encoding` returns the shipped numbers themselves. Seven groups contain only registered configurations, and the other four also contain variants that predate the registry and were never added to it.

One group needed more than a mapping. The target-encoding batch varied `te_k` and `te_half_life`, which `experiment.py` reads only when `te_mode` is not `"none"`, and `"none"` is the default now. Pointed at the originally named configs the group would have rerun as six copies of `base` without raising anything. The batch was actually run with `loo_month` as the default mode, so `configs.py` gained `te_loo_ref`, `te_loo_k4`, `te_loo_k30`, `te_loo_hl24` and `te_loo_hl12`, which pin that mode and keep today's `drop=["HORIZON_M"]`. The log target and no-spatial rows had the same problem one level up, since each varies one switch and inherits the rest, so `te_loo_log` and `te_loo_nospatial` pin the mode for those too. The group uses the nine pinned configs, and the originals stay because the ablation log names them. Under them the batch reproduces exactly, in `outputs/ablation_target_encoding_rerun.csv`: all nine rows match the shipped table to the decimal, fold scores and bias columns included.

The mode is not an incidental detail. The same sweep run under `expanding` instead, kept in the registry as `te_exp_*`, ranks the half-life settings in the opposite order: a 12-month half-life is the best of the four under `loo_month` and the worst under `expanding`. An inherited default can empty an ablation of its contrast, or quietly invert it, without raising anything.

`index_forecast` is the harder of the two, in `outputs/ablation_index_forecast_rerun.csv`. Every row improves by roughly a constant amount and the ranking is unchanged, which is what we mean by a comparison reproducing without its levels:

| Config | As decided | Rerun today |
|---|---|---|
| `idx_damp_w12_d06` (shipped) | 539.71 | **523.32** |
| `idx_damp_w6_d07` | 540.17 | 523.85 |
| `idx_flat` | 541.53 | 526.55 |
| `idx_damp_w18_d085` | 558.51 | 539.73 |
| `idx_linear` | 672.55 | 652.23 |

- Index extrapolation dominates everything else. Undamped linear extrapolation scores 672.6, an 18-month damped trend 558.5, a flat index 541.5, and the 12-month damped trend we ship 539.7. A long window inherits the 2022 surge and carries it straight through a regime break.
- Dropping the index altogether costs 86 RMSE on LightGBM and 140 on CatBoost, the largest single effect in the study.
- Target encoding of own-block rent increases RMSE by 6 relative to no encoding in the evaluated configuration. Giving each training row block history from a random 1 to 16 months earlier increases RMSE by 9 to 15. This attempt to match training and prediction staleness does not improve the model. It does not isolate staleness as the cause or exclude its interaction with encoding variance.
- Model differences are modest in this batch. CatBoost scores 521.5, LightGBM 523.3, XGBoost 528.7, and one-hot ridge 534.3. These results motivated retaining the tested boosting models; they do not establish the performance of untested model classes.
- Blending LightGBM and CatBoost 50/50 gives 518.4, adding the notebook at 0.3 takes it to 516.7, and the per-flat-type index with block composition reaches 515.7. Both refinements survive at the month-39 origin as well as on the rolling folds. That origin is a second gate we required changes to clear, not an untouched test set, so it confirms less than an independent holdout would.
- Snapping predictions to a price grid does not improve the evaluated model. Although 98.7% of observed rents are exact multiples of S$50, snapping predictions to that grid increases RMSE by 0.16. Squared error is minimized by the conditional mean, which need not be a multiple of S$50.
- COE premiums correlate 0.91 with the rent index in levels and still lose to a random walk out of sample. In `src/macro.py` every forecaster, the macro regressions included, is fitted on the index estimated from months before its own cut, so none sees a rent approved after its origin.
- The MRT columns were built from the supplied table as joined, which lists 243 rows at 192 distinct locations because an interchange appears once per line, and which marks Changi Terminal 5 as open although that station is not built. Rebuilding all four columns from the 191 locations left once duplicates and Terminal 5 are dropped moves the score by 0.02 (`outputs/mrt_station_check.csv`). That measures de-duplication and one unbuilt station rather than the network a tenant travels on, so the counts remain proxies for centrality rather than exact station counts.
- Four candidates were rejected after failing the holdout: a ridge member, a notebook member given our features, horizon-dependent blend weights, and a LightGBM retune.
- The block-history member's inner tuning was repaired after the fact. `TimeSeriesSplit` divides the training rows by position, so a fold boundary falls inside a month and the search is scored partly on months it has already trained on, which rewards a setting that fits a month level over one that carries a level forward. `--inner month` puts the inner origins on whole months, at the outer 16-month horizon wherever the span allows and 14 months at the earliest cut, and rebuilds the block rent history from each fold's own training months. Both protocols draw the same eight candidates from the same sampler and seed, so the only thing that changes is how a setting is scored. The repair changes which setting wins at two of the four origins, cuts 27 and 39, the two where the positional search had taken a learning rate of 0.03 rather than 0.1. At the other two it picks the same setting and the fold predictions are identical to every decimal. Where it changes the outcome the member improves, by 1.87 at cut 27 and 0.22 at cut 39, so its rolling mean moves from 527.02 to 526.40 and the confirmation cut from 467.76 to 467.54, and the blend moves from 516.66 to 516.54 and from 463.77 to 463.82 with 30% still the best weight. So the defect was costing the member slightly rather than flattering it. The scored submissions predate the repair, so `src/predict.py` with its default `--inner rows` is what reproduces them.

Two results deserve emphasis. Dropping any single POI group costs between 0.0 and 0.2 RMSE, but dropping all ten together costs 1.58, because each group proxies the same question of how central a block is; a leave-one-out table read on its own would have justified deleting the entire POI join. And the blend
under-predicts furthest out, with a bias of +9.8 at horizons 1 to 4 against -35.6 at horizons 13 to 16, which is exactly where the test window is heaviest. The obvious fix is more trend persistence, and it is worse overall. It costs 56 RMSE on the fold that follows the 2022 surge.

For context, the corrected within-group noise standard deviation is S$472 for 2024 onward and S$443 for January to March 2025. These estimates use the quarter of rows with a feature-identical twin and describe dispersion on that subset. They do not establish an error floor for the leaderboard score. The report compares the same type of conditional estimate with model error on the corresponding repeated-predictor validation rows.

## Data Provenance

`data/RentingOutofFlatsfromJan2021.csv` is HDB's public rental dataset and the source this competition's train/test split was cut from. It runs to 2026-07, so it contains the test labels, and it is deliberately not joined anywhere in this pipeline. `grep -rn RentingOutofFlats src/ notebook_baseline.py` returns nothing, which is the check we ran before each submission. Because nothing reads it, it is not tracked in this repository and not included in the submitted bundle, and this note is kept as the record of where the competition data comes from. Re-downloading it from data.gov.sg is not needed to reproduce anything here.

The pipeline uses `data/train.csv`, `data/test.csv`, `data/auxiliary/*` and `data/HDBPropertyInformation.csv`. The last is data.gov.sg's static block register (units, flat-type mix, shops, car park). It carries no rents and no dates, so it cannot leak labels.
