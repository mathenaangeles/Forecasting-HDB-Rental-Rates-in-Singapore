"""CS5228 Project - HDB Rental Rate Prediction.

Single-file port of notebook_baseline.ipynb: run end-to-end, print the holdout residual
(RMSE) + permutation importance, and write the submission CSV.

    python notebook_baseline.py
    python notebook_baseline.py --skip-submission     # eval only, no refit on full data
    python notebook_baseline.py --n-iter 20           # wider hyperparam search
    python notebook_baseline.py --holdout-months 16   # match the real 16-month test horizon
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import BallTree

#CONFIG - EDIT HERE, NOT IN THE BODY
DATA_DIR = Path('data')
AUX_DIR = DATA_DIR / 'auxiliary'
SUBMISSION_PATH = Path('submissions/histgb_blockhist.csv')

EARTH_RADIUS_KM = 6371
RANDOM_STATE = 42
HOLDOUT_MONTHS = 12          #LAST N CALENDAR MTHS HELD OUT FOR THE UNBIASED SCORE
                             #WHY MTHS NOT %: TEST.CSV SPANS 16 MTHS PAST TRAIN END, SO THE
                             #HOLDOUT HORIZON MUST MATCH THAT - AND A % CUT SPLITS ONE MTH
                             #ACROSS BOTH SIDES (SAME BLOCK+MTH IN TRAIN *AND* VAL = LEAK)
N_SPLITS = 4                 #TIMESERIESSPLIT FOLDS INSIDE THE SEARCH
N_ITER = 8                   #RANDOMIZEDSEARCHCV SAMPLES
PERM_REPEATS = 5

PARAM_DIST = {
    'learning_rate': [0.03, 0.05, 0.1, 0.15],
    'max_leaf_nodes': [15, 31, 63],
    'max_iter': [100, 200, 300],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 0.1, 1.0],
}

NUM_FEATS = ['FLOOR_AREA_SQM', 'LEASE_AGE_YR', 'MTH_SINCE_START', 'MTH_OF_YR',
             'DIST_MRT_KM', 'DIST_MALL_KM', 'DIST_SCH_KM',
             'COE_PREMIUM_AVG', 'STOCK_CLOSE_AVG']
CAT_FEATS = ['TOWN', 'FLAT_TYPE', 'FLAT_MODEL']

#LAGGED BLOCK HISTORY - MEAN RENT OF THE SAME BLOCK+FLAT_TYPE, PAST ROWS ONLY.
#GRAIN CHOSEN BY ABLATION: BLOCK ALONE BEATS BLOCK+STREET+TOWN (487.4 VS 489.5 RMSE
#ON A 16-MTH HOLDOUT, 5 SEEDS) - COARSER GRAINS JUST RESTATE THE TOWN/STREET CATS.
HIST_KEYS = ['STREET', 'BLOCK', 'FLAT_TYPE']
HIST_COLS = ['HIST_BLK', 'HIST_BLK_N']
FEAT_COLS = NUM_FEATS + CAT_FEATS + HIST_COLS

MTH_NUM = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

#BENCHMARKS FOR CONTEXT AT THE END OF THE RUN
LEADERBOARD_RMSE = 436.05    #OUR OWN PUBLIC LB SCORE, HISTGB + BLOCK HISTORY (2026-09)
                             #FOR REFERENCE: 441.63 WAS "CS5228 Rent Kakis" ON 2026-08-31,
                             #AND THE HOLDOUT HERE READS ~30-50 PTS WORSE THAN THE REAL LB
                             #(IT TRAINS ON LESS DATA OVER A LONGER HORIZON - BY DESIGN)
BASELINE_RMSE = 700.33       #PRIOR NAIVE LINEAR REGRESSION


def load_data():
    """LOAD CORE TRAIN/TEST + AUX TABLES (GEO/LANDMARK/MACRO SRC)."""
    df_train = pd.read_csv(DATA_DIR / 'train.csv')
    df_test = pd.read_csv(DATA_DIR / 'test.csv')
    print('train:', df_train.shape, '| test:', df_test.shape)

    aux = {
        'hdb_block': pd.read_csv(AUX_DIR / 'sg-hdb-block.csv'),
        'mrt': pd.read_csv(AUX_DIR / 'sg-mrt-stations.csv'),
        'mall': pd.read_csv(AUX_DIR / 'sg-shopping-malls.csv'),
        'school': pd.read_csv(AUX_DIR / 'sg-schools.csv'),
        'coe': pd.read_csv(AUX_DIR / 'sg-coe-prices.csv'),
        'stock': pd.read_csv(AUX_DIR / 'sg-stock-prices.csv'),
    }
    return df_train, df_test, aux


def clean(df):
    """NORMALIZE CATS, DROP ZERO-VARIANCE COLS.

    WHY: "4 ROOM"/"4-ROOM" ARE 1 CAT SPLIT IN 2 BY FORMATTING - FEE/FURNISHED
    ARE CONST ACROSS ALL ROWS (KNOWN FROM PRIOR EDA), 0 SIGNAL.
    """
    df = df.copy()
    df['FLAT_TYPE'] = df['FLAT_TYPE'].str.replace(' ', '-', regex=False)
    df['STREET'] = df['STREET'].str.lower().str.strip()

    return df.drop(columns=['FEE', 'FURNISHED'])


def add_temporal_feats(df):
    """MTH SINCE START, MTH-OF-YR, LEASE AGE.

    WHY: 2021-2026 RENT SHOWS STRONG UPWARD TREND & TEST SET IS FUTURE-ONLY
    --> MODEL MUST EXTRAPOLATE ON AN ORDINAL TIME TERM, NOT JUST INTERPOLATE.
    """
    df['RENT_APPROVAL_DATE'] = pd.to_datetime(df['RENT_APPROVAL_DATE'], format='%Y-%m')
    df['MTH_SINCE_START'] = (df['RENT_APPROVAL_DATE'].dt.year - 2021) * 12 + df['RENT_APPROVAL_DATE'].dt.month
    df['MTH_OF_YR'] = df['RENT_APPROVAL_DATE'].dt.month
    df['LEASE_AGE_YR'] = df['RENT_APPROVAL_DATE'].dt.year - df['LEASE_COMMENCE_DATE']

    return df


def add_geo(df, df_hdb_block):
    """MERGE ON TOWN+BLOCK+STREET FOR LAT/LON.

    WHY: LAT/LON REQ FOR MRT/MALL/SCHOOL DIST FEATS - ONLY AVAIL JOIN KEY.
    ~80% MATCH (KNOWN FROM PRIOR EDA) - REST STAY NaN, HISTGB HANDLES NATIVELY.
    """
    join_cols = ['TOWN', 'BLOCK', 'STREET']

    return df.merge(df_hdb_block[join_cols + ['LATITUDE', 'LONGITUDE']], on=join_cols, how='left')


def nearest_dist_km(src_lat, src_lon, ref_lat, ref_lon):
    """NEAREST LANDMARK DIST (BALLTREE, HAVERSINE) - KM."""
    valid = src_lat.notna() & src_lon.notna()
    dist_km = pd.Series(np.nan, index=src_lat.index)

    if valid.sum() == 0:
        return dist_km

    ref_rad = np.radians(np.column_stack([ref_lat, ref_lon]))
    tree = BallTree(ref_rad, metric='haversine')
    src_rad = np.radians(np.column_stack([src_lat[valid], src_lon[valid]]))
    dist_rad, _ = tree.query(src_rad, k=1)
    dist_km.loc[valid] = dist_rad[:, 0] * EARTH_RADIUS_KM

    return dist_km


def add_landmark_dists(df, df_mrt, df_mall, df_school):
    """DIST TO NEAREST MRT/MALL/PRIMARY SCHOOL.

    WHY: PROXIMITY TO AMENITIES = CLASSIC SG HOUSING PRICE DRIVER (PER PROJECT
    SPEC) - SCHOOLS FILTERED TO PRIMARY ONLY, SPEC ASKS FOR PRIMARY NOT SECONDARY.
    """
    df_school_primary = df_school[df_school['MAINLEVEL_CODE'] == 'primary']

    df['DIST_MRT_KM'] = nearest_dist_km(df['LATITUDE'], df['LONGITUDE'], df_mrt['LATITUDE'], df_mrt['LONGITUDE'])
    df['DIST_MALL_KM'] = nearest_dist_km(df['LATITUDE'], df['LONGITUDE'], df_mall['LATITUDE'], df_mall['LONGITUDE'])
    df['DIST_SCH_KM'] = nearest_dist_km(df['LATITUDE'], df['LONGITUDE'], df_school_primary['LATITUDE'], df_school_primary['LONGITUDE'])

    return df


def build_coe_monthly(df_coe):
    """MEAN PREMIUM ACROSS CAT+ROUND, 1 SIGNAL/MTH.

    WHY: SPEC FLAGS COE AS ECON INDICATOR OF RENT "CLIMATE".
    """
    df_coe = df_coe.copy()
    df_coe['QUOTA_PREMIUM'] = df_coe['QUOTA_PREMIUM'].str.replace(r'[\$,]', '', regex=True).astype(float)
    df_coe['MTH_SINCE_START'] = (df_coe['YEAR'] - 2021) * 12 + df_coe['MONTH'].map(MTH_NUM)

    return df_coe.groupby('MTH_SINCE_START')['QUOTA_PREMIUM'].mean().reset_index(name='COE_PREMIUM_AVG')


def build_stock_monthly(df_stock):
    """MEAN CLOSE ACROSS ALL CO., 1 SIGNAL/MTH.

    WHY: SPEC FLAGS STOCK PRICES AS GEN ECON INDICATOR - SAME COLLAPSE-TO-1-SIGNAL
    APPROACH AS COE.
    """
    df_stock = df_stock.copy()
    df_stock['DATE'] = pd.to_datetime(df_stock['DATE'])
    df_stock['MTH_SINCE_START'] = (df_stock['DATE'].dt.year - 2021) * 12 + df_stock['DATE'].dt.month

    return (df_stock.dropna(subset=['CLOSE'])
            .groupby('MTH_SINCE_START')['CLOSE'].mean()
            .reset_index(name='STOCK_CLOSE_AVG'))


def build_features(df_train, df_test, aux):
    """FULL FEATURE PIPELINE - RETURNS THE 2 ENRICHED FRAMES."""
    df_train, df_test = clean(df_train), clean(df_test)
    df_train, df_test = add_temporal_feats(df_train), add_temporal_feats(df_test)

    df_train = add_geo(df_train, aux['hdb_block'])
    df_test = add_geo(df_test, aux['hdb_block'])
    print(f'geo match train: {df_train["LATITUDE"].notna().mean():.1%} | test: {df_test["LATITUDE"].notna().mean():.1%}')

    df_train = add_landmark_dists(df_train, aux['mrt'], aux['mall'], aux['school'])
    df_test = add_landmark_dists(df_test, aux['mrt'], aux['mall'], aux['school'])

    coe_monthly = build_coe_monthly(aux['coe'])
    df_train = df_train.merge(coe_monthly, on='MTH_SINCE_START', how='left')
    df_test = df_test.merge(coe_monthly, on='MTH_SINCE_START', how='left')
    print(f'COE match train: {df_train["COE_PREMIUM_AVG"].notna().mean():.1%} | test: {df_test["COE_PREMIUM_AVG"].notna().mean():.1%}')

    stock_monthly = build_stock_monthly(aux['stock'])
    df_train = df_train.merge(stock_monthly, on='MTH_SINCE_START', how='left')
    df_test = df_test.merge(stock_monthly, on='MTH_SINCE_START', how='left')
    print(f'stock match train: {df_train["STOCK_CLOSE_AVG"].notna().mean():.1%} | test: {df_test["STOCK_CLOSE_AVG"].notna().mean():.1%}')

    return df_train, df_test


def cast_categoricals(df_train, df_test):
    """CAST CAT DTYPES IN PLACE.

    WHY: HISTGB NATIVE CAT SUPPORT NEEDS PANDAS 'CATEGORY' DTYPE - UNION
    TRAIN+TEST CATS SO NO UNSEEN-CAT MISMATCH AT PREDICT TIME. STREET/BLOCK
    STAY OUT OF CAT_FEATS (NEAR-UNIQUE, WOULD OVERFIT) - THEY ARE ONLY JOIN
    KEYS, INCL. FOR THE BLOCK-HISTORY FEATURE BELOW.
    """
    for col in CAT_FEATS:
        all_cats = pd.concat([df_train[col], df_test[col]]).unique()
        df_train[col] = pd.Categorical(df_train[col], categories=all_cats)
        df_test[col] = pd.Categorical(df_test[col], categories=all_cats)


def add_block_history(source, target, is_future):
    """MEAN RENT OF THE SAME BLOCK+FLAT_TYPE FROM *PAST* ROWS ONLY.

    WHY: "WHAT DID THE IDENTICAL FLAT NEXT DOOR GO FOR" IS THE STRONGEST SIGNAL
    AVAILABLE - BUT IT IS BUILT FROM THE TARGET, SO IT LEAKS UNLESS EVERY VALUE
    IS COMPUTED STRICTLY FROM ROWS THE MODEL WOULD REALLY HAVE HAD.

    TWO MODES, MIRRORING THE TWO REAL SITUATIONS:
      is_future=True   TARGET SITS ENTIRELY AFTER SOURCE (holdout, test.csv) ->
                       USE ALL OF SOURCE. THIS IS THE "FROZEN CUTOFF": AT PREDICT
                       TIME THE NEWEST LABEL IS 1-16 MTHS OLD, AND SO IS THIS.
      is_future=False  TARGET *IS* SOURCE (the training rows) -> EXPANDING MEAN
                       OVER STRICTLY EARLIER MONTHS. THE shift(1) IS WHAT DROPS
                       THE CURRENT MONTH, SO A ROW NEVER SEES ITSELF NOR ITS
                       OWN MONTH'S SIBLINGS.

    RETURNS (mean, n) ALIGNED TO target'S ROW ORDER. NO-HISTORY ROWS STAY NaN -
    HISTGB SPLITS ON MISSING NATIVELY, AND HIST_BLK_N TELLS IT HOW MUCH TO TRUST
    THE MEAN IT DID GET.
    """
    if is_future:
        stats = (source.groupby(HIST_KEYS, observed=True)['MONTHLY_RENT']
                 .agg(HIST_BLK='mean', HIST_BLK_N='size').reset_index())
        out = target[HIST_KEYS].merge(stats, on=HIST_KEYS, how='left')
    else:
        by_month = (source.groupby(HIST_KEYS + ['MTH_SINCE_START'], observed=True)
                    ['MONTHLY_RENT'].agg(s='sum', c='size')
                    .reset_index().sort_values('MTH_SINCE_START'))
        g = by_month.groupby(HIST_KEYS, observed=True)
        by_month[['cs', 'cc']] = g[['s', 'c']].cumsum()
        by_month[['cs', 'cc']] = by_month.groupby(HIST_KEYS, observed=True)[['cs', 'cc']].shift(1)
        by_month['HIST_BLK'] = by_month['cs'] / by_month['cc']
        by_month['HIST_BLK_N'] = by_month['cc']
        out = target[HIST_KEYS + ['MTH_SINCE_START']].merge(
            by_month[HIST_KEYS + ['MTH_SINCE_START'] + HIST_COLS],
            on=HIST_KEYS + ['MTH_SINCE_START'], how='left')

    return out['HIST_BLK'].to_numpy(), out['HIST_BLK_N'].to_numpy()


def build_X(target, source, is_future):
    """FEATURE MATRIX FOR target, WITH BLOCK HISTORY DRAWN FROM source."""
    X = target[NUM_FEATS + CAT_FEATS].copy()
    X['HIST_BLK'], X['HIST_BLK_N'] = add_block_history(source, target, is_future)

    return X[FEAT_COLS]


def split_by_month(df, holdout_months):
    """SORT BY DATE, HOLD OUT THE LAST N WHOLE CALENDAR MTHS.

    WHY: KAGGLE TEST.CSV IS ITSELF A FUTURE-ONLY SPLIT OF TRAIN.CSV - RANDOM
    SPLIT WOULD LEAK FUTURE ROWS INTO VAL & OVERSTATE SCORE. CUTTING ON A MTH
    BOUNDARY (NOT A ROW %) KEEPS THE HORIZON INTERPRETABLE AND STOPS THE SAME
    BLOCK+MTH LANDING ON BOTH SIDES OF THE SPLIT.

    RETURNS (sorted full frame, search frame, holdout frame). SPLITTING FRAMES
    RATHER THAN MATRICES MATTERS: build_X NEEDS STREET/BLOCK/MTH_SINCE_START
    AFTERWARDS, AND THE HISTORY FEATURE MUST BE BUILT *AFTER* THE CUT SO THE
    HOLDOUT NEVER FEEDS ITS OWN SOURCE STATS.
    """
    df = df.sort_values('RENT_APPROVAL_DATE').reset_index(drop=True)
    dates = df['RENT_APPROVAL_DATE']

    #KEEP months AS A PANDAS SERIES - .to_numpy()/.unique() WOULD DOWNGRADE THE
    #ELEMENTS TO np.datetime64, WHICH CANNOT BE strftime-FORMATTED IN AN f-STRING
    months = dates.drop_duplicates().sort_values()
    if not 0 < holdout_months < len(months):
        raise ValueError(f'holdout_months must be in 1..{len(months) - 1}, got {holdout_months}')

    cut_date = months.iloc[-holdout_months]
    is_hold = dates >= cut_date
    df_search, df_hold = df[~is_hold], df[is_hold]

    print(f'search : {dates.min():%Y-%m} -> {df_search["RENT_APPROVAL_DATE"].max():%Y-%m}  '
          f'({len(months) - holdout_months} mths, {len(df_search)} rows)')
    print(f'holdout: {cut_date:%Y-%m} -> {dates.max():%Y-%m}  '
          f'({holdout_months} mths, {len(df_hold)} rows, horizon 1-{holdout_months} mths ahead)')
    print(f'         [real test.csv horizon is 1-16 mths ahead]')

    return df, df_search, df_hold


def run_search(X_search, y_search, n_iter):
    """RANDOMIZEDSEARCHCV + TIMESERIESSPLIT.

    WHY: TIMESERIESSPLIT KEEPS EACH CV FOLD CHRONOLOGICAL (NO FUTURE-LEAK) -
    HOLDOUT SET STAYS FULLY UNTOUCHED FOR THE FINAL UNBIASED SCORE.
    """
    base_model = HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model, PARAM_DIST, n_iter=n_iter, cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring='neg_root_mean_squared_error', random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_search, y_search)

    print('best params:', search.best_params_)
    print(f'best CV RMSE: {-search.best_score_:.2f}')

    return search


def evaluate(model, X_hold, y_hold):
    """HOLDOUT RESIDUAL (UNBIASED - NEVER SEEN DURING SEARCH) + PERM IMPORTANCE."""
    hold_pred = model.predict(X_hold)
    resid = y_hold - hold_pred
    rmse_hold = root_mean_squared_error(y_hold, hold_pred)

    print()
    print('=== HOLDOUT RESIDUALS ===')
    print(f'RMSE  : {rmse_hold:.2f}')
    print(f'MAE   : {resid.abs().mean():.2f}')
    print(f'BIAS  : {resid.mean():+.2f}   (mean signed residual; <0 = over-predicting)')
    print(f'quantiles (5/25/50/75/95): {np.percentile(resid, [5, 25, 50, 75, 95]).round(1)}')
    print(f'leaderboard benchmark: {LEADERBOARD_RMSE} | prior LR baseline: {BASELINE_RMSE}')

    perm = permutation_importance(model, X_hold, y_hold, n_repeats=PERM_REPEATS,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=FEAT_COLS).sort_values(ascending=False)
    print()
    print('=== PERMUTATION IMPORTANCE (HOLDOUT) ===')
    print(importance.to_string())

    return rmse_hold


def write_submission(best_params, X_train_full, y_train_full, X_test):
    """REFIT ON FULL LABELED DATA + PREDICT TEST + SAVE.

    WHY: STD KAGGLE PRACTICE - ONCE HYPERPARAMS CHOSEN, REFIT ON 100% OF LABELED
    ROWS BEFORE PREDICTING REAL TEST SET, NO ROWS WASTED ON FINAL MODEL.
    """
    final_model = HistGradientBoostingRegressor(
        categorical_features='from_dtype', random_state=RANDOM_STATE, **best_params)
    final_model.fit(X_train_full, y_train_full)
    test_pred = final_model.predict(X_test)

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({'Id': range(len(test_pred)), 'Predicted': test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print()
    print(f'wrote {SUBMISSION_PATH} ({len(submission)} rows)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-iter', type=int, default=N_ITER, help='RandomizedSearchCV samples')
    parser.add_argument('--holdout-months', type=int, default=HOLDOUT_MONTHS,
                        help='hold out the last N calendar months (test.csv spans 16)')
    parser.add_argument('--skip-submission', action='store_true', help='eval only, skip the full refit + CSV')
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)

    df_train, df_test, aux = load_data()
    df_train, df_test = build_features(df_train, df_test, aux)
    cast_categoricals(df_train, df_test)
    df_train, df_search, df_hold = split_by_month(df_train, args.holdout_months)

    #EVAL: HISTORY COMES FROM df_search ONLY - THE HOLDOUT IS "THE FUTURE" HERE
    X_search = build_X(df_search, df_search, is_future=False)
    X_hold = build_X(df_hold, df_search, is_future=True)
    search = run_search(X_search, df_search['MONTHLY_RENT'], args.n_iter)
    evaluate(search.best_estimator_, X_hold, df_hold['MONTHLY_RENT'])

    if not args.skip_submission:
        #SUBMIT: HYPERPARAMS ARE FIXED NOW, SO HISTORY MAY USE ALL LABELED ROWS
        X_full = build_X(df_train, df_train, is_future=False)
        X_test = build_X(df_test, df_train, is_future=True)
        write_submission(search.best_params_, X_full, df_train['MONTHLY_RENT'], X_test)


if __name__ == '__main__':
    main()