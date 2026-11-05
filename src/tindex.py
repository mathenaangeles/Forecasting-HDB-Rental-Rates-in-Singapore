"""Mix-adjusted monthly rent index + out-of-sample extrapolation.

The test set lies 1-16 months *beyond* the last training month. Tree ensembles
cannot extrapolate: a split on a time feature simply saturates at the last
value seen in training. We therefore factor the target multiplicatively,

    RENT_it ~= BASE_i * INDEX_t

estimate INDEX_t on the training window, forecast it forward with a damped
linear trend, and let the GBDT learn only the (stationary) BASE part.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRAIN = ["STREET", "BLOCK", "FLAT_TYPE"]


def fit_index(df: pd.DataFrame, n_iter: int = 12, grain=GRAIN) -> pd.Series:
    """Iterative two-way (unit x month) index.

    Each pass sets every unit's base to the mean of its rents divided by the
    current index, then sets every month's index to the mean of its rents
    divided by the current bases. The updates are arithmetic means of ratios
    on rent in levels, not least squares on log-rent. This controls for
    changes in the *mix* of flats transacted each month, which a raw monthly
    mean does not.
    """
    y = df["MONTHLY_RENT"].to_numpy(float)
    t = df["TM"].to_numpy()
    g = df.groupby(grain, observed=True).ngroup().to_numpy()
    n_t, n_g = t.max() + 1, g.max() + 1

    idx = np.ones(n_t)
    for _ in range(n_iter):
        base = np.bincount(g, y / idx[t], n_g) / np.maximum(np.bincount(g, minlength=n_g), 1)
        base[base == 0] = y.mean()
        idx = np.bincount(t, y / base[g], n_t) / np.maximum(np.bincount(t, minlength=n_t), 1)
        idx[idx == 0] = 1.0
        idx /= idx[-1]  # Normalise so that the last observed month equals one.
    return pd.Series(idx, index=np.arange(n_t), name="INDEX")


def extrapolate(index: pd.Series, horizon: int, window: int = 18,
                damp: float = 0.85, method: str = "damped") -> pd.Series:
    """Extend `index` `horizon` months past its end.

    `method` selects the forecast. `flat` holds the last level, which is a
    random walk. `linear` extends the OLS slope of the last `window` months
    undamped. `damped` applies that slope with geometric damping
    (Gardner-McKenzie), and is the default because the Singapore rental market
    visibly decelerated after the 2022-23 surge, where an undamped trend
    overshoots.
    """
    last_t = int(index.index.max())
    lvl = float(index.iloc[-1])
    if method == "flat":
        slope = 0.0
    else:
        w = index.iloc[-window:]
        x = w.index.to_numpy(float)
        slope = np.polyfit(x, w.to_numpy(), 1)[0]
    phi = 1.0 if method == "linear" else damp
    steps = np.arange(1, horizon + 1)
    # The geometric sum phi + phi^2 + ... + phi^h.
    cum = np.cumsum(phi ** steps) if phi < 1 else steps.astype(float)
    fut = pd.Series(lvl + slope * cum, index=last_t + steps)
    return pd.concat([index, fut])
