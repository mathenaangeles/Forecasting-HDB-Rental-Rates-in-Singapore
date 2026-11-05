"""EDA + diagnostic figures for the report."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np, pandas as pd
import tindex, macro

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
df = pd.read_parquet("outputs/base.parquet")
tr = df[df._split == "train"]
te = df[df._split == "test"]

# Figure 1: the rent level over time, and how the index is extrapolated. This
# one goes into the report, so it is drawn at the width it is printed at, the
# full two-column text block of the IEEE template. Drawing it at its final size
# means the font sizes set here are the font sizes that reach the page.
W1, H1 = 7.16, 1.64
with plt.rc_context({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": .6, "xtick.major.width": .6,
                     "ytick.major.width": .6, "legend.fontsize": 6.8}):
    fig, ax = plt.subplots(1, 2, figsize=(W1, H1))
    m = tr.groupby("TM")["MONTHLY_RENT"].agg(["mean", "median",
                                              lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
    m.columns = ["mean", "median", "p10", "p90"]
    ax[0].fill_between(m.index, m.p10, m.p90, alpha=.18, label="10th to 90th percentile")
    ax[0].plot(m.index, m["mean"], lw=1.4, label="Mean")
    ax[0].axvspan(50, 66, alpha=.12, color="crimson", label="Test window")
    ax[0].set(xlabel="Months since January 2021", ylabel="Monthly rent (S$)",
              title="(a) Monthly Rent, 2021 to 2025")
    ax[0].legend(frameon=False, loc="upper left", handlelength=1.4,
                 borderpad=.2, labelspacing=.3)

    idx = tindex.fit_index(tr)
    for meth, w, d, lab in [("flat", 12, .6, "Flat (random walk)"),
                            ("damped", 12, .6, "Damped trend, 12 months, $\\phi$ = 0.6"),
                            ("linear", 18, 1, "Linear trend, 18 months")]:
        e = tindex.extrapolate(idx, 16, w, d, meth)
        ax[1].plot(e.index[45:], e.iloc[45:], lw=1.3, ls="--" if meth != "damped" else "-", label=lab)
    ax[1].plot(idx.index[45:], idx.iloc[45:], color="k", lw=1.6, label="Fitted index")
    ax[1].axvline(50, color="grey", lw=.7)
    ax[1].set(xlabel="Months since January 2021", ylabel="Index (March 2025 = 1)",
              title="(b) Index Extrapolation, Test Window")
    ax[1].legend(frameon=False, loc="upper left", handlelength=1.4,
                 borderpad=.2, labelspacing=.3)
    fig.tight_layout(pad=.4, w_pad=1.4)

    # Nothing may run past the edge of the canvas, because the figure is placed
    # at exactly this width and anything outside it is simply cut off.
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"fig1 content bbox {bb.x0:+.3f} {bb.y0:+.3f} to {bb.x1:.3f} {bb.y1:.3f} in, canvas {W1} x {H1}")
    if bb.x0 < -0.005 or bb.y0 < -0.005 or bb.x1 > W1 + 0.005 or bb.y1 > H1 + 0.005:
        raise SystemExit("fig1 overflows the canvas, adjust the layout")

    fig.savefig(f"{FIG}/fig_market_index.png", dpi=400); plt.close(fig)

# Figure 2: the target distribution, and why we do not take logs. This one is
# printed in the appendix at the full two-column text width, so it is drawn at
# that width and on every labeled rental rather than on a slice of them. The
# skew values in the titles are the ones quoted in Section II-B.
W2, H2 = 7.16, 2.05
with plt.rc_context({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": .6, "xtick.major.width": .6,
                     "ytick.major.width": .6}):
    fig, ax = plt.subplots(1, 3, figsize=(W2, H2))
    r = tr["MONTHLY_RENT"]
    ax[0].hist(r, bins=60)
    ax[0].set(title=f"(a) Monthly Rent (Skew {r.skew():+.2f})",
              xlabel="Monthly rent (S$)", ylabel="Rentals")
    ax[1].hist(np.log(r), bins=60, color="darkorange")
    ax[1].set(title=f"(b) Log Monthly Rent (Skew {np.log(r).skew():+.2f})",
              xlabel="Log monthly rent")
    ax[2].bar(*np.unique(r % 100, return_counts=True), width=8)
    ax[2].set(title="(c) Rent Modulo S$100", xlabel="Remainder (S$)")
    fig.tight_layout(pad=.4, w_pad=1.2)

    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"fig2 content bbox {bb.x0:+.3f} {bb.y0:+.3f} to {bb.x1:.3f} {bb.y1:.3f} in, canvas {W2} x {H2}")
    if bb.x0 < -0.005 or bb.y0 < -0.005 or bb.x1 > W2 + 0.005 or bb.y1 > H2 + 0.005:
        raise SystemExit("fig2 overflows the canvas, adjust the layout")

    fig.savefig(f"{FIG}/fig_target_distribution.png", dpi=400); plt.close(fig)

# Figure 3: the noise floor, and the geography underneath it.
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
k = ["STREET", "BLOCK", "FLAT_TYPE", "TM"]
res = []
for y, d in tr.groupby(tr.TM // 12 + 2021):
    g = d.groupby(k)["MONTHLY_RENT"]
    n, e = g.transform("size"), d["MONTHLY_RENT"] - g.transform("mean")
    mm = n > 1
    ng = d[mm].groupby(k).ngroups
    res.append((y, np.sqrt((e[mm] ** 2).sum() / (mm.sum() - ng))))
res = pd.DataFrame(res, columns=["year", "within_rmse"])
lab = [f"{y}" if y < 2025 else "2025\n(Jan to Mar)" for y in res.year]
ax[0].bar(lab, res.within_rmse, color="steelblue", width=.6)
for i, v in enumerate(res.within_rmse):
    ax[0].text(i, v + 12, f"{v:.0f}", ha="center", fontsize=7.5)
ax[0].axhline(436.05, color="crimson", ls="--", lw=1.2, label="HistGB public leaderboard (436.05)")
ax[0].set_ylim(0, 700)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[0].set(ylabel="RMSE (S$)", title="(a) RMSE Among Rows with Identical Features")

sc = tr.groupby(["STREET", "BLOCK"]).agg(lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
                                         r=("MONTHLY_RENT", "size"))
# Per square metre, so the map shows geography rather than the local mix of
# flat sizes.
rat = tr.assign(R=tr.MONTHLY_RENT / idx.reindex(tr.TM).to_numpy() / tr.FLOOR_AREA_SQM) \
        .groupby(["STREET", "BLOCK"])["R"].mean()
sc["rat"] = rat
s = ax[1].scatter(sc.lon, sc.lat, c=sc.rat, s=3, cmap="RdYlBu_r",
                  vmin=sc.rat.quantile(.02), vmax=sc.rat.quantile(.98))
cb = fig.colorbar(s, ax=ax[1], shrink=.85, pad=.02)
cb.set_label("Detrended rent (S$ per m² per month)", fontsize=7.5)
ax[1].set(title="(b) Detrended Rent per Square Metre by Block", xlabel="Longitude", ylabel="Latitude")
ax[1].xaxis.set_major_locator(MaxNLocator(4)); ax[1].yaxis.set_major_locator(MaxNLocator(4))
ax[1].set_aspect("equal"); ax[1].grid(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_eda_floor_and_geography.png"); plt.close(fig)

# Figure 4: the auxiliary macro series, correlated in levels but useless out of
# sample.
M = macro.monthly()
li = np.log(idx)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].plot(li.index, (li - li.iloc[0]) * 100, lw=2, color="k", label="Rent index")
for c, lab in [("MACRO_COE", "COE premium"), ("MACRO_PROP", "SGX property and REIT basket"),
               ("MACRO_BANK", "SGX banks")]:
    ax[0].plot(M.index, M[c] * 100, lw=1.2, alpha=.8, label=lab)
ax[0].axvline(50, color="grey", lw=.8)
ax[0].set(xlabel="Months since January 2021", ylabel="Log change since January 2021 (%)",
          title="(a) Rent Index and Macro Series")
ax[0].set_ylim(-25, 165)
ax[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
# The bars are recomputed by macro.backtest on the three rolling origins, so the
# holdout month plays no part in them.
bt = macro.backtest(tr, cuts=(27, 31, 35))
bars = [("damped", "Damped\ntrend"), ("flat", "Random\nwalk"),
        ("lvl:MACRO_PROP", "Property\nlevels"), ("lvl:MACRO_COE", "COE\nlevels"),
        ("lvl:MACRO_COE_by_cat", "COE by\ncategory"), ("d12:MACRO_COE", "COE 12-month\ndifferences")]
val = [bt.loc[m].mean() for m, _ in bars]
ax[1].bar([b for _, b in bars], val, width=.6, color=["steelblue"] * 2 + ["indianred"] * 4)
for i, v in enumerate(val):
    ax[1].text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=7.5)
ax[1].set_ylim(0, max(val) * 1.12)
ax[1].set(ylabel="Mean absolute index error (%)",
          title="(b) Index Forecast Error, Three Rolling Folds")
plt.setp(ax[1].get_xticklabels(), fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_eda_macro_series.png"); plt.close(fig)
print("figures written to", FIG)
