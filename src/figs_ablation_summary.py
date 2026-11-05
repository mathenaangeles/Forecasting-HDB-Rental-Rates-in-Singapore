import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# The figure is drawn at the width it occupies in the report, which is the full
# two-column text block of the IEEE template. Because the drawing size and the
# printed size agree, the font sizes set here are the sizes that reach the page.
W, H = 7.16, 1.82

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "xtick.major.width": .6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "ytick.major.width": .6})

# Every number below is read from the committed ablation outputs, so the figure
# can be regenerated without rerunning any model.
A = pd.read_csv("outputs/ablations_all.csv").set_index("name")["mean_rmse"]
# These are the tables written under the rows inner protocol, which is what the
# report quotes. The _repaired tables are the other protocol and are not mixed in.
J = pd.read_csv("outputs/ablation_blend_weights.csv").set_index("name")

# The axes are positioned by hand rather than by tight_layout, because the two
# panels carry long category names and the automatic layout spills them over
# the edge of the canvas.
fig = plt.figure(figsize=(W, H))
ax0 = fig.add_axes([0.196, 0.238, 0.254, 0.672])
ax1 = fig.add_axes([0.670, 0.238, 0.315, 0.672])

# Left panel: how the score improved, batch by batch. Each bar is the best
# setting of its batch on the three rolling folds. The stages are separate
# configurations rather than a continuous quantity, so they are drawn as bars,
# laid out horizontally so that every stage name fits on a single line.
steps = [("Damped index (B)", A["B3 index=damped w12 d0.6"]),
         ("No horizon feature (D)", A["D0 loo -HORIZON_M (new default)"]),
         ("No target encoding (D)", A["D6 no TE"]),
         ("CatBoost (F)", A["F1 CatBoost"]),
         ("LightGBM + CatBoost (J)", J.loc["J0_ref", "2way_roll16"]),
         ("+ HistGB (J)", J.loc["J0_ref", "3way_roll16"]),
         ("+ Index, block info (J)", J.loc["Jc_ft05_blockinfo", "3way_roll16"])]
vals = [v for _, v in steps]
y = np.arange(len(vals))
BASE = 505  # The axis is truncated, so every bar carries its own value.
ax0.barh(y, [v - BASE for v in vals], left=BASE, height=.6, color="#1b4965", zorder=3)
for i, v in enumerate(vals):
    ax0.text(v + 1.0, i, f"{v:.1f}", va="center", fontsize=6.5, zorder=4)
ax0.set_yticks(y); ax0.set_yticklabels([s for s, _ in steps])
ax0.invert_yaxis()
ax0.set_xlabel("Rolling-origin CV RMSE (S$)", labelpad=2)
ax0.set_title("(a) CV RMSE by Development Stage", pad=4)
ax0.set_xlim(BASE, 549); ax0.set_xticks([505, 515, 525, 535, 545])
ax0.grid(axis="x", alpha=.3)

# Right panel: what removing or changing each component costs. Each delta is
# measured against the default of its own ablation batch, because the default
# itself moved as the study progressed, so the index-extrapolation variants
# (marked with a dagger) are referenced to the batch-B default, and CatBoost
# without detrending (double dagger) to CatBoost with it.
B, CUR = A["B3 index=damped w12 d0.6"], A["D6 no TE"]
items = [("No detrending (LightGBM)", A["F4 LGBM no-detrend"] - CUR),
         ("No detrending (CatBoost) $\\ddag$", A["F5 CatBoost no-detrend"] - A["F1 CatBoost"]),
         ("Linear extrapolation $\\dag$", A["B5 index=linear w18"] - B),
         ("18-month damped index $\\dag$", A["B4 index=damped w18 d0.85"] - B),
         ("Flat index (random walk) $\\dag$", A["B1 index=flat"] - B),
         ("18-month training window", A["F7 train_window 18"] - CUR),
         ("Ridge instead of GBDT", A["F3 Ridge"] - CUR),
         ("Hierarchical target encoding", A["D0 loo -HORIZON_M (new default)"] - CUR),
         ("STREET, BLOCK as categorical", A["E3 +STREET +BLOCK cat"] - CUR),
         ("Recency-weighted rows", A["E5 sample halflife 24"] - CUR),
         ("XGBoost instead of LightGBM", A["F2 XGBoost"] - CUR),
         ("No spatial KNN prior", A["E1 no-TE no-spatial"] - CUR)]
items.sort(key=lambda r: r[1])
lbl = [i[0] for i in items]; d = [i[1] for i in items]
y = np.arange(len(d))
ax1.barh(y, d, height=.7,
         color=["#c1121f" if v > 40 else "#e07a5f" if v > 5 else "#a8a8a8" for v in d])
for i, v in enumerate(d):
    ax1.text(v + 3, i, f"+{v:.1f}", va="center", fontsize=6.5)
ax1.set_yticks(y); ax1.set_yticklabels(lbl)
ax1.axvline(0, color="k", lw=.6)
ax1.set_xlabel("Increase in CV RMSE (S$)", labelpad=2)
ax1.set_title("(b) Cost of Rejected Alternatives", pad=4)
ax1.set_xlim(-3, 182); ax1.set_xticks([0, 40, 80, 120, 160])
ax1.grid(axis="x", alpha=.3)

# The two notes sit below their own panels, anchored to the figure rather than
# to the axes so that they cannot collide with the axis labels.
fig.text(0.196, 0.042, f"Axis starts at {BASE}.", ha="left", fontsize=6, color="#555")
fig.text(0.985, 0.042,
         rf"$\dag$ Against batch-B ({B:.1f}).  $\ddag$ Against CatBoost ({A['F1 CatBoost']:.1f}).",
         ha="right", fontsize=6, color="#555")

os.makedirs("figures", exist_ok=True)

# Nothing may run past the edge of the canvas, because the figure is placed at
# exactly this width in the report and anything outside it is simply cut off.
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print(f"content bbox {bb.x0:+.3f} {bb.y0:+.3f} to {bb.x1:.3f} {bb.y1:.3f} in, canvas {W} x {H}")
if bb.x0 < -0.005 or bb.y0 < -0.005 or bb.x1 > W + 0.005 or bb.y1 > H + 0.005:
    raise SystemExit("content overflows the canvas, adjust the axes positions")

plt.savefig("figures/fig_ablation_summary.png", dpi=400)
print("wrote figures/fig_ablation_summary.png")
