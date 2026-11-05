import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# The figure is placed at the full two-column text width of the IEEE template,
# so it is drawn at exactly that width and the font sizes below are the ones
# that reach the page.
W, H_ = 7.16, 5.05
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": .6, "xtick.major.width": .6,
                     "ytick.major.width": .6, "axes.spines.top": False,
                     "axes.spines.right": False})

# Like Figure 5, this only reads the committed ablation outputs.
I = pd.read_csv("outputs/ablation_feature_groups.csv").set_index("name")["mean_rmse"]
# These are the tables written under the rows inner protocol, which is what the
# report quotes. The _repaired tables are the other protocol and are not mixed in.
H = pd.read_csv("outputs/ablation_head_to_head.csv")
J = pd.read_csv("outputs/ablation_blend_weights.csv").set_index("name")
K = pd.read_csv("outputs/ablation_blend_members.csv")

fig = plt.figure(figsize=(W, H_))
gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1], hspace=.42, wspace=.30,
                      left=.213, right=.985, top=.955, bottom=.072)
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# Panel A: the change from dropping (or adding) one feature group at a time.
ref = I["I0 all features (default)"]
d = (I.drop("I0 all features (default)") - ref).sort_values()
NAMES = {"HORIZON_M": "months-ahead feature", "ALL POI (10 cols)": "all POI (10 columns)",
         "ALL POI + coords": "all POI and coordinates", "full POI set (+20 cols)": "full POI set (20 columns)",
         "sec. schools": "secondary schools", "fine geo cats": "fine location categories",
         "MRT (open)": "MRT (open)", "MRT (planned)": "MRT (planned)"}
lbl = [("Drop " if n.startswith("I- ") else "Add ") + NAMES.get(n[3:], n[3:]) for n in d.index]
ax[0].barh(np.arange(len(d)), d, color=["#c1121f" if v > 1 else "#2a9d8f" if v < -1 else "#a8a8a8" for v in d])
for i, v in enumerate(d):
    ax[0].text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.2f}", va="center",
               ha="left" if v >= 0 else "right", fontsize=6.2)
ax[0].set_yticks(np.arange(len(d))); ax[0].set_yticklabels(lbl, fontsize=6.4)
ax[0].axvline(0, color="k", lw=.8)
ax[0].set_xlim(-4, 18)
ax[0].set_xlabel(f"Change in CV RMSE from {ref:.2f} (S$)")
ax[0].set_title("(a) Feature-Group Ablation")

# Panel B: HistGB against LightGBM and the blend on identical splits.
lab = [f"Fold {c}" if p == "rolling16" else f"Holdout {c}" for p, c in zip(H.protocol, H.cut)]
x = np.arange(len(H)); w = .27
for k, (col, name, c) in enumerate([("notebook", "HistGB", "#a8a8a8"),
                                    ("ours_lgbm", "LightGBM", "#5fa8d3"),
                                    ("ours_blend", "LightGBM + CatBoost", "#1b4965")]):
    ax[1].bar(x + (k - 1) * w, H[col], w, label=name, color=c)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=6.4)
ax[1].set_ylim(440, 600); ax[1].set_ylabel("RMSE (S$)"); ax[1].set_xlabel("Validation split (cut month)")
ax[1].legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.1,
             borderpad=.2, labelspacing=.25)
ax[1].set_title("(b) Models on Identical Splits")

# Panel C: candidates that win on the rolling folds but lose on the holdout.
ax[2].scatter(J["3way_roll16"], J["3way_hold12"], color="#1b4965", label="Index and block-info variants")
alt = K[~K.members.str.startswith("nb (shipped")]
ax[2].scatter(alt.roll16, alt.hold12, color="#e07a5f", marker="s", label="Alternative blend members")
s = J.loc["Jc_ft05_blockinfo"]
ax[2].scatter(s["3way_roll16"], s["3way_hold12"], s=160, facecolors="none", edgecolors="k", lw=1.5)
ax[2].annotate("Submitted blend", (s["3way_roll16"], s["3way_hold12"]), textcoords="offset points",
               xytext=(8, -3), fontsize=6.4)
ax[2].margins(.12)
ax[2].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[2].set_xlabel("Rolling-origin CV RMSE (S$)"); ax[2].set_ylabel("Month-39 holdout RMSE (S$)")
ax[2].legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.1,
             borderpad=.2, labelspacing=.25)
ax[2].set_title("(c) Rolling Folds Against Holdout")

for a in ax: a.grid(alpha=.3)
os.makedirs("figures", exist_ok=True)

# Nothing may run past the edge of the canvas, because the figure is placed at
# exactly this width and anything outside it is simply cut off.
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print(f"fig6 content bbox {bb.x0:+.3f} {bb.y0:+.3f} to {bb.x1:.3f} {bb.y1:.3f} in, canvas {W} x {H_}")
if bb.x0 < -0.005 or bb.y0 < -0.005 or bb.x1 > W + 0.005 or bb.y1 > H_ + 0.005:
    raise SystemExit("fig6 overflows the canvas, adjust the layout")

plt.savefig("figures/fig_ablation_detail.png", dpi=400)
print("wrote figures/fig_ablation_detail.png")
