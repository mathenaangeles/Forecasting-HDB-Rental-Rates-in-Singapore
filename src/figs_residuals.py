"""Residual diagnostics by forecast horizon, town, and flat type."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False})

H = pd.read_csv("outputs/residual_by_horizon.csv")
S = pd.read_csv("outputs/residual_slices.csv")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))

# Panel (a): prediction bias varies by forecast origin and horizon. Showing
# each fold alongside the pooled summary retains both sources of variation
# without attributing them to a single mechanism.
LABELS = ["1-4", "5-8", "9-12", "13-16"]
folds = [c for c in H.cut.unique() if c != "rolling"]
x = np.arange(len(LABELS)); w = .19
colors = ["#1b4965", "#5fa8d3", "#e07a5f", "#9c6644"]
for k, cut in enumerate(folds):
    g = H[H.cut == cut].set_index("horizon").reindex(LABELS)
    name = f"Fold {cut}" if str(cut) != "39" else "Confirmation 39"
    ax[0].bar(x + (k - 1.5) * w, g.bias.fillna(0), w, label=name, color=colors[k])
r = H[H.cut == "rolling"].set_index("horizon").reindex(LABELS)
ax[0].plot(x, r.bias, "k--o", lw=1.6, ms=5, label="Three folds pooled", zorder=4)
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xticks(x); ax[0].set_xticklabels(LABELS)
ax[0].set_xlabel("Months ahead of the cut")
ax[0].set_ylabel("Mean error, prediction minus actual (S$)")
ax[0].set_title("(a) Bias by Horizon, One Fold at a Time")
ax[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
ax[0].set_ylim(-80, 110)
ax[0].text(0.0, -0.21, "Bias varies across forecast origins and horizons.",
           transform=ax[0].transAxes, fontsize=7.5, color="#555")

# Panel (b): all-row slice RMSE against estimated within-group noise SD on
# repeated-predictor rows. Equality does not prove that further gains are
# impossible, and the repeated rows may not represent the whole slice.
d = S.dropna(subset=["noise_sd"])
for by, m, c, lab in [("TOWN", "o", "#1b4965", "Town"), ("FLAT_TYPE", "s", "#e07a5f", "Flat type")]:
    g = d[d.by == by]
    ax[1].scatter(g.noise_sd, g.rmse, marker=m, color=c, s=34, label=lab, zorder=3)
lo, hi = 380, 780
ax[1].plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.2, zorder=1)
ax[1].text(620, 638, "Equal RMSE and noise SD", fontsize=7.5, color="#555",
           rotation=45, rotation_mode="anchor", ha="left", va="bottom")
# Only the extremes are labelled, because the cluster in the middle is dense
# and the point of the panel is that the slices lie on the diagonal.
named = pd.concat([d.nlargest(3, "rmse"), d.nsmallest(2, "rmse")])
for _, row in named.iterrows():
    ax[1].annotate(str(row.value), (row.noise_sd, row.rmse), textcoords="offset points",
                   xytext=(7, -3), fontsize=7)
ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
ax[1].set_aspect("equal")
ax[1].set_xlabel("Estimated within-group noise SD (S$)")
ax[1].set_ylabel("RMSE of the slice (S$)")
ax[1].set_title("(b) RMSE and Estimated Within-Group Noise")
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left")
worst = (d.rmse / d.noise_sd).max()
ax[1].text(0.0, -0.21, f"No slice exceeds its estimated noise SD by more than {(worst - 1) * 100:.0f} percent.",
           transform=ax[1].transAxes, fontsize=7.5, color="#555")

for a in ax: a.grid(alpha=.3)
os.makedirs("figures", exist_ok=True)
plt.tight_layout(); plt.savefig("figures/fig_residual_analysis.png", dpi=190)
print("wrote figures/fig_residual_analysis.png")
