"""Model ladder and error budget: how much model choice and better features can buy."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False,
                     "axes.spines.right": False})

# Both panels read only committed outputs, so the figure needs no model refit.
L = pd.read_csv("outputs/ladder_metrics.csv")
B = pd.read_csv("outputs/error_budget.csv")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})

# Panel (a): every model class, untuned, on the same splits. The three setups
# differ only in how time is handled, which is the comparison that matters.
NAMES = {"median": "Town and flat-type median", "ridge": "Ridge", "knn": "k-nearest neighbours",
         "tree": "Decision tree", "bag": "Bagged trees", "rf": "Random forest",
         "ada": "AdaBoost", "mlp": "Neural network (MLP)", "xgb": "XGBoost",
         "lgbm": "LightGBM", "cat": "CatBoost"}
SETUPS = [("index", "Detrended by the index", "#1b4965"),
          ("raw_tm", "Raw rent, month counter as a feature", "#5fa8d3"),
          ("raw", "Raw rent, no time feature", "#c9cdd1")]
piv = L.pivot(index="model", columns="setup", values="roll16_rmse")
order = piv["index"].sort_values(ascending=False).index
y = np.arange(len(order)); h = .26
for k, (setup, lab, c) in enumerate(SETUPS):
    v = piv.loc[order, setup]
    ax[0].barh(y + (1 - k) * h, v, h, label=lab, color=c)
for i, m in enumerate(order):
    ax[0].text(piv.loc[m, "index"] + 6, i + h, f"{piv.loc[m, 'index']:.0f}",
               va="center", fontsize=7.5, color="#1b4965")
ax[0].set_yticks(y); ax[0].set_yticklabels([NAMES[m] for m in order], fontsize=8)
ax[0].set_xlim(0, 1300)
ax[0].set_xlabel("Rolling-origin CV RMSE (S$), untuned")
ax[0].set_title("(a) Every Model Class on One Simplified Setup")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper right")
# The spread across model classes is small next to the spread across setups.
spread = piv.loc[order, "index"].max() - piv.loc[order, "index"].min()
ax[0].text(0.0, -0.22,
           f"Within the index setup the {len(order)} models span only {spread:.0f} RMSE.",
           transform=ax[0].transAxes, fontsize=7.5, color="#555")

# Panel (b): what is left to win. On rows that have a feature-identical twin,
# the achieved error, the error with a perfectly forecast index, and the spread
# among the twins themselves are almost the same number.
x = np.arange(len(B)); w = .26
# All three bars must describe the same rows, so the oracle error is recovered
# on the twin rows from the share of their MSE that the oracle removes.
B["twin_rmse_oracle"] = B.twin_rmse * np.sqrt(1 - B.forecast_share)
bars = [("twin_rmse", "Shipped model", "#1b4965"),
        ("twin_rmse_oracle", "With a perfect index forecast", "#5fa8d3"),
        ("twin_noise_sd", "Spread among identical rows", "#e07a5f")]
for k, (col, lab, c) in enumerate(bars):
    ax[1].bar(x + (k - 1) * w, B[col], w, label=lab, color=c)
    for i, v in enumerate(B[col]):
        ax[1].text(i + (k - 1) * w, v + 4, f"{v:.0f}", ha="center", fontsize=6.8)
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"Fold {c}" if c != 39 else "Holdout 39" for c in B.cut], fontsize=8)
ax[1].set_ylim(400, 620)
ax[1].set_ylabel("RMSE (S$)")
ax[1].set_xlabel("Validation split (cut month)")
ax[1].set_title("(b) The Error Is Almost All Irreducible Noise")
ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
ax[1].text(0.0, -0.22, "Rows with an identical twin only. The axis starts at 400.",
           transform=ax[1].transAxes, fontsize=7.5, color="#555")

for a in ax:
    a.grid(axis="x" if a is ax[0] else "y", alpha=.3)
os.makedirs("figures", exist_ok=True)
plt.tight_layout(); plt.savefig("figures/fig_model_ladder.png", dpi=190)
print("wrote figures/fig_model_ladder.png")
