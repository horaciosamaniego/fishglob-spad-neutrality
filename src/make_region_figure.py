"""Cross-region comparison figure."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from paths import OUTPUT, FIGURES

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9.5, "axes.titleweight": "medium",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

d = pd.read_csv(os.path.join(OUTPUT, "region_results.csv"))
order = ["NW Atlantic", "Gulf of Mexico", "NE Atlantic", "N Pacific"]
d["b"] = pd.Categorical(d.basin, categories=order, ordered=True)
d = d.sort_values(["b", "taylor_slope"]).reset_index(drop=True)

# insert a gap row between basins
ypos, labels, seps = [], [], []
y = 0.0
for i, r in d.iterrows():
    if i > 0 and r.basin != d.basin[i - 1]:
        seps.append(y + 0.5)
        y += 1.0
    ypos.append(y)
    labels.append(r.label)
    y += 1.0
ypos = np.array(ypos)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(wspace=0.06, left=0.20, right=0.975, top=0.87, bottom=0.10)

# ---- A: Taylor slopes -----------------------------------------------------
ax = axes[0]
ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.axvline(1.0, color=ORANGE, lw=2.2, zorder=2)
ax.axvline(2.0, color=MUTED, lw=1.2, ls=":", zorder=2)
ax.errorbar(d.taylor_slope, ypos, xerr=1.96 * d.taylor_se, fmt="o", ms=5.5,
            color=BLUE, ecolor=BLUE, elinewidth=1.6, capsize=2.6, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8, color=INK)
ax.set_ylim(ypos.max() + 1.0, ypos.min() - 1.0)
ax.set_xlim(0.86, 2.22)
ax.set_xlabel("Taylor exponent  (slope of $\\log$ variance on $\\log$ mean)")
ax.set_title("A   Every region scales near 2, none near the neutral value of 1", loc="left")
ax.annotate("neutral prediction", (1.0, ypos.max() + 0.62), ha="center", va="center",
            fontsize=7.5, color=ORANGE, weight="medium")
ax.annotate("constant CV", (2.0, ypos.max() + 0.62), ha="center", va="center",
            fontsize=7.5, color=SEC)
for s in seps:
    ax.axhline(s, color=GRID, lw=0.8, zorder=1)
for b in order:
    rows = np.where(d.basin.values == b)[0]
    if len(rows):
        ax.annotate(b, (0.895, ypos[rows].min() - 0.42), fontsize=7.5,
                    color=SEC, style="italic", va="center")

# ---- B: nu spread vs alpha spread ----------------------------------------
ax = axes[1]
ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
for yy, (a, n) in zip(ypos, zip(d.sd_log10_alpha, d.sd_log10_nu)):
    ax.plot([a, n], [yy, yy], color=GRID, lw=2.4, zorder=2, solid_capstyle="round")
ax.scatter(d.sd_log10_nu, ypos, s=36, color=BLUE, zorder=4,
           edgecolor="white", linewidth=0.8)
ax.scatter(d.sd_log10_alpha, ypos, s=36, color=ORANGE, zorder=4,
           edgecolor="white", linewidth=0.8)
ax.set_yticks(ypos); ax.set_yticklabels([])
ax.set_ylim(ypos.max() + 1.0, ypos.min() - 1.0)
ax.set_xlim(0, 1.22)
ax.set_xlabel("spread across species  (sd of $\\log_{10}$)")
ax.set_title("B   $\\alpha$ is always the tight parameter, $\\nu$ the loose one", loc="left")
for s in seps:
    ax.axhline(s, color=GRID, lw=0.8, zorder=1)
i0 = 0
ax.annotate(r"$\alpha_i = \nu_i \bar{Z}_i$", (d.sd_log10_alpha[i0], ypos.max() + 0.62),
            ha="center", fontsize=8, color=ORANGE, weight="medium")
ax.annotate(r"$\nu_i$", (d.sd_log10_nu[i0], ypos.max() + 0.62), ha="center",
            fontsize=8, color=BLUE, weight="medium")

fig.suptitle("Shared-concentration test across 14 FishGlob survey units  ·  "
             "rejected in 14 of 14",
             fontsize=11, x=0.20, ha="left", y=0.955, color=INK)
fig.savefig(os.path.join(FIGURES, "nu_test_across_regions.png"), dpi=200,
            bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(FIGURES, "nu_test_across_regions.pdf"),
            bbox_inches="tight", facecolor="white")
print("saved. slopes %.2f-%.2f, median %.2f" %
      (d.taylor_slope.min(), d.taylor_slope.max(), d.taylor_slope.median()))
