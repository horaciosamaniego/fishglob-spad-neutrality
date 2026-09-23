"""Four-panel figure for the shared-concentration test.

Importable (`make_figure.build(X, boot, outdir)`) or runnable on its own, in
which case it loads the cached share matrix and bootstrap from the current
directory. Power values are read from power_curve.csv when present, so the
figure always matches the run that produced it.
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

import nu_test as nt
from paths import DATA, OUTPUT, FIGURES

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


def build(X, boot, outdir=None):
    outdir = outdir or FIGURES
    p, nu, _ = nt.fit_free(X.values)
    nu0 = boot["nu_common"]
    m, v = X.mean().values, X.var(ddof=1).values
    cols = list(X.columns)
    lo_i, hi_i = int(np.argmin(nu)), int(np.argmax(nu))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.09, right=0.97,
                        top=0.92, bottom=0.08)

    # ---- A: nu vs mean share --------------------------------------------
    ax = axes[0, 0]
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(nu0, color=ORANGE, lw=2, zorder=2,
               label=r"neutral prediction: shared $\nu$ = %.0f" % nu0)
    xs = np.logspace(np.log10(m.min() * 0.6), np.log10(m.max() * 1.6), 50)
    ax.plot(xs, np.median(nu * p) / xs, color=MUTED, lw=1.2, ls=":", zorder=2,
            label=r"constant $\alpha$: $\nu \propto 1/\bar{Z}$")
    ax.scatter(m, nu, s=34, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"mean annual share  $\bar{Z}_i$")
    ax.set_ylabel(r"fitted concentration  $\hat{\nu}_i$")
    ax.set_title(r"A   Concentration is not shared: $\hat{\nu}$ spans %.0f$\times$"
                 % (nu.max() / nu.min()), loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=SEC)
    for i, dx, dy in ((hi_i, 6, -10), (lo_i, -8, -15)):
        ax.annotate(cols[i], (m[i], nu[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color=SEC, style="italic",
                    ha="left" if dx > 0 else "right")

    # ---- B: Taylor's law -------------------------------------------------
    ax = axes[0, 1]
    ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    lr = linregress(np.log10(m), np.log10(v))
    xs = np.linspace(np.log10(m.min()) - 0.25, np.log10(m.max()) + 0.25, 50)
    c_neutral = np.mean(np.log10(v)) - 1.0 * np.mean(np.log10(m))
    ax.plot(10 ** xs, 10 ** (c_neutral + 1.0 * xs), color=ORANGE, lw=2, zorder=2,
            label="neutral: slope 1")
    ax.plot(10 ** xs, 10 ** (lr.intercept + lr.slope * xs), color=BLUE, lw=2,
            ls="--", zorder=3, label="observed: slope %.2f" % lr.slope)
    ax.scatter(m, v, s=34, color=BLUE, zorder=4, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"mean annual share  $\bar{Z}_i$")
    ax.set_ylabel("variance of annual share")
    ax.set_title("B   Taylor's law: exponent %.2f, not 1" % lr.slope, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=SEC)

    # ---- C: bootstrap null vs observed ----------------------------------
    ax = axes[1, 0]
    ax.grid(True, color=GRID, lw=0.6, axis="y", zorder=0); ax.set_axisbelow(True)
    ln = boot["lr_null"]
    hi = max(boot["lr_obs"], ln.max()) * 1.6
    ax.hist(ln, bins=np.logspace(np.log10(ln.min() * 0.8), np.log10(hi), 45),
            color=BLUE, alpha=0.85, zorder=2)
    ymax = ax.get_ylim()[1]
    ax.axvline(boot["lr_obs"], color=ORANGE, lw=2.2, zorder=4)
    ax.annotate("observed\nLR = %.0f" % boot["lr_obs"], (boot["lr_obs"], ymax * 0.55),
                textcoords="offset points", xytext=(-8, 0), ha="right",
                fontsize=8, color=ORANGE, weight="medium")
    n95 = np.percentile(ln, 95)
    ax.axvline(n95, color=MUTED, lw=1.2, ls=":", zorder=3)
    ax.annotate("95%% of null = %.0f" % n95, (n95, ymax * 0.70),
                textcoords="offset points", xytext=(7, 0), fontsize=7.5, color=SEC)
    ax.annotate("null distribution\nunder shared $\\nu$ (B = %d)" % len(ln),
                (np.median(ln), ymax * 0.93), ha="center", fontsize=7.5, color=SEC)
    ax.set_xscale("log")
    ax.set_xlabel("likelihood-ratio statistic (log scale)")
    ax.set_ylabel("bootstrap replicates")
    ax.set_title("C   Observed LR is %.0f$\\times$ the largest of %d null draws"
                 % (boot["lr_obs"] / ln.max(), len(ln)), loc="left")

    # ---- D: power --------------------------------------------------------
    ax = axes[1, 1]
    ax.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    pc = os.path.join(OUTPUT, "power_curve.csv")
    if os.path.exists(pc):
        d = pd.read_csv(pc)
        sd, pw = d.sd_log10_nu.values, d.rejection_pct.values
        obs_sd = float(np.std(np.log10(nu)))
        i15 = int(np.argmin(np.abs(sd - 0.15)))
        ax.plot(sd, pw, color=BLUE, lw=2, marker="o", ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.axhline(5, color=MUTED, lw=1, ls=":", zorder=2)
        ax.annotate("nominal 5% size", (sd.max() * 0.55, 8), fontsize=7.5, color=SEC)
        ax.axvline(obs_sd, color=ORANGE, lw=2, zorder=2)
        ax.annotate("observed\nspread", (obs_sd, 45), textcoords="offset points",
                    xytext=(-8, 0), ha="right", fontsize=8, color=ORANGE,
                    weight="medium")
        ax.set_ylim(-4, 106)
        ax.set_title("D   Power: detects a %.0f$\\times$ spread %.0f%% of the time"
                     % (d.ratio_max_min.values[i15], pw[i15]), loc="left")
    else:
        ax.text(0.5, 0.5, "run the power stage to fill this panel",
                ha="center", va="center", fontsize=8, color=SEC,
                transform=ax.transAxes)
        ax.set_title("D   Power", loc="left")
    ax.set_xlabel(r"heterogeneity in $\nu$  (sd of $\log_{10}\nu$)")
    ax.set_ylabel("rejection rate at 5% (%)")

    fig.suptitle("Shared-concentration test  ·  Georges Bank autumn, 1970–2008  ·  "
                 "%d species × %d years" % (X.shape[1], X.shape[0]),
                 fontsize=10.5, x=0.09, ha="left", y=0.975, color=INK)
    png = os.path.join(outdir, "nu_homogeneity_test.png")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, "nu_homogeneity_test.pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Taylor slope = %.3f (SE %.3f)" % (lr.slope, lr.stderr))
    print("  -> nu_homogeneity_test.png / .pdf")
    return lr


if __name__ == "__main__":
    X = pd.read_csv(os.path.join(DATA, "shares_gb_fall_1970_2008.csv"), index_col=0)
    boot = pickle.load(open(os.path.join(OUTPUT, "boot_h0.pkl"), "rb"))
    build(X, boot, FIGURES)
