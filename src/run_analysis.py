#!/usr/bin/env python3
"""
run_analysis.py — the shared-concentration test, end to end.

    python run_analysis.py              # full run, ~10 min
    python run_analysis.py --quick      # reduced B, ~1 min, for a smoke test
    python run_analysis.py --stage boot # run one stage only

Stages run in order and cache their outputs, so a later stage will build
whatever it needs if you have not run the earlier ones:

    shares    build the species x year share matrix   -> shares_gb_fall_1970_2008.csv
    fits      per-species Beta fits                   -> per_species_beta_fits.csv
    boot      the test itself (LR + null bootstrap)   -> boot_h0.pkl
    power     size and power curve                    -> power_curve.csv
    subsets   subset robustness checks                -> subset_results.csv
    gof       per-species marginal Beta goodness of fit -> per_species_gof.csv
    figure    the four-panel figure                   -> nu_homogeneity_test.png/.pdf

Needs fishglob.py, betagof.py and nu_test.py beside it. The first run downloads
the 12 MB NEUS file from GitHub into ~/.fishglob (override with FISHGLOB_CACHE).
If you have no network, drop the shipped shares_gb_fall_1970_2008.csv next to
this script and every stage except `shares` will work offline.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import nu_test as nt

from paths import DATA, OUTPUT, FIGURES

SHARES = os.path.join(DATA, "shares_gb_fall_1970_2008.csv")
BOOT = os.path.join(OUTPUT, "boot_h0.pkl")
FITS = os.path.join(OUTPUT, "per_species_beta_fits.csv")
POWER = os.path.join(OUTPUT, "power_curve.csv")
SUBSETS = os.path.join(OUTPUT, "subset_results.csv")
GOF = os.path.join(OUTPUT, "per_species_gof.csv")

SEED_BOOT, SEED_POWER, SEED_SUBSET, SEED_GOF = 20260922, 7, 11, 3


def rule(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------- shares ---
def _load_cached_shares():
    """Return the cached share matrix, or None with a reason if unusable.

    Existence is not enough: a cloud-sync placeholder, an interrupted download
    or a partial write all leave a file that `os.path.exists` is happy with and
    pandas is not. Anything unusable is reported and rebuilt from source.
    """
    if not os.path.exists(SHARES):
        return None, "not present"
    size = os.path.getsize(SHARES)
    if size < 100:
        return None, f"only {size} bytes — looks empty or unsynced"
    try:
        X = pd.read_csv(SHARES, index_col=0)
    except Exception as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if X.shape[0] < 5 or X.shape[1] < 2:
        return None, f"too small ({X.shape[0]} rows x {X.shape[1]} columns)"
    if X.isna().any().any():
        return None, "contains missing values"
    vals = X.to_numpy(dtype=float)
    if not ((vals > 0) & (vals < 1)).all():
        return None, "contains values outside the open interval (0, 1)"
    return X, None


def stage_shares(force=False):
    if not force:
        X, why = _load_cached_shares()
        if X is not None:
            print(f"[shares]  using existing {os.path.basename(SHARES)} "
                  f"({X.shape[0]} years x {X.shape[1]} species)")
            return X
        if why != "not present":
            print(f"[shares]  cached {os.path.basename(SHARES)} is unusable: {why}")
            print("[shares]  rebuilding it from the survey data")
    rule("STAGE shares — building the annual share matrix")
    try:
        X, rep = nt.build_shares()
    except Exception as exc:
        print(f"\n  could not build the share matrix: {type(exc).__name__}: {exc}\n")
        print("  This stage needs either a working copy of "
              f"{os.path.basename(SHARES)} or the ability to download the")
        print("  FishGlob survey file. Things to check:")
        print("    * the shipped CSV really has content — it should be about")
        print("      19 KB and 40 lines, not 0 bytes. A cloud-sync placeholder")
        print("      (Nextcloud, Dropbox, OneDrive) is the usual cause; force it")
        print("      to download, or copy the file to a plain local directory.")
        print("    * pyreadr is installed:  pip install pyreadr")
        print("    * GitHub is reachable, or set FISHGLOB_CACHE to a directory")
        print("      that already holds NEUS_clean_main.RData")
        raise SystemExit(1)
    for k, v in rep.items():
        if k == "dropped_zero_weight":
            print("  dropped for zero-weight cells (num > 0 but wgt == 0):")
            for s, yrs in v.items():
                print(f"     {s}: {yrs}")
        elif k == "share_sum_by_year":
            print("  share of community biomass covered:",
                  ", ".join(f"{i} {x:.3f}" for i, x in v.items()))
        else:
            print(f"  {k}: {v}")
    X.to_csv(SHARES)
    print(f"  -> {os.path.basename(SHARES)}  ({X.shape[0]} years x {X.shape[1]} species)")
    return X


# ------------------------------------------------------------------ fits ---
def stage_fits(X):
    rule("STAGE fits — per-species Beta fits")
    p, nu, ll1 = nt.fit_free(X.values)
    a = nt.estimate_mean_reversion(X.values)
    d = pd.DataFrame(
        {"mean_share": X.mean().values, "var_share": X.var(ddof=1).values,
         "alpha": nu * p, "beta": nu * (1 - p), "nu": nu,
         "cv": np.sqrt(X.var(ddof=1).values) / X.mean().values,
         "lag1_rho": np.exp(-a)},
        index=X.columns).sort_values("mean_share", ascending=False)
    d.round(5).to_csv(FITS)
    print(d.round(3).to_string())
    print(f"\n  nu: min {nu.min():.1f}  median {np.median(nu):.1f}  max {nu.max():.1f}"
          f"  (spread {nu.max()/nu.min():.0f}x)")
    print(f"  alpha = nu*p: min {(nu*p).min():.2f}  median {np.median(nu*p):.2f}"
          f"  max {(nu*p).max():.2f}")
    from scipy.stats import linregress
    lr = linregress(np.log10(d.mean_share), np.log10(d.var_share))
    print(f"\n  Taylor slope = {lr.slope:.3f} (SE {lr.stderr:.3f}), R2 = {lr.rvalue**2:.3f}")
    print(f"     neutral prediction is 1  -> {(lr.slope-1)/lr.stderr:.1f} SE away")
    print(f"     constant CV would be 2   -> {(lr.slope-2)/lr.stderr:.1f} SE away")
    print(f"  -> {os.path.basename(FITS)}")
    return d


# ------------------------------------------------------------------ boot ---
def stage_boot(X, n_boot):
    rule(f"STAGE boot — the test (B = {n_boot})")
    t0 = time.perf_counter()
    res = nt.bootstrap_lr(X.values, n_boot=n_boot, rng=SEED_BOOT)
    ln = res["lr_null"]
    print(f"  common nu under H0 : {res['nu_common']:.2f}")
    print(f"  observed LR        : {res['lr_obs']:.1f}")
    print(f"  null: median {np.median(ln):.1f}  95% {np.percentile(ln,95):.1f}"
          f"  99% {np.percentile(ln,99):.1f}  max {ln.max():.1f}")
    print(f"  bootstrap p        : {res['p_value']:.4f}   (floor {1/(n_boot+1):.4f})")
    from scipy.stats import chi2
    c95 = chi2.ppf(0.95, X.shape[1] - 1)
    print(f"\n  chi2_{X.shape[1]-1} 95% critical value = {c95:.1f}")
    print(f"  bootstrap 95% critical value = {np.percentile(ln,95):.1f}"
          f"  ({np.percentile(ln,95)/c95:.1f}x wider)")
    pickle.dump(res, open(BOOT, "wb"))
    print(f"\n  ({(time.perf_counter()-t0)/60:.1f} min)  -> {os.path.basename(BOOT)}")
    return res


# ----------------------------------------------------------------- power ---
def stage_power(X, boot, n_rep):
    rule(f"STAGE power — size and power ({n_rep} datasets per point)")
    crit = np.percentile(boot["lr_null"], 95)
    p0, nu0, a_hat = boot["p_common"], boot["nu_common"], boot["a_hat"]
    K, nyr = X.shape[1], X.shape[0]
    rng = np.random.default_rng(SEED_POWER)
    rows = []
    print(f"  reject when LR > {crit:.1f}")
    print("  sd(log10 nu)   median max/min nu   rejection rate at 5%")
    for sd in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.586):
        nus = np.full((n_rep, K), nu0) if sd == 0 else nu0 * 10 ** rng.normal(0, sd, (n_rep, K))
        sims = np.empty((n_rep, nyr, K))
        for j in range(K):
            sims[:, :, j] = nt.simulate_jacobi_vec(
                np.full(n_rep, p0[j]), nus[:, j], np.full(n_rep, a_hat[j]), nyr, rng)
        lrs = np.array([nt.lr_statistic(sims[b]) for b in range(n_rep)])
        rate = 100 * np.mean(lrs > crit)
        ratio = float(np.median(nus.max(axis=1) / nus.min(axis=1)))
        rows.append({"sd_log10_nu": sd, "ratio_max_min": ratio, "rejection_pct": rate})
        print(f"     {sd:.3f}          {ratio:8.1f}           {rate:5.1f}%", flush=True)
    pd.DataFrame(rows).to_csv(POWER, index=False)
    print(f"  -> {os.path.basename(POWER)}")


# --------------------------------------------------------------- subsets ---
def stage_subsets(X, boot, n_boot):
    rule(f"STAGE subsets — robustness (B = {n_boot} each)")
    order = X.mean().sort_values(ascending=False)
    feller = list(order.index[order.values > 1 / boot["nu_common"]])
    jobs = [
        (list(X.columns), "all species (reference)"),
        (feller, "only species meeting Feller u_i > 1"),
        (list(order.index[:10]), "top 10 by mean share"),
        (list(order.index[1:]), f"drop {order.index[0]}"),
        (list(order.index[-10:]), "bottom 10 by mean share"),
    ]
    rows = []
    print("  %-40s %3s %8s %9s %8s %8s" % ("subset", "K", "nu0", "LR", "null95", "p"))
    for cols, label in jobs:
        r = nt.bootstrap_lr(X[cols].values, n_boot=n_boot, rng=SEED_SUBSET)
        n95 = float(np.percentile(r["lr_null"], 95))
        print("  %-40s %3d %8.2f %9.1f %8.1f %8.4f"
              % (label, len(cols), r["nu_common"], r["lr_obs"], n95, r["p_value"]), flush=True)
        rows.append({"subset": label, "K": len(cols), "nu0": r["nu_common"],
                     "LR": r["lr_obs"], "null95": n95, "p": r["p_value"]})
    pd.DataFrame(rows).to_csv(SUBSETS, index=False)

    R = np.corrcoef(np.log(X.values).T)
    off = R[np.triu_indices_from(R, 1)]
    tot = X.sum(axis=1)
    print(f"\n  compositional dependence (unmodelled by the bootstrap):")
    print(f"    species carry {100*tot.min():.0f}%-{100*tot.max():.0f}% of community "
          f"biomass (median {100*tot.median():.0f}%)")
    print(f"    mean pairwise correlation of log shares {off.mean():.3f}; "
          f"{100*np.mean(np.abs(off)>0.5):.0f}% of pairs above |r| = 0.5")
    print(f"  -> {os.path.basename(SUBSETS)}")


# ------------------------------------------------------------------- gof ---
def stage_gof(X, n_boot):
    rule(f"STAGE gof — per-species marginal Beta fit (Tn, B = {n_boot})")
    from betagof import beta_gof_test
    rows = []
    for c in X.mean().sort_values(ascending=False).index:
        r = beta_gof_test(X[c].values, n_boot=n_boot, rng=SEED_GOF, include_edf=False)
        rows.append({"species": c, "alpha": r.a_hat, "beta": r.b_hat,
                     "nu": r.a_hat + r.b_hat, "Tn_p": r.p_values["Tn"]})
        print(f"  {c[:34]:<34s} nu {r.a_hat+r.b_hat:8.1f}   Tn p = {r.p_values['Tn']:.3f}"
              f"{' *' if r.p_values['Tn'] < 0.05 else ''}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(GOF, index=False)
    try:
        from scipy.stats import false_discovery_control as fdr
        n_bh = int((fdr(d.Tn_p.values) < 0.05).sum())
    except ImportError:                       # scipy < 1.11
        n_bh = None
    print(f"\n  rejected at 5%: {int((d.Tn_p < 0.05).sum())} of {len(d)}")
    if n_bh is not None:
        print(f"  after Benjamini-Hochberg: {n_bh} of {len(d)}")
    print("  (at n = 39 power is low, so non-rejection is weak evidence of fit)")
    print(f"  -> {os.path.basename(GOF)}")


# ---------------------------------------------------------------- figure ---
def stage_figure(X, boot):
    rule("STAGE figure")
    import make_figure
    make_figure.build(X, boot, FIGURES)


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="reduced replication for a fast smoke test")
    ap.add_argument("--stage", default="all",
                    choices=["all", "shares", "fits", "boot", "power", "subsets",
                             "gof", "figure"])
    ap.add_argument("--rebuild-shares", action="store_true")
    args = ap.parse_args()

    if args.quick:
        n_boot, n_sub, n_rep, n_gof = 99, 99, 25, 199
    else:
        n_boot, n_sub, n_rep, n_gof = 999, 499, 150, 999

    t0 = time.perf_counter()
    X = stage_shares(force=args.rebuild_shares)
    want = args.stage

    def need_boot():
        if os.path.exists(BOOT):
            return pickle.load(open(BOOT, "rb"))
        print("[boot]  no cached bootstrap, running it first")
        return stage_boot(X, n_boot)

    if want in ("all", "fits"):
        stage_fits(X)
    if want in ("all", "boot"):
        boot = stage_boot(X, n_boot)
    if want in ("all", "power"):
        boot = boot if want == "all" else need_boot()
        stage_power(X, boot, n_rep)
    if want in ("all", "subsets"):
        boot = boot if want == "all" else need_boot()
        stage_subsets(X, boot, n_sub)
    if want in ("all", "gof"):
        stage_gof(X, n_gof)
    if want in ("all", "figure"):
        boot = boot if want == "all" else need_boot()
        stage_figure(X, boot)

    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
