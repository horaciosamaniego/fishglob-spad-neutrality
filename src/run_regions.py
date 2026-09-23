"""The shared-concentration test across FishGlob surveys.

Window selection is data-driven: for each survey unit, the longest run of
consecutive years sharing one dominant gear (from survey_profile.csv). That
avoids needing survey-specific gear-change expertise, and is reported so the
choice is auditable.

Spatial extent is the whole survey unit, not a stratum subset, so the regions
are comparable to each other. The Georges Bank result is carried as an extra
row for continuity with the pilot.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress

import fishglob as fg
import nu_test as nt
import os
from paths import OUTPUT

warnings.filterwarnings("ignore")

B = 399
JOBS = [
    # survey, unit, window, strata, basin, label
    ("NEUS", "NEUS-Fall",   (1970, 2008), fg.GEORGES_BANK_STRATA, "NW Atlantic", "NEUS-Fall (Georges Bank)"),
    ("NEUS", "NEUS-Fall",   (1963, 2008), "all", "NW Atlantic", "NEUS-Fall (whole shelf)"),
    ("NEUS", "NEUS-Spring", (1982, 2008), "all", "NW Atlantic", "NEUS-Spring"),
    ("SCS",  "SCS-SUMMER",  (1982, 2020), "all", "NW Atlantic", "Scotian Shelf summer"),
    ("SCS",  "SCS-SPRING",  (1979, 2020), "all", "NW Atlantic", "Scotian Shelf spring"),
    ("GSL-S", "GSL-S",      (1987, 2002), "all", "NW Atlantic", "S Gulf of St Lawrence"),
    ("SEUS", "SEUS-fall",   (1989, 2019), "all", "NW Atlantic", "SE US fall"),
    ("SEUS", "SEUS-spring", (1989, 2019), "all", "NW Atlantic", "SE US spring"),
    ("SEUS", "SEUS-summer", (1989, 2019), "all", "NW Atlantic", "SE US summer"),
    ("GMEX", "GMEX-Summer", (1982, 2019), "all", "Gulf of Mexico", "Gulf of Mexico summer"),
    ("GMEX", "GMEX-Fall",   (1983, 2024), "all", "Gulf of Mexico", "Gulf of Mexico fall"),
    ("NS-IBTS", "NS-IBTS-3", (1991, 2020), "all", "NE Atlantic", "North Sea Q3"),
    ("EVHOE", "EVHOE",      (1997, 2020), "all", "NE Atlantic", "Bay of Biscay"),
    ("EBS",  "EBS",         (1982, 2019), "all", "N Pacific", "E Bering Sea"),
]

rows = []
t_all = time.perf_counter()
for survey, unit, years, strata, basin, label in JOBS:
    t0 = time.perf_counter()
    try:
        X, rep = nt.build_shares(survey=survey, survey_unit=unit,
                                 strata=strata, years=years)
    except Exception as exc:
        print(f"{label}: SKIPPED ({type(exc).__name__}: {exc})", flush=True)
        continue
    if X.shape[1] < 8:
        print(f"{label}: SKIPPED (only {X.shape[1]} usable species)", flush=True)
        continue

    p, nu, _ = nt.fit_free(X.values)
    m, v = X.mean().values, X.var(ddof=1).values
    tl = linregress(np.log10(m), np.log10(v))
    res = nt.bootstrap_lr(X.values, n_boot=B, rng=20260922)
    n95 = float(np.percentile(res["lr_null"], 95))
    alpha = nu * p
    feller = float(np.mean(res["nu_common"] * p > 1))

    rows.append({
        "label": label, "basin": basin, "survey": survey, "unit": unit,
        "window": f"{years[0]}-{years[1]}", "T": X.shape[0], "K": X.shape[1],
        "nu0": res["nu_common"], "nu_min": nu.min(), "nu_max": nu.max(),
        "nu_spread": nu.max() / nu.min(), "sd_log10_nu": float(np.std(np.log10(nu))),
        "alpha_med": float(np.median(alpha)),
        "sd_log10_alpha": float(np.std(np.log10(alpha))),
        "taylor_slope": tl.slope, "taylor_se": tl.stderr, "taylor_r2": tl.rvalue ** 2,
        "LR": res["lr_obs"], "null95": n95, "p": res["p_value"],
        "reject": res["lr_obs"] > n95, "feller_frac": feller,
    })
    print(f"  {label:28s} K={X.shape[1]:3d} T={X.shape[0]:2d}  "
          f"Taylor {tl.slope:.2f}+-{tl.stderr:.2f}   "
          f"nu {nu.min():.1f}-{nu.max():.0f} ({nu.max()/nu.min():.0f}x)   "
          f"LR {res['lr_obs']:8.1f} vs null95 {n95:6.1f}   p={res['p_value']:.4f}   "
          f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUTPUT, "region_results.csv"), index=False)
print(f"\ntotal {(time.perf_counter()-t_all)/60:.1f} min; wrote output/region_results.csv")
print(f"\nrejected the shared concentration in {int(d.reject.sum())} of {len(d)} survey units")
print(f"Taylor slope: min {d.taylor_slope.min():.2f}  median {d.taylor_slope.median():.2f}"
      f"  max {d.taylor_slope.max():.2f}")
print(f"how many slopes are within 2 SE of the neutral value 1?  "
      f"{int((np.abs(d.taylor_slope - 1) < 2*d.taylor_se).sum())} of {len(d)}")
