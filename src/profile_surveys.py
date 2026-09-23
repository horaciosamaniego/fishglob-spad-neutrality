"""Profile FishGlob surveys: gear history, coverage, species availability."""
import sys
import warnings

import numpy as np
import pandas as pd

import fishglob as fg
import os
from paths import OUTPUT

warnings.filterwarnings("ignore")

SURVEYS = ["NEUS", "SCS", "SEUS", "GMEX", "GSL-N", "GSL-S",
           "EBS", "GOA", "NS-IBTS", "EVHOE"]
FISH = ("Teleostei", "Elasmobranchii")


def gear_runs(h):
    """Dominant gear per year, and the longest consecutive single-gear run."""
    if "gear" not in h or h.gear.isna().all():
        yrs = sorted(h.year.unique())
        return {"gear": "n/a", "y0": min(yrs), "y1": max(yrs), "n_years": len(yrs)}
    dom = h.dropna(subset=["gear"]).groupby("year").gear.agg(
        lambda s: s.value_counts().idxmax())
    best = None
    i = 0
    yrs = list(dom.index)
    while i < len(yrs):
        j = i
        while (j + 1 < len(yrs) and dom.iloc[j + 1] == dom.iloc[i]
               and yrs[j + 1] == yrs[j] + 1):
            j += 1
        run = {"gear": dom.iloc[i], "y0": yrs[i], "y1": yrs[j], "n_years": j - i + 1}
        if best is None or run["n_years"] > best["n_years"]:
            best = run
        i = j + 1
    return best


rows = []
for code in SURVEYS:
    try:
        df = fg.load_survey(code)
    except Exception as exc:
        print(f"{code}: FAILED ({type(exc).__name__}: {exc})", flush=True)
        continue
    for unit in sorted(df.survey_unit.dropna().unique()):
        sub = df[df.survey_unit == unit]
        h = fg.haul_table(sub)
        run = gear_runs(h)
        win = sub[(sub.year >= run["y0"]) & (sub.year <= run["y1"])]
        hw = fg.haul_table(win)
        s = win[(win["rank"] == "Species") & win["class"].isin(FISH)]
        tot = s.groupby("year").wgt_cpua.sum()
        sp = s.groupby(["accepted_name", "year"]).wgt_cpua.sum()
        share = (sp / tot).unstack(0).reindex(range(run["y0"], run["y1"] + 1))
        full = share.columns[share.notna().all(axis=0)]
        interior = [c for c in full if ((share[c] > 0) & (share[c] < 1)).all()]
        per_year = hw.groupby("year").size()
        rows.append({
            "survey": code, "unit": unit,
            "all_years": f"{h.year.min()}-{h.year.max()}",
            "gear": str(run["gear"]), "window": f"{run['y0']}-{run['y1']}",
            "n_years": run["n_years"], "hauls": len(hw),
            "hauls_per_yr": int(per_year.median()) if len(per_year) else 0,
            "strata": int(hw.stratum.nunique()) if "stratum" in hw else 0,
            "sp_any": int(share.notna().any().sum()),
            "sp_full": len(full), "sp_usable": len(interior),
        })
        print(f"  {code:9s} {unit:16s} gear {str(run['gear']):>6s}  "
              f"{run['y0']}-{run['y1']} ({run['n_years']:2d} yr)  "
              f"{len(hw):5d} hauls  usable species {len(interior):3d}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUTPUT, "survey_profile.csv"), index=False)
print()
print("=" * 100)
print("VIABLE for the test (>= 15 years and >= 8 usable species), best window per survey unit")
print("=" * 100)
v = d[(d.n_years >= 15) & (d.sp_usable >= 8)].sort_values("sp_usable", ascending=False)
print(v.to_string(index=False))
