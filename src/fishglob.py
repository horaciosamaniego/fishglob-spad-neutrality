"""
fishglob — reproducible access to the FishGlob bottom-trawl database from Python.

FishGlob distributes one ``<SURVEY>_clean.RData`` file per survey from the
GitHub repository fishglob/FishGlob_data. This module downloads them, caches
them, converts to Parquet once, and rebuilds the haul-by-taxon matrix with the
explicit zeros that the source format omits.

Data citation and reuse policy:
    Maureaud, A. et al. (2024). An integrated database of fish biodiversity
    sampled with scientific bottom trawl surveys. Scientific Data 11, 24.
    https://doi.org/10.1038/s41597-023-02866-w
    Repository: https://github.com/fishglob/FishGlob_data  (CC BY 4.0)
    Read the disclaimer and citation policy before publishing; FishGlob asks to
    be told which datasets are used so that data providers get credit.

Three facts about the format that determine everything downstream
-----------------------------------------------------------------
1.  The tables are PRESENCE-ONLY LONG. A row exists only where a taxon was
    caught in a haul. For NEUS that is 417k rows against 36,481 x 490 possible
    haul-taxon combinations, so roughly 98% of the matrix is structural zero.
    ``abundance_matrix`` reinstates those zeros; nothing else will.

2.  Effort standardisation. ``area_swept`` is complete for NEUS, ``haul_dur``
    is not (~80%). Prefer the per-area columns ``wgt_cpua`` / ``num_cpua``
    over the per-hour ``_cpue`` columns.

3.  Seasons are separate surveys. ``survey_unit`` splits NEUS into
    ``NEUS-Spring`` and ``NEUS-Fall``. They differ in gear-season interaction
    and species availability, so never pool them.

Known limitation
----------------
The per-survey files carry no flag columns. The spatio-temporal trimming flags
(``flag_trimming_hex7_0`` etc.) live only in the 88 MB compiled file
``outputs/Compiled_data/FishGlob_public_std_clean.RData``, which currently
fails to load in ``pyreadr`` with a UnicodeDecodeError (non-UTF-8 bytes in the
verbatim taxon names). Until that is resolved, either convert that file once
in R, or define the footprint yourself with ``footprint_summary`` below, which
is transparent and reproducible in Python.

Requires: pandas, numpy, pyreadr, pyarrow.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SURVEYS",
    "fetch_survey",
    "load_survey",
    "haul_table",
    "subset",
    "abundance_matrix",
    "footprint_summary",
    "GEORGES_BANK_STRATA",
    "GULF_OF_MAINE_STRATA",
]

_RAW = "https://raw.githubusercontent.com/FishGlob/FishGlob_data/{ref}/outputs/Cleaned_data/{code}_clean.RData"

#: Survey codes relevant to the North-West Atlantic. Full list in
#: standard_formats/Surveys_ID.xlsx of the repository.
SURVEYS = {
    "NEUS": "Northeast US shelf (NOAA NEFSC, via OceanAdapt)",
    "SCS": "Scotian Shelf (DFO, via OceanAdapt)",
    "SEUS": "Southeast US",
    "GMEX": "Gulf of Mexico",
    "GSL-N": "Northern Gulf of St Lawrence",
    "GSL-S": "Southern Gulf of St Lawrence",
    "DFO-NF": "Newfoundland",
}

#: NEFSC offshore strata. 1130-1250 is Georges Bank, the classic well-sampled
#: offshore bank; 1260-1400 is the Gulf of Maine basin.
GEORGES_BANK_STRATA = tuple(range(1130, 1260, 10))
GULF_OF_MAINE_STRATA = tuple(range(1260, 1410, 10))


def _cache_dir(cache_dir=None) -> Path:
    p = Path(cache_dir or os.environ.get("FISHGLOB_CACHE", "~/.fishglob")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_survey(code: str, ref: str = "main", cache_dir=None, force: bool = False) -> Path:
    """Download ``<code>_clean.RData`` and return the local path.

    ``ref`` is a git ref. Pin it to a release tag (e.g. ``"v2.1.0"``) for a
    reproducible pipeline; ``"main"`` is a moving target.
    """
    code = code.upper()
    dest = _cache_dir(cache_dir) / f"{code}_clean_{ref}.RData"
    if dest.exists() and not force:
        return dest
    url = _RAW.format(ref=ref, code=code)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def load_survey(code: str, ref: str = "main", cache_dir=None, force: bool = False) -> pd.DataFrame:
    """Load a survey as a DataFrame, caching a Parquet copy of the RData.

    The RData round-trip through ``pyreadr`` costs several seconds; the Parquet
    cache turns subsequent loads into a fraction of a second.
    """
    code = code.upper()
    pq = _cache_dir(cache_dir) / f"{code}_clean_{ref}.parquet"
    if pq.exists() and not force:
        return pd.read_parquet(pq)

    import pyreadr  # imported lazily so the module works from Parquet alone

    path = fetch_survey(code, ref=ref, cache_dir=cache_dir, force=force)
    objects = pyreadr.read_r(str(path))
    key = "data" if "data" in objects else next(iter(objects))
    df = objects[key]
    df.to_parquet(pq, index=False)
    return df


def haul_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per haul: the sampling design, stripped of the taxon records."""
    cols = [
        "survey", "survey_unit", "haul_id", "year", "month", "quarter", "season",
        "latitude", "longitude", "depth", "stratum", "station", "sub_area",
        "haul_dur", "area_swept", "gear", "sbt", "sst",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates("haul_id").reset_index(drop=True)


def subset(
    df: pd.DataFrame,
    survey_unit: str | None = None,
    strata=None,
    years: tuple[int, int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    depth: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Filter records by season, stratum, year range, bounding box or depth.

    ``bbox`` is ``(lat_min, lat_max, lon_min, lon_max)``.
    """
    m = pd.Series(True, index=df.index)
    if survey_unit is not None:
        m &= df["survey_unit"] == survey_unit
    if strata is not None:
        m &= df["stratum"].isin(list(strata))
    if years is not None:
        m &= df["year"].between(*years)
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        m &= df["latitude"].between(la0, la1) & df["longitude"].between(lo0, lo1)
    if depth is not None:
        m &= df["depth"].between(*depth)
    return df.loc[m].copy()


def abundance_matrix(
    df: pd.DataFrame,
    value: str = "wgt_cpua",
    rank: str | None = "Species",
    classes=None,
    min_occurrences: int = 0,
) -> pd.DataFrame:
    """Hauls x taxa matrix with the structural zeros reinstated.

    Parameters
    ----------
    value : column to place in the cells. ``wgt_cpua`` (kg/km2) and
        ``num_cpua`` (individuals/km2) are effort-standardised; ``wgt`` and
        ``num`` are raw catch.
    rank : keep only this WoRMS rank. The FishGlob tables mix species-level
        records with family- and genus-level bins for individuals that could
        not be identified further. Leaving those in double-counts biomass.
    classes : optional iterable of WoRMS classes to keep, e.g.
        ``("Teleostei", "Elasmobranchii")``.
    min_occurrences : drop taxa present in fewer than this many hauls.

    Notes
    -----
    Every haul in ``df`` becomes a row, including hauls whose only records were
    filtered out by ``rank`` or ``classes``; those rows are all-zero, which is
    the ecologically correct representation of an empty haul.
    """
    recs = df
    if rank is not None:
        recs = recs[recs["rank"] == rank]
    if classes is not None:
        recs = recs[recs["class"].isin(list(classes))]

    all_hauls = pd.Index(df["haul_id"].unique(), name="haul_id")
    mat = (
        recs.pivot_table(
            index="haul_id", columns="accepted_name", values=value,
            aggfunc="sum", fill_value=0.0,
        )
        .reindex(all_hauls, fill_value=0.0)
        .astype(float)
    )
    if min_occurrences > 0:
        mat = mat.loc[:, (mat > 0).sum(axis=0) >= min_occurrences]
    mat.columns.name = "accepted_name"
    return mat


def footprint_summary(hauls: pd.DataFrame, by: str = "stratum") -> pd.DataFrame:
    """How consistently each spatial unit was sampled through time.

    A defensible, transparent stand-in for FishGlob's trimming flags: keep the
    units sampled in nearly every year of the period you intend to analyse, so
    that changes in the data reflect the ocean rather than the survey design.
    """
    n_years = hauls["year"].nunique()
    out = hauls.groupby(by).agg(
        n_hauls=("haul_id", "size"),
        n_years=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        depth=("depth", "mean"),
    )
    out["year_coverage"] = out["n_years"] / n_years
    out["hauls_per_year"] = out["n_hauls"] / out["n_years"]
    return out.sort_values("n_hauls", ascending=False)


if __name__ == "__main__":
    df = load_survey("NEUS")
    gb = subset(df, survey_unit="NEUS-Fall", strata=GEORGES_BANK_STRATA, years=(1970, 2019))
    hauls = haul_table(gb)
    mat = abundance_matrix(gb, classes=("Teleostei", "Elasmobranchii"))
    print(f"{len(hauls)} hauls, {mat.shape[1]} species, "
          f"{100 * (mat > 0).to_numpy().mean():.1f}% of cells non-zero")
