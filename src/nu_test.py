"""
nu_test — test whether the Beta concentration is shared across species.

THE HYPOTHESIS
--------------
Rebolledo, Tejo, Freilich & Marquet (2026), sections 4.3-4.4, give for both the
CIR and the stochastic-logistic construction

    Z_i(t) = X_i(t) / sum_j X_j(t)  ~  Beta( u_i , nu - u_i ),   nu = sum_j u_j

so each species has its own mean p_i = u_i / nu but they all share the
concentration nu, which carries no species index. Equivalently: the vector of
relative abundances is Dirichlet(u_1, ..., u_K), and this is a marginal test of
that Dirichlet structure.

Writing each species' Beta in (mean, concentration) form,

    alpha_i = nu_i p_i ,   beta_i = nu_i (1 - p_i)

the test is

    H0 :  nu_1 = nu_2 = ... = nu_K        (24 parameters: K means + 1 nu)
    H1 :  nu_i free                       (2K parameters)

compared by likelihood ratio. Heterogeneous nu is a mechanistic rejection of
neutrality: the paper itself names "heterogeneous fitness across species" as the
natural non-neutral generalisation (p. 33).

WHY THE NULL IS NOT CALIBRATED BY CHI-SQUARED
---------------------------------------------
Two reasons, both established earlier in this project:

1. n = 39 annual observations per species is small, and the Beta concentration
   is a notoriously hard parameter to estimate. The asymptotic chi-squared
   reference for the LR statistic is not trustworthy here.

2. A species' annual shares are a Jacobi *path*, not an i.i.d. sample. They are
   autocorrelated, so an i.i.d. bootstrap understates the sampling variance of
   nu_i and would over-reject homogeneity.

So the null distribution is obtained by a parametric bootstrap that simulates
autocorrelated Jacobi paths under H0, with each species' mean-reversion rate
estimated from its own observed lag-1 autocorrelation.

THE JACOBI DIFFUSION
--------------------
    dX = a (b - X) dt + sigma sqrt( X (1-X) ) dW

has stationary law Beta(2ab/sigma^2, 2a(1-b)/sigma^2), i.e. mean b and
concentration nu = 2a/sigma^2, and autocorrelation corr(X_t, X_{t+h}) = exp(-a h).
Given a target (p, nu) and a mean-reversion rate a, set b = p and
sigma^2 = 2a/nu.

Requires numpy, scipy, pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, digamma, polygamma

__all__ = [
    "build_shares",
    "fit_free",
    "fit_common_nu",
    "lr_statistic",
    "estimate_mean_reversion",
    "simulate_jacobi",
    "bootstrap_lr",
]

_EPS = 1e-10


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def build_shares(
    survey="NEUS",
    survey_unit="NEUS-Fall",
    strata=None,
    years=(1970, 2008),
    classes=("Teleostei", "Elasmobranchii"),
    require_all_years=True,
):
    """Annual biomass shares: a (years x species) frame.

    Share_i(t) = (summed wgt_cpua of species i in year t) / (community total in
    year t), where the community total runs over every species-level fish
    record in the subset -- not only the species retained.

    Keeps species that are present in every year AND whose shares are all
    strictly inside (0, 1). Exact zeros arising from the FishGlob zero-weight
    records (num > 0 but wgt == 0) are reported, not silently squeezed.
    """
    import fishglob as fg

    if strata is None:
        strata = fg.GEORGES_BANK_STRATA        # the default pilot region
    elif isinstance(strata, str) and strata.lower() == "all":
        strata = None                          # whole survey unit, no stratum filter

    df = fg.load_survey(survey)
    sub = fg.subset(df, survey_unit=survey_unit, strata=strata, years=years)
    s = sub[(sub["rank"] == "Species") & sub["class"].isin(list(classes))]

    total = s.groupby("year").wgt_cpua.sum()
    per = s.groupby(["accepted_name", "year"]).wgt_cpua.sum()
    share = (per / total).unstack(0).reindex(range(years[0], years[1] + 1))

    report = {"n_years": share.shape[0], "n_species_any": int(share.notna().any().sum())}

    if require_all_years:
        full = share.columns[share.notna().all(axis=0)]
        share = share[full]
        report["n_species_all_years"] = len(full)
        zero_hit = share.columns[(share == 0).any(axis=0)]
        report["dropped_zero_weight"] = {
            c: list(share.index[share[c] == 0]) for c in zero_hit
        }
        share = share.drop(columns=list(zero_hit))

    report["n_species_used"] = share.shape[1]
    report["share_sum_by_year"] = (share.sum(axis=1)).describe()[["min", "50%", "max"]]
    return share, report


# --------------------------------------------------------------------------
# Likelihood machinery, in (mean, concentration) parameterisation
# --------------------------------------------------------------------------


def _beta_loglik(x, p, nu):
    """Sum of Beta log-densities for one series, mean p and concentration nu."""
    a, b = nu * p, nu * (1.0 - p)
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return -np.inf
    return float(
        np.sum((a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)) - len(x) * betaln(a, b)
    )


def _profile_p(x, nu):
    """Max over the mean p of the Beta log-likelihood, at fixed concentration.

    Solving d/dp = 0 gives the score equation
        psi(nu p) - psi(nu(1-p)) = mean(log x) - mean(log(1-x))
    which is monotone in p, so a safeguarded Newton iteration is reliable.
    """
    L1 = np.mean(np.log(x))
    L2 = np.mean(np.log1p(-x))
    target = L1 - L2

    lo, hi = 1e-8, 1.0 - 1e-8
    p = float(np.clip(np.mean(x), 1e-4, 1 - 1e-4))
    for _ in range(100):
        a, b = nu * p, nu * (1.0 - p)
        f = digamma(a) - digamma(b) - target
        if abs(f) < 1e-12:
            break
        fp = nu * (polygamma(1, a) + polygamma(1, b))  # d/dp, always > 0
        if f > 0:
            hi = p
        else:
            lo = p
        step = f / fp
        p_new = p - step
        if not (lo < p_new < hi):
            p_new = 0.5 * (lo + hi)
        if abs(p_new - p) < 1e-14:
            p = p_new
            break
        p = p_new
    return p, _beta_loglik(x, p, nu)


def _fit_one(x):
    """Unrestricted Beta MLE for one series; returns (p, nu, loglik)."""

    def neg(lognu):
        _, ll = _profile_p(x, np.exp(lognu))
        return -ll

    m, v = np.mean(x), np.var(x, ddof=1)
    nu0 = max(m * (1 - m) / max(v, 1e-12) - 1.0, 0.5)
    lo, hi = np.log(nu0) - 6.0, np.log(nu0) + 6.0
    res = minimize_scalar(neg, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    nu = float(np.exp(res.x))
    p, ll = _profile_p(x, nu)
    return p, nu, ll


def fit_free(X):
    """Per-species unrestricted fit. X is (years x species). Returns a frame."""
    X = np.asarray(X, dtype=float)
    out = [_fit_one(X[:, j]) for j in range(X.shape[1])]
    p = np.array([o[0] for o in out])
    nu = np.array([o[1] for o in out])
    ll = np.array([o[2] for o in out])
    return p, nu, float(ll.sum())


def fit_common_nu(X):
    """Restricted fit: one nu shared by all species, means free.

    Profiles the means out at each nu, so the outer problem is one-dimensional
    and well behaved.
    """
    X = np.asarray(X, dtype=float)
    K = X.shape[1]

    def neg(lognu):
        nu = np.exp(lognu)
        return -sum(_profile_p(X[:, j], nu)[1] for j in range(K))

    _, nu_free, _ = fit_free(X)
    c = np.log(np.median(nu_free))
    res = minimize_scalar(neg, bounds=(c - 6.0, c + 6.0), method="bounded",
                          options={"xatol": 1e-10})
    nu = float(np.exp(res.x))
    p = np.array([_profile_p(X[:, j], nu)[0] for j in range(K)])
    ll = sum(_profile_p(X[:, j], nu)[1] for j in range(K))
    return p, nu, float(ll)


def lr_statistic(X):
    """2 * (unrestricted - restricted) log-likelihood."""
    _, _, ll1 = fit_free(X)
    _, _, ll0 = fit_common_nu(X)
    return 2.0 * (ll1 - ll0)


# --------------------------------------------------------------------------
# Autocorrelation and Jacobi path simulation
# --------------------------------------------------------------------------


def estimate_mean_reversion(X, floor=0.05, cap=5.0):
    """Mean-reversion rate a per species, from lag-1 autocorrelation.

    For the Jacobi diffusion corr(X_t, X_{t+h}) = exp(-a h), so with annual
    sampling a = -log(rho_1). Non-positive rho_1 means no detectable
    persistence, which maps to a large a (fast reversion, effectively i.i.d.);
    the cap keeps the Euler step stable.
    """
    X = np.asarray(X, dtype=float)
    a = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        xc = x - x.mean()
        denom = np.sum(xc * xc)
        rho = np.sum(xc[:-1] * xc[1:]) / denom if denom > 0 else 0.0
        a[j] = cap if rho <= 1e-6 else float(np.clip(-np.log(rho), floor, cap))
    return a


def simulate_jacobi(p, nu, a, n_years, n_paths, rng, dt=1.0 / 128, burn=40):
    """Simulate Jacobi paths sampled annually.

        dX = a (b - X) dt + sqrt(2a/nu) sqrt(X(1-X)) dW

    Stationary law Beta(nu*p, nu*(1-p)); autocorrelation exp(-a h). Paths start
    from the stationary distribution and are burned in anyway, so the returned
    series is stationary by construction.

    Returns an (n_paths, n_years) array.
    """
    sigma = np.sqrt(2.0 * a / nu)
    steps = int(round(1.0 / dt))
    x = rng.beta(nu * p, nu * (1.0 - p), size=n_paths)

    for _ in range(burn * steps):
        x = _euler_step(x, a, p, sigma, dt, rng)

    out = np.empty((n_paths, n_years))
    for t in range(n_years):
        for _ in range(steps):
            x = _euler_step(x, a, p, sigma, dt, rng)
        out[:, t] = x
    return out


def _euler_step(x, a, b, sigma, dt, rng):
    diff = sigma * np.sqrt(np.maximum(x * (1.0 - x), 0.0))
    x = x + a * (b - x) * dt + diff * np.sqrt(dt) * rng.standard_normal(x.shape)
    return np.clip(x, _EPS, 1.0 - _EPS)


def simulate_jacobi_vec(p, nu, a, n_years, rng, dt=1.0 / 128, burn=8):
    """Vectorised simulator: p, nu, a are arrays, one entry per path.

    Simulates every path simultaneously, which matters because the per-call
    numpy overhead otherwise dominates when thousands of short paths are
    needed. Returns an (n_paths, n_years) array.
    """
    p = np.asarray(p, dtype=float)
    nu = np.asarray(nu, dtype=float)
    a = np.asarray(a, dtype=float)
    sigma = np.sqrt(2.0 * a / nu)
    steps = int(round(1.0 / dt))
    sq = np.sqrt(dt)

    x = rng.beta(nu * p, nu * (1.0 - p))
    for _ in range(burn * steps):
        x = np.clip(
            x + a * (p - x) * dt
            + sigma * np.sqrt(np.maximum(x * (1.0 - x), 0.0)) * sq
            * rng.standard_normal(x.shape),
            _EPS, 1.0 - _EPS)

    out = np.empty((x.size, n_years))
    for t in range(n_years):
        for _ in range(steps):
            x = np.clip(
                x + a * (p - x) * dt
                + sigma * np.sqrt(np.maximum(x * (1.0 - x), 0.0)) * sq
                * rng.standard_normal(x.shape),
                _EPS, 1.0 - _EPS)
        out[:, t] = x
    return out


# --------------------------------------------------------------------------
# Bootstrap test
# --------------------------------------------------------------------------


def bootstrap_lr(X, n_boot=999, rng=None, nu_true=None, verbose=False):
    """Parametric Jacobi-path bootstrap for the shared-concentration test.

    Under H0 each species keeps its own fitted mean and its own estimated
    mean-reversion rate, but all share the common concentration. Passing
    nu_true (an array) instead simulates a heterogeneous-nu alternative, which
    is how the power check is run.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n_years, K = X.shape

    p0, nu0, _ = fit_common_nu(X)
    a_hat = estimate_mean_reversion(X)
    lr_obs = lr_statistic(X)

    nu_sim = np.full(K, nu0) if nu_true is None else np.asarray(nu_true, dtype=float)

    lr_null = np.empty(n_boot)
    chunk = 250
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        sims = np.empty((m, n_years, K))
        for j in range(K):
            sims[:, :, j] = simulate_jacobi_vec(
                np.full(m, p0[j]), np.full(m, nu_sim[j]),
                np.full(m, a_hat[j]), n_years, rng
            )
        for b in range(m):
            lr_null[done + b] = lr_statistic(sims[b])
        done += m
        if verbose:
            print(f"    bootstrap {done}/{n_boot}", flush=True)

    pval = float((1 + np.sum(lr_null >= lr_obs)) / (n_boot + 1))
    return {
        "lr_obs": lr_obs,
        "lr_null": lr_null,
        "p_value": pval,
        "nu_common": nu0,
        "p_common": p0,
        "a_hat": a_hat,
        "n_years": n_years,
        "n_species": K,
    }
