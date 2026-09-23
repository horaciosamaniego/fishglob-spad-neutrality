"""
betagof — goodness-of-fit tests for the two-parameter Beta family.

Python implementation of the L2-type test of

    Ebner, B. & Liebenberg, S.C. (2021). On a new test of fit to the beta
    distribution. Stat 10:e341. https://doi.org/10.1002/sta4.341

together with the classical EDF competitors (Kolmogorov-Smirnov,
Cramer-von Mises, Anderson-Darling), all calibrated by the same
parametric bootstrap so that they are directly comparable.

Design notes
------------
1.  The test statistic is

        T_n = n * int_0^1 | (1/n) sum_j ((a+b) X_j - a) 1{X_j >= t}
                            - t^a (1-t)^b / B(a,b) |^2 dt

    with (a, b) replaced by consistent estimators. The paper's "numerically
    stable representation" (their Eq. 3 / published Eq. 4) is an O(n^2)
    double sum. Here the first term is evaluated in O(n log n) by noting

        sum_{j,k} psi_j psi_k min(X_j, X_k) = sum_i (x_(i) - x_(i-1)) * R_i^2

    where x_(i) are the order statistics (x_(0) = 0), psi_j = (a+b)X_j - a,
    and R_i is the suffix sum of psi over the sorted sample. Exact, not an
    approximation; ties are handled automatically.

2.  H_0 is *composite* (any Beta), so parameters are re-estimated inside
    every bootstrap replicate. The paper is explicit about this: "for each
    computation of T*_{n,j}, parameter estimation has to be done separately
    for each j." Holding (a, b) fixed across replicates gives the wrong null
    distribution.

3.  MLE is obtained by solving the digamma system

        psi(a) - psi(a+b) = mean(log X)
        psi(b) - psi(a+b) = mean(log(1-X))

    with a damped Newton iteration, vectorised over bootstrap replicates so
    that B = 1000 replicates cost a fraction of a second.

Requires numpy and scipy only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.special import betainc, betaln, digamma, polygamma

__all__ = [
    "beta_mom",
    "beta_mle",
    "tn_statistic",
    "tn_statistic_naive",
    "edf_statistics",
    "beta_gof_test",
    "squeeze_unit_interval",
    "BetaGofResult",
]

_TINY = 1e-12


# --------------------------------------------------------------------------
# Parameter estimation
# --------------------------------------------------------------------------


def beta_mom(x: np.ndarray) -> tuple[float, float]:
    """Method-of-moments estimates of (a, b). May be invalid (<= 0) if the
    sample variance is too large for any Beta; caller should check."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    v = x.var(ddof=1)
    common = m * (1.0 - m) / v - 1.0
    return float(m * common), float((1.0 - m) * common)


def _mom_batch(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise method of moments for a (B, n) array, clipped to be usable
    as a Newton starting value."""
    m = x.mean(axis=1)
    v = x.var(axis=1, ddof=1)
    common = m * (1.0 - m) / np.maximum(v, _TINY) - 1.0
    a0 = m * common
    b0 = (1.0 - m) * common
    bad = ~np.isfinite(a0) | ~np.isfinite(b0) | (a0 <= 0) | (b0 <= 0)
    a0 = np.where(bad, 1.0, a0)
    b0 = np.where(bad, 1.0, b0)
    return np.clip(a0, 1e-3, 1e4), np.clip(b0, 1e-3, 1e4)


def _mle_batch(
    x: np.ndarray, tol: float = 1e-10, max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised MLE for a (B, n) array of Beta samples.

    Damped Newton on the digamma score equations, run simultaneously on all
    B rows. Returns arrays (a_hat, b_hat) of length B.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    L1 = np.log(x).mean(axis=1)
    L2 = np.log1p(-x).mean(axis=1)
    a, b = _mom_batch(x)

    for _ in range(max_iter):
        dab = digamma(a + b)
        f1 = digamma(a) - dab - L1
        f2 = digamma(b) - dab - L2
        if np.max(np.abs(f1)) < tol and np.max(np.abs(f2)) < tol:
            break

        t_ab = polygamma(1, a + b)
        j11 = polygamma(1, a) - t_ab
        j22 = polygamma(1, b) - t_ab
        j12 = -t_ab
        det = j11 * j22 - j12 * j12
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)

        da = (j22 * f1 - j12 * f2) / det
        db = (-j12 * f1 + j11 * f2) / det

        # Step halving to keep both parameters strictly positive.
        step = np.ones_like(a)
        for _ in range(60):
            a_new = a - step * da
            b_new = b - step * db
            bad = (a_new <= 1e-8) | (b_new <= 1e-8)
            if not bad.any():
                break
            step = np.where(bad, step * 0.5, step)
        a = np.maximum(a - step * da, 1e-8)
        b = np.maximum(b - step * db, 1e-8)

    return a, b


def beta_mle(x: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood estimates of (a, b) for a single sample."""
    x = _check_sample(np.asarray(x, dtype=float))
    a, b = _mle_batch(x[None, :])
    return float(a[0]), float(b[0])


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def _check_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 5:
        raise ValueError("need at least 5 observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("sample contains non-finite values")
    n_bad = int(np.sum((x <= 0.0) | (x >= 1.0)))
    if n_bad:
        raise ValueError(
            f"{n_bad} observation(s) are exactly 0 or 1 (or outside [0,1]). "
            "The Beta log-likelihood is undefined there. Either exclude them "
            "on substantive grounds or apply squeeze_unit_interval(), but note "
            "that squeezing changes the hypothesis being tested."
        )
    return x


def squeeze_unit_interval(x: np.ndarray) -> np.ndarray:
    """Smithson & Verkuilen (2006) transform: (x*(n-1) + 0.5)/n.

    Maps [0, 1] into the open interval so that boundary observations do not
    break the likelihood. This is a modelling decision, not a fix: with many
    exact zeros the resulting sample is not a draw from any Beta and the test
    will reject for that reason.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    out = (x * (n - 1) + 0.5) / n
    frac_boundary = float(np.mean((x <= 0.0) | (x >= 1.0)))
    if frac_boundary > 0.05:
        warnings.warn(
            f"{frac_boundary:.1%} of observations sit on the boundary; "
            "squeezing creates an artificial spike that T_n will detect.",
            stacklevel=2,
        )
    return out


# --------------------------------------------------------------------------
# Test statistics
# --------------------------------------------------------------------------


def _b1_ratio(a, b):
    """B(a+1, b+1) / B(a, b) = a*b / ((a+b)(a+b+1)), computed exactly."""
    return a * b / ((a + b) * (a + b + 1.0))


def _b2_ratio(a, b):
    """B(2a+1, 2b+1) / B(a, b)^2, via log-gamma for numerical safety."""
    return np.exp(betaln(2.0 * a + 1.0, 2.0 * b + 1.0) - 2.0 * betaln(a, b))


def _tn_batch(x_sorted: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T_n for each row of a (B, n) array of *sorted* samples."""
    n = x_sorted.shape[1]
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[:, None]

    psi = (a + b) * x_sorted - a

    # Term 1: (1/n) sum_{j,k} psi_j psi_k min(x_j, x_k), in O(n log n).
    suffix = np.cumsum(psi[:, ::-1], axis=1)[:, ::-1]
    gaps = np.diff(x_sorted, axis=1, prepend=0.0)
    term1 = np.sum(gaps * suffix**2, axis=1) / n

    # Term 2: cross term against g(t) = t^a (1-t)^b / B(a, b).
    cdf = betainc(a + 1.0, b + 1.0, x_sorted)
    term2 = 2.0 * _b1_ratio(a[:, 0], b[:, 0]) * np.sum(psi * cdf, axis=1)

    # Term 3: n * ||g||^2.
    term3 = n * _b2_ratio(a[:, 0], b[:, 0])

    return term1 - term2 + term3


def tn_statistic(x: np.ndarray, a: float, b: float) -> float:
    """T_n of Ebner & Liebenberg (2021) for one sample and given (a, b)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    return float(_tn_batch(x[None, :], np.array([a]), np.array([b]))[0])


def tn_statistic_naive(x: np.ndarray, a: float, b: float) -> float:
    """O(n^2) transcription of the published formula. Reference only."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    psi = (a + b) * x - a
    term1 = np.sum(np.outer(psi, psi) * np.minimum.outer(x, x)) / n
    term2 = 2.0 * _b1_ratio(a, b) * np.sum(psi * betainc(a + 1.0, b + 1.0, x))
    term3 = n * _b2_ratio(a, b)
    return float(term1 - term2 + term3)


def _edf_batch(x_sorted: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """KS, Cramer-von Mises and Anderson-Darling for each row."""
    n = x_sorted.shape[1]
    u = betainc(np.asarray(a)[:, None], np.asarray(b)[:, None], x_sorted)
    u = np.clip(u, _TINY, 1.0 - _TINY)
    i = np.arange(1, n + 1)

    d_plus = np.max(i / n - u, axis=1)
    d_minus = np.max(u - (i - 1) / n, axis=1)
    ks = np.maximum(d_plus, d_minus)

    cvm = np.sum((u - (2 * i - 1) / (2 * n)) ** 2, axis=1) + 1.0 / (12 * n)
    ad = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1.0 - u[:, ::-1])), axis=1) / n

    return {"KS": ks, "CvM": cvm, "AD": ad}


def edf_statistics(x: np.ndarray, a: float, b: float) -> dict:
    """KS / CvM / AD statistics for one sample and given (a, b)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    out = _edf_batch(x[None, :], np.array([a]), np.array([b]))
    return {k: float(v[0]) for k, v in out.items()}


# --------------------------------------------------------------------------
# Parametric bootstrap
# --------------------------------------------------------------------------


@dataclass
class BetaGofResult:
    """Outcome of a parametric-bootstrap goodness-of-fit test for the Beta."""

    n: int
    a_hat: float
    b_hat: float
    estimator: str
    n_boot: int
    statistics: dict = field(default_factory=dict)
    p_values: dict = field(default_factory=dict)
    critical_values: dict = field(default_factory=dict)
    null_draws: dict = field(default_factory=dict, repr=False)
    n_boundary_clipped: int = 0

    def summary(self) -> str:
        lines = [
            f"Beta goodness-of-fit, n = {self.n}, B = {self.n_boot}",
            f"  {self.estimator.upper()} estimates: a = {self.a_hat:.4f}, "
            f"b = {self.b_hat:.4f}",
            f"  mean = {self.a_hat / (self.a_hat + self.b_hat):.4f}",
            "",
            f"  {'test':>6}  {'statistic':>12}  {'p (boot)':>9}  "
            f"{'crit 0.05':>10}",
        ]
        for k in self.statistics:
            crit = self.critical_values.get(k, {}).get(0.05, float("nan"))
            lines.append(
                f"  {k:>6}  {self.statistics[k]:12.5f}  "
                f"{self.p_values[k]:9.4f}  {crit:10.5f}"
            )
        if self.n_boundary_clipped:
            lines.append(
                f"  note: {self.n_boundary_clipped} bootstrap draw(s) clipped "
                "off the boundary"
            )
        return "\n".join(lines)


def beta_gof_test(
    x,
    n_boot: int = 1000,
    estimator: str = "mle",
    levels=(0.01, 0.05, 0.10),
    rng=None,
    include_edf: bool = True,
    chunk: int = 500,
) -> BetaGofResult:
    """Test H0: the sample comes from *some* Beta(a, b) distribution.

    Parameters
    ----------
    x : array-like
        Observations, strictly inside (0, 1).
    n_boot : int
        Number of parametric bootstrap replicates (B). 1000 is a reasonable
        default; use >= 5000 when a p-value near a decision boundary matters.
    estimator : {"mle", "mom"}
        Estimator used both on the data and inside every replicate. The paper
        finds MLE gives uniformly more power than moments.
    levels : sequence of float
        Nominal levels for reported critical values.
    rng : numpy Generator, int seed, or None
        Randomness source. Pass a seed for reproducible pipelines.
    include_edf : bool
        Also compute KS / CvM / AD on the same bootstrap samples, correctly
        calibrated for estimated parameters.

    Returns
    -------
    BetaGofResult
    """
    x = _check_sample(x)
    n = x.size
    rng = np.random.default_rng(rng)

    if estimator not in ("mle", "mom"):
        raise ValueError("estimator must be 'mle' or 'mom'")

    def _fit(arr2d):
        if estimator == "mle":
            return _mle_batch(arr2d)
        a0, b0 = _mom_batch(arr2d)
        return a0, b0

    a_hat, b_hat = _fit(x[None, :])
    a_hat, b_hat = float(a_hat[0]), float(b_hat[0])

    xs = np.sort(x)
    obs = {"Tn": tn_statistic(xs, a_hat, b_hat)}
    if include_edf:
        obs.update(edf_statistics(xs, a_hat, b_hat))

    null = {k: np.empty(n_boot) for k in obs}
    n_clipped = 0
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        xb = rng.beta(a_hat, b_hat, size=(m, n))
        n_clipped += int(np.sum((xb <= 0.0) | (xb >= 1.0)))
        np.clip(xb, _TINY, 1.0 - _TINY, out=xb)
        ab, bb = _fit(xb)
        xb.sort(axis=1)
        null["Tn"][done : done + m] = _tn_batch(xb, ab, bb)
        if include_edf:
            for k, v in _edf_batch(xb, ab, bb).items():
                null[k][done : done + m] = v
        done += m

    p_values, crit = {}, {}
    for k, t_obs in obs.items():
        draws = null[k]
        # (1 + #{T* >= T_obs}) / (B + 1): exact for a Monte Carlo test.
        p_values[k] = float((1 + np.sum(draws >= t_obs)) / (n_boot + 1))
        crit[k] = {lv: float(np.quantile(draws, 1.0 - lv)) for lv in levels}

    return BetaGofResult(
        n=n,
        a_hat=a_hat,
        b_hat=b_hat,
        estimator=estimator,
        n_boot=n_boot,
        statistics=obs,
        p_values=p_values,
        critical_values=crit,
        null_draws=null,
        n_boundary_clipped=n_clipped,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(20260908)
    demo = rng.beta(2.0, 5.0, size=150)
    print(beta_gof_test(demo, n_boot=1000, rng=rng).summary())
