"""Estimator weights and the estimators derived from signed measures."""

from __future__ import annotations

from math import gcd

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

from telescopic._types import FloatArray, IntArray, averaging, integer


def averaging_times(k: int, m: int, thin: int) -> IntArray:
    """The averaging set :math:`S = \\{k, k + s, k + 2s, \\dots\\} \\cap [k, m]` for ``thin`` = s."""
    k, m, thin = averaging(k, m, thin)
    return np.arange(k, m + 1, thin, dtype=np.int64)


def correction_coefficients(t: npt.ArrayLike, k: int, m: int, lag: int, thin: int) -> FloatArray:
    r"""Coefficients :math:`c_t` of the differences in the thinned estimator :math:`H_S`.

    .. math:: c_t = |S|^{-1}\,\#\{l \in S : l \le t - L,\ L \mid (t - l)\}

    with :math:`S` from :func:`averaging_times` (spec R-EST-5). For ``thin`` = 1 it is
    :math:`v_t/(m-k+1)` with the lag weights :math:`v_t` of spec R-EST-4. The count is
    computed in constant time per :math:`t`:
    :math:`l = k + js` satisfies the congruence :math:`js \equiv t - k \pmod L`, whose
    solutions form one residue class modulo :math:`L/\gcd(s, L)` when
    :math:`\gcd(s, L)` divides :math:`t - k`, and none otherwise.

    Parameters
    ----------
    t : array_like of int
        Times of the differences :math:`h(X_t) - h(Y_{t-L})`.
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        The spacing :math:`s \ge 1` of the averaging set.

    Returns
    -------
    ndarray of float, same shape as ``t``
    """
    k, m, thin = averaging(k, m, thin)
    lag = integer("lag", lag, minimum=1)
    t = np.asarray(t, dtype=np.int64)
    n_averaged = (m - k) // thin + 1
    g = gcd(thin, lag)
    period = lag // g
    offset = t - k
    solvable = (offset % g == 0) & (t >= k + lag)
    # The smallest j >= 0 with j * thin = offset (mod lag), and the largest admissible j.
    first = (offset // g) * pow(thin // g, -1, period) % period
    last = np.minimum(n_averaged - 1, (t - lag - k) // thin)
    count = np.where(solvable & (first <= last), (last - first) // period + 1, 0)
    return count / n_averaged


def unbiased_covariance(first: FloatArray, second: FloatArray) -> FloatArray:
    r"""Unbiased estimate of :math:`\mathrm{Cov}_\pi(h)` from :math:`R \ge 2` replicates.

    .. math:: \widehat{\mathrm{Cov}}_\pi(h) = \bar b - \bar a\bar a^\top + S_a/R

    with :math:`a_r = \hat\pi_r(h)`, :math:`\bar b` the mean of
    :math:`\hat\pi_r(hh^\top)` and :math:`S_a` the sample covariance of the :math:`a_r`
    (spec R-EST-8). The last two terms are minus the U-statistic
    :math:`\sum_{r \ne s} a_r a_s^\top / (R(R-1))`, which is unbiased for
    :math:`\pi(h)\pi(h)^\top`; the plug-in :math:`\bar a\bar a^\top` is biased by
    :math:`\mathrm{Cov}(\bar a)`.

    Parameters
    ----------
    first : ndarray, shape (R, p)
        The values :math:`a_r`.
    second : ndarray, shape (p, p)
        The mean :math:`\bar b` over replicates of :math:`\hat\pi_r(hh^\top)`, which
        pools all atoms: :math:`R^{-1}\sum_i \omega_i h(x_i)h(x_i)^\top`.

    Returns
    -------
    ndarray, shape (p, p)
    """
    n = len(first)
    if n < 2:
        raise ValueError(f"an unbiased covariance needs at least 2 replicates, not {n}")
    mean = first.mean(axis=0)
    sample = np.atleast_2d(np.cov(first, rowvar=False))
    # Symmetrising the second moments makes the result exactly symmetric.
    covariance: FloatArray = (second + second.T) / 2 - np.outer(mean, mean) + sample / n
    return covariance


def unbiased_variance(first: FloatArray, second: FloatArray) -> FloatArray:
    r"""Unbiased estimates of the variances of :math:`h` (spec R-EST-8).

    The diagonal of :func:`unbiased_covariance`, without the off-diagonal cost:
    :math:`\bar b_j - \bar a_j^2 + S_{a,jj}/R`.

    Parameters
    ----------
    first : ndarray, shape (R, p)
        The values :math:`a_r = \hat\pi_r(h)`.
    second : ndarray, shape (p,)
        The mean over replicates of :math:`\hat\pi_r(h^2)`.

    Returns
    -------
    ndarray, shape (p,)
    """
    n = len(first)
    if n < 2:
        raise ValueError(f"an unbiased variance needs at least 2 replicates, not {n}")
    mean = first.mean(axis=0)
    variance: FloatArray = second - mean**2 + first.var(axis=0, ddof=1) / n
    return variance


def weighted_quantile(values: FloatArray, weights: FloatArray, q: npt.ArrayLike) -> FloatArray:
    r"""Quantiles of a signed measure on the line (spec section 3.4).

    Equal values are merged, the cumulative weights are projected onto non-decreasing
    sequences by isotonic regression and clipped to [0, 1], and the quantile at level
    :math:`\beta` is the smallest value whose projected cumulative weight is at least
    :math:`\beta`; at a level that a cumulative weight equals exactly, rounding of the
    sums decides between two neighbouring values. For non-negative weights this is the
    inverted-CDF quantile. The
    result is consistent as the number of replicates grows, not unbiased (Jacob,
    O'Leary & Atchade 2020, Proposition 3.2).

    Parameters
    ----------
    values : ndarray, shape (N,)
    weights : ndarray, shape (N,)
        Signed weights, summing to 1.
    q : array_like of float in [0, 1]

    Returns
    -------
    ndarray, shape of ``q``
    """
    q = np.asarray(q, dtype=np.float64)
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError(f"q must be in [0, 1], not {q}")
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse.ravel(), weights=weights)
    cdf = np.clip(isotonic_regression(np.cumsum(mass)).x, 0.0, 1.0)
    position = np.searchsorted(cdf, q)
    quantiles: FloatArray = unique[np.minimum(position, len(unique) - 1)]
    return quantiles
