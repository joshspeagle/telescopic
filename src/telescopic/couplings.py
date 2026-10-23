"""Couplings of pairs of distributions: the primitives that coupled kernels build on."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import softmax

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, Distribution, FloatArray, IndexArray


def reflection_maximal(
    mu_x: FloatArray, mu_y: FloatArray, chol: FloatArray, inverse: FloatArray, draws: Draws
) -> tuple[FloatArray, FloatArray, BoolArray]:
    r"""Reflection-maximal coupling of :math:`\mathcal N(\mu_x, CC^\top)` and :math:`\mathcal N(\mu_y, CC^\top)`.

    With :math:`\delta = C^{-1}(\mu_x - \mu_y)`, :math:`z \sim \mathcal N(0, I)` and
    :math:`x = \mu_x + Cz`, the pair is identical when
    :math:`\log u < -z\cdot\delta - \lVert\delta\rVert^2/2`; otherwise
    :math:`y = \mu_y + C(z - 2(e\cdot z)e)` with :math:`e = \delta/\lVert\delta\rVert`
    (Jacob, O'Leary & Atchadé 2020, section 4.1; spec R-CPL-1). The coupling is maximal:
    :math:`P(x = y) = 2\Phi(-\lVert\delta\rVert/2)`.

    Parameters
    ----------
    mu_x, mu_y : ndarray, shape (n, d)
        Means of the two Gaussians of each row.
    chol : ndarray, shape (d, d)
        Lower-triangular factor :math:`C` of the shared covariance.
    inverse : ndarray, shape (d, d)
        :math:`C^{-1}`, which whitens the difference of the means.
    draws : Draws
        Random numbers for the ``n`` rows; each row uses ``d`` normals and 1 uniform.

    Returns
    -------
    x, y : ndarray, shape (n, d)
        The coupled draws. On identical rows ``y`` is a copy of ``x``, so equality is exact.
    identical : ndarray of bool, shape (n,)
        Whether each row's draws are equal.
    """
    z = draws.normal(mu_x.shape[1])
    log_u = draws.log_uniform()
    delta = matvec(inverse, mu_x - mu_y)
    squared = np.einsum("ij,ij->i", delta, delta)
    identical = log_u < -np.einsum("ij,ij->i", z, delta) - 0.5 * squared
    norm = np.sqrt(squared)[:, None]
    e = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
    reflected = z - 2.0 * np.einsum("ij,ij->i", e, z)[:, None] * e
    x = mu_x + matvec(chol, z)
    y = np.where(identical[:, None], x, mu_y + matvec(chol, reflected))
    return x, y, identical


def maximal(
    p: list[Distribution], q: list[Distribution], draws: Draws, eta: float = 1.0
) -> tuple[FloatArray, FloatArray, BoolArray, IndexArray]:
    r"""Rejection maximal coupling of the distributions ``p[i]`` and ``q[i]`` of each row.

    Draw :math:`X \sim p`; the pair is identical with probability
    :math:`\min(\eta, q(X)/p(X))`. Otherwise draw :math:`Y \sim q` until
    :math:`u > \eta\,p(Y)/q(Y)` (Thorisson's algorithm, Jacob, O'Leary & Atchadé 2020
    Algorithm 2; with :math:`\eta < 1`, the bounded-variance version of Gerber & Lee,
    Atchadé & Jacob 2024 Algorithm 2; spec R-CPL-2). The meeting probability is
    :math:`\int\min(\eta p, q)`, maximal at :math:`\eta = 1`; the number of draws from
    ``p`` and ``q`` has mean 2 and variance at most :math:`2\eta/(1-\eta)`.

    Parameters
    ----------
    p, q : list of distributions, length n
        Objects with ``rvs(random_state=...)`` and a normalised ``logpdf`` (or ``logpmf``),
        such as frozen SciPy distributions. All draws have the same shape.
    draws : Draws
        Random numbers for the ``n`` rows; each row draws from its own generator.
    eta : float
        Cap on the meeting probability, in (0, 1].

    Returns
    -------
    x, y : ndarray, shape (n, ...)
        The coupled draws; on identical rows ``y`` is a copy of ``x``.
    identical : ndarray of bool, shape (n,)
    n_draws : ndarray of int, shape (n,)
        Draws from ``p`` and ``q`` used by each row.
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], not {eta}")
    log_eta = np.log(eta)
    xs, ys, n_draws = [], [], np.ones(len(p), np.intp)
    for row, (p_row, q_row) in enumerate(zip(p, q, strict=True)):
        generator = draws.generator(row)
        x = y = p_row.rvs(random_state=generator)
        log_ratio = _log_density(q_row, x) - _log_density(p_row, x)
        accepted = _log_uniform(generator) <= min(log_eta, log_ratio)
        while not accepted:
            y = q_row.rvs(random_state=generator)
            n_draws[row] += 1
            log_ratio = _log_density(p_row, y) - _log_density(q_row, y)
            accepted = _log_uniform(generator) > log_eta + log_ratio
        xs.append(x)
        ys.append(y)
    x_array, y_array = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    return x_array, y_array, n_draws == 1, n_draws


def discrete_maximal(
    log_p: FloatArray, log_q: FloatArray, draws: Draws
) -> tuple[IndexArray, IndexArray]:
    r"""Maximal coupling of two distributions on :math:`\{0, \dots, K-1\}`, row by row.

    With probability :math:`\alpha = \sum_k \min(p_k, q_k)` both indices are drawn from
    :math:`\min(p, q)/\alpha`; otherwise they are drawn independently from the normalised
    residuals :math:`p - \min(p, q)` and :math:`q - \min(p, q)`, whose supports are
    disjoint (Jacob, O'Leary & Atchadé 2020, section 5.4; spec R-CPL-3). So
    :math:`P(i = j) = \alpha = 1 - \mathrm{TV}(p, q)`.

    Parameters
    ----------
    log_p, log_q : ndarray, shape (n, K)
        Log-probabilities, normalised or not.
    draws : Draws
        Random numbers for the ``n`` rows; each row uses 3 uniforms.

    Returns
    -------
    i, j : ndarray of int, shape (n,)
        The coupled indices.
    """
    p, q = softmax(log_p, axis=1), softmax(log_q, axis=1)
    overlap = np.minimum(p, q)
    alpha = overlap.sum(axis=1)
    meet = draws.uniform() < alpha
    u_i, u_j = draws.uniform(), draws.uniform()
    i = np.where(meet, _categorical(overlap, u_i), _categorical(p - overlap, u_i))
    j = np.where(meet, i, _categorical(q - overlap, u_j))
    return i, j


def _log_uniform(generator: np.random.Generator) -> float:
    with np.errstate(divide="ignore"):
        return float(np.log(generator.random()))


def _log_density(distribution: Any, value: Any) -> float:
    log_density = getattr(distribution, "logpdf", None) or distribution.logpmf
    return float(log_density(value))


def _categorical(weights: FloatArray, u: FloatArray) -> IndexArray:
    # Inverse-CDF draw from unnormalised non-negative weights: the first index whose
    # cumulative weight exceeds u times the total, so zero-weight categories are never
    # drawn. The clip only matters for rounding at u -> 1 and for an all-zero row (a
    # residual that is never used).
    cumulative = np.cumsum(weights, axis=1)
    index: IndexArray = np.minimum(
        (cumulative <= (u * cumulative[:, -1])[:, None]).sum(axis=1), weights.shape[1] - 1
    )
    return index
