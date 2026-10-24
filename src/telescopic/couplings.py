"""Couplings of pairs of distributions: the primitives that coupled kernels build on."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import softmax

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, Distribution, FloatArray, IndexArray, real

_DIRICHLET = type(stats.dirichlet([1.0, 1.0]))


def reflection_maximal(
    mu_x: FloatArray, mu_y: FloatArray, chol: FloatArray, inverse: FloatArray, draws: Draws
) -> tuple[FloatArray, FloatArray, BoolArray]:
    r"""Reflection-maximal coupling of :math:`\mathcal N(\mu_x, CC^\top)` and :math:`\mathcal N(\mu_y, CC^\top)`.

    With :math:`\delta = C^{-1}(\mu_x - \mu_y)`, :math:`z \sim \mathcal N(0, I)` and
    :math:`x = \mu_x + Cz`, the pair is identical when
    :math:`\log u < -z\cdot\delta - \lVert\delta\rVert^2/2`; otherwise
    :math:`y = \mu_y + C(z - 2(e\cdot z)e)` with :math:`e = \delta/\lVert\delta\rVert`, or
    :math:`e = 0` where :math:`\lVert\delta\rVert` is 0 or not finite
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
    # An infinite delta, from a drift that overflowed, can give inf - inf = NaN, which
    # fails the test as it should.
    with np.errstate(invalid="ignore"):
        identical = log_u < -np.einsum("ij,ij->i", z, delta) - 0.5 * squared
    norm = np.sqrt(squared)[:, None]
    # e = 0 where the norm is 0 or not finite: y is then mu_y + C z, which keeps its law.
    e = np.divide(delta, norm, out=np.zeros_like(delta), where=(norm > 0) & (norm < np.inf))
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
    Algorithm 2; with :math:`\eta < 1`, the bounded-variance version of Gerber & Lee 2020,
    Atchadé & Jacob 2024 Algorithm 2; spec R-CPL-2). The meeting probability is
    :math:`\int\min(\eta p, q)` over the draws whose log-densities are finite, maximal at
    :math:`\eta = 1`; the number of draws from ``p`` and ``q`` has mean 2 and variance
    at most :math:`2\eta/(1-\eta)`.

    Where either log-density of a draw is infinite, the pair is not identical and the
    loop accepts the draw. Such a draw, as one that underflowed to the edge of its
    support, stands for the mass of an interval whose ratio of densities is unknown, and
    each chain keeps its sampler's law. Such draws can be frequent: a third of
    Beta(0.01, 0.01)'s are 1.0, its mass within :math:`2^{-53}` of 1, where the pair is
    not identical even for :math:`p = q`, though its draws can be equal.

    Parameters
    ----------
    p, q : list of distributions, length n
        Objects with ``rvs(random_state=...)`` and a normalised ``logpdf`` (or ``logpmf``),
        such as frozen SciPy distributions. All draws have the same shape. Values of
        ``logpdf`` for the coordinates of a draw are summed, as the joint log-density of
        independent coordinates; NaN raises ``ValueError``.
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
    # NumPy's log, whose last bit can differ from math.log's.
    log_eta = float(np.log(real("eta", eta, 0, 1, closed=True)))
    xs, ys, n_draws = [], [], np.ones(len(p), np.intp)
    for row, (p_row, q_row) in enumerate(zip(p, q, strict=True)):
        generator = draws.generator(row)
        x = y = p_row.rvs(random_state=generator)
        # Not finite exactly where either log-density is infinite.
        log_ratio = _log_density(q_row, x) - _log_density(p_row, x)
        accepted = math.isfinite(log_ratio) and _log_uniform(generator) < min(log_eta, log_ratio)
        while not accepted:
            y = q_row.rvs(random_state=generator)
            n_draws[row] += 1
            log_ratio = _log_density(p_row, y) - _log_density(q_row, y)
            accepted = (
                not math.isfinite(log_ratio) or _log_uniform(generator) > log_eta + log_ratio
            )
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
        Log-probabilities, normalised or not; each row needs a finite maximum and no NaN.
    draws : Draws
        Random numbers for the ``n`` rows; each row uses 3 uniforms.

    Returns
    -------
    i, j : ndarray of int, shape (n,)
        The coupled indices.
    """
    for name, log in (("log_p", log_p), ("log_q", log_q)):
        # The maximum is NaN for a row with NaN, and infinite for one of zero or infinite mass.
        valid = np.isfinite(np.max(log, axis=1))
        if not valid.all():
            row = int(np.argmin(valid))
            raise ValueError(
                f"each row of {name} needs a finite maximum and no NaN, but row {row} is "
                f"{log[row].tolist()}"
            )
    p, q = softmax(log_p, axis=1), softmax(log_q, axis=1)
    overlap = np.minimum(p, q)
    alpha = overlap.sum(axis=1)
    meet = draws.uniform() < alpha
    u_i, u_j = draws.uniform(), draws.uniform()
    i = np.where(meet, _categorical(overlap, u_i), _categorical(p - overlap, u_i))
    j = np.where(meet, i, _categorical(q - overlap, u_j))
    return i, j


def _log_uniform(generator: np.random.Generator) -> float:
    u = generator.random()
    return float(np.log(u)) if u > 0 else -np.inf


def _log_density(distribution: Any, value: Any) -> float:
    name = "logpdf" if hasattr(distribution, "logpdf") else "logpmf"
    if isinstance(distribution, _DIRICHLET):
        # SciPy's dirichlet draws shape (1, k), and its logpdf reads the first axis as the
        # components: it refuses (1, k) for k > 2, and reads (1, 2) as two points.
        value = np.ravel(value)
        # It also refuses a component of 0 whose alpha is below 1, a pole that draws reach
        # by underflow.
        if np.any((value == 0) & (distribution.alpha < 1)):
            return math.inf
    log_density = getattr(distribution, name)(value)
    # One value per coordinate, from array parameters: the joint log-density of
    # independent coordinates is their sum. A float skips np.sum, which doubled the time
    # of a coupled Gibbs sweep of examples.pumps. A Python float, as NumPy's warns at
    # inf - inf.
    log_density = float(log_density if isinstance(log_density, float) else np.sum(log_density))
    if math.isnan(log_density):
        raise ValueError(f"{name} returned nan at {np.asarray(value).tolist()}")
    return log_density


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
