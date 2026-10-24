"""Targets with exact answers, for tests, tutorials and benchmarks."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import integrate, special

from telescopic._types import Distribution, FloatArray
from telescopic.initial import Normal, Uniform


@dataclass(frozen=True, eq=False)
class Example:
    """A target with exact answers (spec section 9).

    Attributes
    ----------
    name : str
    log_prob, grad : callable
        The log-density, up to a constant, and its gradient, for a point of shape
        ``(d,)`` or a batch of shape ``(n, d)``; pass ``vectorize=True`` with batches.
    initial : object with ``rvs(random_state=...)``
        A default initial distribution, with at least the target's variance in each
        coordinate.
    mean, cov : ndarray, shapes (d,) and (d, d)
        The exact mean and covariance.
    ndim : int
        The dimension ``d``.
    conditional : callable, optional
        For :class:`~telescopic.kernels.Gibbs`, ``conditional(x, b)``: the full
        conditional of block ``b`` given the other coordinates of ``x``.
    blocks : list of ndarray, optional
        The blocks of ``conditional``.
    """

    name: str
    log_prob: Callable[[FloatArray], Any]
    grad: Callable[[FloatArray], FloatArray]
    initial: Distribution
    mean: FloatArray
    cov: FloatArray
    conditional: Callable[[FloatArray, int], Any] | None = None
    blocks: list[npt.NDArray[np.int64]] | None = None

    @property
    def ndim(self) -> int:
        """The dimension ``d``."""
        return len(self.mean)


def gaussian(d: int = 10, rho: float = 0.9) -> Example:
    r"""A correlated Gaussian :math:`\mathcal N(0, \Sigma)` with :math:`\Sigma_{ij} = \rho^{|i-j|}`.

    Its default initial distribution is :math:`\mathcal N(0, 4I)`.

    Parameters
    ----------
    d : int
        Dimension, at least 1.
    rho : float
        Correlation of neighbouring coordinates, in (-1, 1).

    Returns
    -------
    Example
    """
    if d < 1:
        raise ValueError(f"d must be at least 1, not {d}")
    if not -1 < rho < 1:
        raise ValueError(f"rho must be in (-1, 1), not {rho}")
    steps = np.arange(d)
    cov = float(rho) ** np.abs(steps[:, None] - steps[None, :])
    precision = np.linalg.inv(cov)
    return Example(
        "gaussian",
        functools.partial(_gaussian_log_prob, precision=precision),
        functools.partial(_gaussian_grad, precision=precision),
        Normal(np.zeros(d), 4 * np.eye(d)),
        np.zeros(d),
        cov,
    )


def banana() -> Example:
    r"""The banana of Heng & Jacob (2019): :math:`\log\pi(x) = -(1 - x_0)^2 - 10(x_1 - x_0^2)^2`.

    Exactly, :math:`x_0 \sim \mathcal N(1, 1/2)` and
    :math:`x_1 \mid x_0 \sim \mathcal N(x_0^2, 1/20)`, so the mean is (1, 1.5) and the
    covariance [[0.5, 1], [1, 2.55]]. Its default initial distribution is
    :math:`U[-5, 5]^2`.

    Returns
    -------
    Example
    """
    cov = np.array([[0.5, 1.0], [1.0, 2.55]])
    initial = Uniform([-5.0, -5.0], [5.0, 5.0])
    return Example("banana", _banana_log_prob, _banana_grad, initial, np.array([1.0, 1.5]), cov)


def mixture() -> Example:
    r"""The bimodal mixture :math:`\tfrac12\mathcal N(-4, 1) + \tfrac12\mathcal N(4, 1)`.

    Jacob, O'Leary & Atchade (2020), section 5.1. Its mean is 0 and its variance 17;
    its default initial distribution is :math:`\mathcal N(10, 10^2)`, theirs.

    Returns
    -------
    Example
    """
    initial = Normal([10.0], [[100.0]])
    return Example(
        "mixture", _mixture_log_prob, _mixture_grad, initial, np.zeros(1), np.array([[17.0]])
    )


def funnel(d: int = 10) -> Example:
    r"""Neal's funnel: :math:`v \sim \mathcal N(0, 9)` and :math:`x_i \mid v \sim \mathcal N(0, e^v)`.

    The funnel of Neal (2003). The point is :math:`(v, x_1, \dots, x_{d-1})`. The mean is
    0, and the coordinates are uncorrelated with variances 9 and :math:`E[e^v] = e^{9/2}`.
    Its default initial distribution is :math:`\mathcal N(0, \Sigma)`, with that
    covariance.

    Parameters
    ----------
    d : int
        Dimension, at least 2.

    Returns
    -------
    Example
    """
    if d < 2:
        raise ValueError(f"d must be at least 2, not {d}")
    cov = np.diag([9.0] + [np.exp(4.5)] * (d - 1))
    initial = Normal(np.zeros(d), cov)
    return Example("funnel", _funnel_log_prob, _funnel_grad, initial, np.zeros(d), cov)


# The pump-failure data of George, Makov & Smith (1993): failures and thousands of hours.
PUMP_FAILURES = np.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22])
PUMP_TIMES = np.array([94.32, 15.72, 62.88, 125.76, 5.24, 31.44, 1.05, 1.05, 2.10, 10.48])
# The prior: lambda_i ~ Gamma(ALPHA, rate beta), beta ~ Gamma(GAMMA, rate DELTA).
ALPHA, GAMMA, DELTA = 1.802, 0.01, 1.0


def pumps() -> Example:
    r"""The pump-failure model, as in Jacob, O'Leary & Atchade (2020), section 5.2.

    For 10 pumps, :math:`s_i \sim \mathrm{Poisson}(\lambda_i t_i)`,
    :math:`\lambda_i \sim \mathrm{Gamma}(1.802, \text{rate } \beta)` and
    :math:`\beta \sim \mathrm{Gamma}(0.01, \text{rate } 1)`; the point is
    :math:`(\lambda_1, \dots, \lambda_{10}, \beta)`. The Gibbs sampler draws each
    :math:`\lambda_i` from :math:`\mathrm{Gamma}(1.802 + s_i, \text{rate } \beta + t_i)` and
    :math:`\beta` from :math:`\mathrm{Gamma}(0.01 + 18.02, \text{rate } 1 + \sum\lambda_i)`,
    one coordinate per block. The exact moments come from quadrature over :math:`\beta`,
    whose posterior is one-dimensional once the :math:`\lambda_i` are integrated out. Its
    default initial distribution is :math:`U(0.001, 5)^{11}`.

    Returns
    -------
    Example
    """
    mean, cov = _pump_moments()
    return Example(
        "pumps",
        _pump_log_prob,
        _pump_grad,
        Uniform(np.full(11, 1e-3), np.full(11, 5.0)),
        mean,
        cov,
        _pump_conditional,
        [np.array([i]) for i in range(11)],
    )


# Two-operand einsums, unlike matrix products, give a row the same bits in any batch.
def _gaussian_log_prob(x: FloatArray, precision: FloatArray) -> Any:
    return -0.5 * np.einsum("...j,...j->...", np.einsum("...i,ij->...j", x, precision), x)


def _gaussian_grad(x: FloatArray, precision: FloatArray) -> FloatArray:
    grad: FloatArray = -np.einsum("...i,ij->...j", x, precision)
    return grad


def _banana_log_prob(x: FloatArray) -> Any:
    return -((1 - x[..., 0]) ** 2) - 10 * (x[..., 1] - x[..., 0] ** 2) ** 2


def _banana_grad(x: FloatArray) -> FloatArray:
    bend = x[..., 1] - x[..., 0] ** 2
    return np.stack([2 * (1 - x[..., 0]) + 40 * x[..., 0] * bend, -20 * bend], axis=-1)


def _mixture_log_prob(x: FloatArray) -> Any:
    # log(e^{-(x+4)^2/2} + e^{-(x-4)^2/2}) = -x^2/2 + log(e^{4x} + e^{-4x}) + constant.
    y = x[..., 0]
    return -0.5 * y**2 + np.logaddexp(4 * y, -4 * y)


def _mixture_grad(x: FloatArray) -> FloatArray:
    grad: FloatArray = -x + 4 * np.tanh(4 * x)
    return grad


def _funnel_log_prob(x: FloatArray) -> Any:
    v, rest = x[..., 0], x[..., 1:]
    spread = np.exp(-v)
    return -(v**2) / 18 - 0.5 * rest.shape[-1] * v - 0.5 * spread * np.sum(rest**2, axis=-1)


def _funnel_grad(x: FloatArray) -> FloatArray:
    v, rest = x[..., 0], x[..., 1:]
    spread = np.exp(-v)
    dv = -v / 9 - 0.5 * rest.shape[-1] + 0.5 * spread * np.sum(rest**2, axis=-1)
    return np.concatenate([dv[..., None], -spread[..., None] * rest], axis=-1)


def _pump_log_prob(x: FloatArray) -> Any:
    lam, beta = x[..., :10], x[..., 10]
    inside = np.all(x > 0, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (
            np.sum((ALPHA + PUMP_FAILURES - 1) * np.log(lam) - lam * PUMP_TIMES, axis=-1)
            + (10 * ALPHA + GAMMA - 1) * np.log(beta)
            - beta * (DELTA + np.sum(lam, axis=-1))
        )
    return np.where(inside, value, -np.inf)


def _pump_grad(x: FloatArray) -> FloatArray:
    lam, beta = x[..., :10], x[..., 10]
    dlam = (ALPHA + PUMP_FAILURES - 1) / lam - PUMP_TIMES - beta[..., None]
    dbeta = (10 * ALPHA + GAMMA - 1) / beta - DELTA - np.sum(lam, axis=-1)
    return np.concatenate([dlam, dbeta[..., None]], axis=-1)


# Python numbers: Gibbs kernels call the conditionals one row and block at a time, where
# NumPy's scalar arithmetic is several times slower.
_FAILURES, _TIMES = PUMP_FAILURES.tolist(), PUMP_TIMES.tolist()


def _pump_conditional(x: FloatArray, block: int) -> _Gamma:
    if block < 10:
        return _Gamma(ALPHA + _FAILURES[block], float(x[10]) + _TIMES[block])
    return _Gamma(GAMMA + 10 * ALPHA, DELTA + math.fsum(x[:10]))


class _Gamma:
    # A gamma distribution with the methods of a frozen SciPy one that Gibbs kernels call,
    # 100 times faster to build: Gibbs builds one per row, block and step.
    def __init__(self, shape: float, rate: float) -> None:
        self.shape, self.rate = shape, rate

    def rvs(self, *, random_state: np.random.Generator) -> float:
        return float(random_state.gamma(self.shape, 1 / self.rate))

    def logpdf(self, value: float) -> float:
        a, b = self.shape, self.rate
        if value <= 0:
            return -math.inf
        return a * math.log(b) - math.lgamma(a) + (a - 1) * math.log(value) - b * value

    def ppf(self, u: float) -> float:
        return float(special.gammaincinv(self.shape, u) / self.rate)


def _pump_moments() -> tuple[FloatArray, FloatArray]:
    # Given beta, the lambda_i are independent Gamma(a_i, rate beta + t_i), with
    # a_i = ALPHA + s_i, and integrating them out leaves
    # p(beta | s) ∝ beta^(GAMMA - 1 + 10 ALPHA) e^(-DELTA beta) prod (beta + t_i)^(-a_i).
    shape = ALPHA + PUMP_FAILURES

    def log_density(beta: float) -> float:
        return float(
            (GAMMA - 1 + 10 * ALPHA) * np.log(beta)
            - DELTA * beta
            - np.sum(shape * np.log(beta + PUMP_TIMES))
        )

    # Shifted by the log-density near the posterior mode (about 2.5), so that the
    # integrand stays near 1 where it matters.
    shift = log_density(2.5)

    def moments(beta: float) -> FloatArray:
        weight = np.exp(log_density(beta) - shift)
        rate = beta + PUMP_TIMES
        first = np.append(shape / rate, beta)  # E[x | beta]
        second = np.outer(first, first)
        second[np.arange(10), np.arange(10)] += shape / rate**2  # Var(lambda_i | beta)
        result: FloatArray = weight * np.concatenate([[1.0], first, second.ravel()])
        return result

    total, _ = integrate.quad_vec(moments, 0, np.inf, epsabs=0, epsrel=1e-12)
    total = total / total[0]
    mean = total[1:12]
    cov = total[12:].reshape(11, 11) - np.outer(mean, mean)
    return mean, cov
