"""Synthetic benchmark targets with exact draws and expectations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, ndtr

from telescopic import examples
from telescopic.initial import Normal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Estimand:
    """A function on unitless points of shape (..., d) and its exact moments."""

    name: str
    function: Callable[[Array], Array]
    mean: float
    variance: float


@dataclass(frozen=True)
class BenchmarkCase:
    """A target, two Gaussian starts, and sampler-independent references.

    Target callables accept unitless points of shape (..., d). Exact draws
    consume an explicit generator and return unitless samples of shape (n, d).
    """

    name: str
    dimension: int
    log_prob: Callable[[Array], Array]
    grad: Callable[[Array], Array]
    draw_exact: Callable[[np.random.Generator, int], Array]
    matched: Normal
    wide: Normal
    estimands: tuple[Estimand, Estimand]


def _case(
    name: str,
    log_prob: Callable[[Array], Array],
    grad: Callable[[Array], Array],
    draw_exact: Callable[[np.random.Generator, int], Array],
    mean: Array,
    cov: Array,
    estimands: tuple[Estimand, Estimand],
) -> BenchmarkCase:
    return BenchmarkCase(
        name,
        len(mean),
        log_prob,
        grad,
        draw_exact,
        Normal(mean, cov),
        Normal(mean, 100.0 * cov),
        estimands,
    )


def rotated_gaussian(dimension: int, *, seed: int = 91) -> BenchmarkCase:
    """Build a rotated Gaussian with covariance condition number 10,000."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    eigenvalues = np.geomspace(1.0, 1e-4, dimension)
    covariance = (basis * eigenvalues) @ basis.T
    root = basis * np.sqrt(eigenvalues)
    whitening = basis / np.sqrt(eigenvalues)

    def log_prob(points: Array) -> Array:
        return -0.5 * np.sum((points @ whitening) ** 2, axis=-1)

    def grad(points: Array) -> Array:
        return -((points @ whitening) @ whitening.T)

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        return rng.standard_normal((size, dimension)) @ root.T

    slow = basis[:, 0]
    fast = basis[:, -1]
    return _case(
        "rotated_gaussian",
        log_prob,
        grad,
        draw_exact,
        np.zeros(dimension),
        covariance,
        (
            Estimand("slow_projection", lambda points: points @ slow, 0.0, 1.0),
            Estimand("fast_projection", lambda points: points @ fast, 0.0, 1e-4),
        ),
    )


def banana(dimension: int) -> BenchmarkCase:
    """Build the quadratic banana with conditionally independent coordinates."""

    def log_prob(points: Array) -> Array:
        first = points[..., 0]
        residual = points[..., 1:] - first[..., None] ** 2
        return -((first - 1) ** 2) - 10 * np.sum(residual**2, axis=-1)

    def grad(points: Array) -> Array:
        first = points[..., 0]
        residual = points[..., 1:] - first[..., None] ** 2
        first_grad = -2 * (first - 1) + 40 * first * np.sum(residual, axis=-1)
        return np.concatenate([first_grad[..., None], -20 * residual], axis=-1)

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        first = rng.normal(1.0, np.sqrt(0.5), (size, 1))
        rest = first**2 + rng.normal(0.0, np.sqrt(0.05), (size, dimension - 1))
        return np.concatenate([first, rest], axis=-1)

    mean = np.full(dimension, 1.5)
    mean[0] = 1.0
    cov = np.full((dimension, dimension), 2.5)
    cov[0, :] = cov[:, 0] = 1.0
    cov[0, 0] = 0.5
    cov[1:, 1:] += 0.05 * np.eye(dimension - 1)
    return _case(
        "banana",
        log_prob,
        grad,
        draw_exact,
        mean,
        cov,
        (
            Estimand("first_mean", lambda points: points[..., 0], 1.0, 0.5),
            Estimand("second_mean", lambda points: points[..., 1], 1.5, 2.55),
        ),
    )


def funnel(dimension: int) -> BenchmarkCase:
    """Build the centered funnel with estimands of its log variance."""
    target = examples.funnel(dimension)

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        log_variance = rng.normal(0.0, 3.0, (size, 1))
        rest = np.exp(log_variance / 2) * rng.standard_normal((size, dimension - 1))
        return np.concatenate([log_variance, rest], axis=-1)

    return _case(
        "funnel",
        target.log_prob,
        target.grad,
        draw_exact,
        target.mean,
        target.cov,
        (
            Estimand("log_variance_mean", lambda points: points[..., 0], 0.0, 9.0),
            Estimand("log_variance_square", lambda points: points[..., 0] ** 2, 9.0, 162.0),
        ),
    )


def mixture(
    dimension: int, *, separation: float = 12.0, narrow_scale: float = 0.1
) -> BenchmarkCase:
    """Build a two-mode Gaussian mixture with one spherical component per mode."""
    weights = np.array([0.7, 0.3])
    scales = np.array([1.0, narrow_scale])
    centers = np.zeros((2, dimension))
    centers[:, 0] = [-separation / 2, separation / 2]
    normalizers = np.log(weights) - dimension * np.log(scales)

    def log_components(points: Array) -> Array:
        standardized = (points[..., None, :] - centers) / scales[:, None]
        return np.asarray(normalizers - 0.5 * np.sum(standardized**2, axis=-1), dtype=np.float64)

    def log_prob(points: Array) -> Array:
        return np.asarray(logsumexp(log_components(points), axis=-1), dtype=np.float64)

    def grad(points: Array) -> Array:
        terms = log_components(points)
        responsibilities = np.exp(terms - logsumexp(terms, axis=-1, keepdims=True))
        scores = (centers - points[..., None, :]) / scales[:, None] ** 2
        return np.sum(responsibilities[..., None] * scores, axis=-2)

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        component = rng.choice(2, size=size, p=weights)
        return centers[component] + scales[component, None] * rng.standard_normal(
            (size, dimension)
        )

    mean = weights @ centers
    residual = centers - mean
    cov = (weights @ scales**2) * np.eye(dimension) + (residual.T * weights) @ residual
    left_probability = float(weights @ ndtr(-centers[:, 0] / scales))
    return _case(
        "mixture",
        log_prob,
        grad,
        draw_exact,
        mean,
        cov,
        (
            Estimand(
                "first_mean", lambda points: points[..., 0], float(mean[0]), float(cov[0, 0])
            ),
            Estimand(
                "left_probability",
                lambda points: (points[..., 0] < 0).astype(np.float64),
                left_probability,
                left_probability * (1 - left_probability),
            ),
        ),
    )
