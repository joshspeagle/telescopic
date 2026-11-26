"""Synthetic benchmark targets with exact draws and expectations."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp, ndtr

from telescopic import examples
from telescopic._types import integer, real
from telescopic.initial import Normal

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Estimand:
    """A function on unitless points of shape (..., d) and its reference moments."""

    name: str
    function: Callable[[Array], Array]
    mean: float
    variance: float
    mean_standard_error: float = 0.0


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
    draw_exact: Callable[[np.random.Generator, int], Array] | None
    matched: Normal
    wide: Normal
    estimands: tuple[Estimand, Estimand]
    covariance_source: str = "reference_moments"
    mode_probabilities: tuple[Estimand, ...] = ()

    @property
    def scored_estimands(self) -> tuple[Estimand, ...]:
        """Include every mode region while scoring shared estimands once."""
        names = {item.name for item in self.estimands}
        return self.estimands + tuple(
            item for item in self.mode_probabilities if item.name not in names
        )


def _case(
    name: str,
    log_prob: Callable[[Array], Array],
    grad: Callable[[Array], Array],
    draw_exact: Callable[[np.random.Generator, int], Array] | None,
    mean: Array,
    cov: Array,
    estimands: tuple[Estimand, Estimand],
    *,
    covariance_source: str = "reference_moments",
    mode_probabilities: tuple[Estimand, ...] = (),
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
        covariance_source,
        mode_probabilities,
    )


def rotated_gaussian(dimension: int, *, seed: int = 91) -> BenchmarkCase:
    """Build a rotated Gaussian with covariance condition number 10,000."""
    dimension = integer("dimension", dimension, minimum=2)
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

    dimension = integer("dimension", dimension, minimum=2)

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
            Estimand(
                "first_mean", lambda points: points[..., 0], float(mean[0]), float(cov[0, 0])
            ),
            Estimand(
                "second_mean", lambda points: points[..., 1], float(mean[1]), float(cov[1, 1])
            ),
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
    dimension = integer("dimension", dimension, minimum=2)
    separation = real("separation", separation, 0, np.inf)
    narrow_scale = real("narrow_scale", narrow_scale, 0, np.inf)
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
    right_probability = 1 - left_probability
    bernoulli_variance = left_probability * right_probability
    left = Estimand(
        "left_probability",
        lambda points: (points[..., 0] < 0).astype(np.float64),
        left_probability,
        bernoulli_variance,
    )
    right = Estimand(
        "right_probability",
        lambda points: (points[..., 0] >= 0).astype(np.float64),
        right_probability,
        bernoulli_variance,
    )
    first = Estimand("first_mean", lambda points: points[..., 0], float(mean[0]), float(cov[0, 0]))
    return _case(
        "mixture",
        log_prob,
        grad,
        draw_exact,
        mean,
        cov,
        (first, left),
        mode_probabilities=(left, right),
    )


def four_mode(dimension: int) -> BenchmarkCase:
    """Build a four-quadrant mixture with a curved upper-right component.

    The curved component scales its second Gaussian coordinate by
    exp(0.1 * z**2), where z is the first centered coordinate. This bends
    the conditional mean while preserving the sign of the second coordinate.
    """
    dimension = integer("dimension", dimension, minimum=2)
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    log_weights = np.log(weights)
    centers = np.zeros((4, dimension))
    centers[:, :2] = [[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]]
    curvature = np.array([0.0, 0.0, 0.0, 0.1])

    def terms(points: Array) -> tuple[Array, Array, Array, Array]:
        residual = points[..., None, :] - centers
        first = residual[..., 0]
        log_scale = curvature * first**2
        inverse_scale = np.exp(-log_scale)
        latent_second = points[..., 1, None] * inverse_scale
        residual[..., 1] = latent_second - centers[:, 1]
        component = log_weights - log_scale - 0.5 * np.sum(residual**2, axis=-1)
        return component, residual, inverse_scale, latent_second

    def log_prob(points: Array) -> Array:
        component, _, _, _ = terms(points)
        return np.asarray(logsumexp(component, axis=-1), dtype=np.float64)

    def grad(points: Array) -> Array:
        component, residual, inverse_scale, latent_second = terms(points)
        responsibilities = np.exp(component - logsumexp(component, axis=-1, keepdims=True))
        scores = -residual
        scores[..., 0] += 2 * curvature * residual[..., 0] * (residual[..., 1] * latent_second - 1)
        scores[..., 1] *= inverse_scale
        return np.sum(responsibilities[..., None] * scores, axis=-2)

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        component = rng.choice(4, size=size, p=weights)
        residual = rng.standard_normal((size, dimension))
        points = centers[component] + residual
        points[:, 1] *= np.exp(curvature[component] * residual[:, 0] ** 2)
        return points

    # E[exp(a Z**2)] = (1 - 2a)**(-1/2) for a standard normal Z.
    component_means = centers.copy()
    component_means[:, 1] /= np.sqrt(1 - 2 * curvature)
    second_moment = (1 + centers[:, 1] ** 2) / np.sqrt(1 - 4 * curvature)
    second_variance = second_moment - component_means[:, 1] ** 2
    mean = weights @ component_means
    offset = component_means - mean
    cov = np.eye(dimension) + (offset.T * weights) @ offset
    cov[1, 1] += float(weights @ second_variance) - 1
    modes = []
    for name, positive in (
        ("lower_left", (False, False)),
        ("lower_right", (True, False)),
        ("upper_left", (False, True)),
        ("upper_right", (True, True)),
    ):
        signs = np.where(positive, 1, -1)
        probability = float(weights @ np.prod(ndtr(centers[:, :2] * signs), axis=1))

        def membership(points: Array, positive: tuple[bool, bool] = positive) -> Array:
            return np.all((points[..., :2] >= 0) == positive, axis=-1).astype(np.float64)

        modes.append(
            Estimand(
                f"{name}_probability", membership, probability, probability * (1 - probability)
            )
        )
    return _case(
        "four_mode",
        log_prob,
        grad,
        draw_exact,
        mean,
        cov,
        (modes[0], modes[3]),
        mode_probabilities=tuple(modes),
    )
