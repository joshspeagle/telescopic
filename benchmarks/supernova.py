"""A synthetic Type Ia supernova distance-modulus benchmark."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import special

from benchmarks import targets

Array = targets.Array
DATA_SEED = 84
OMEGA_LOGIT_CENTER = float(special.logit(0.3))
OMEGA_LOGIT_SCALE = 0.9
W_SCALE = 0.5
OFFSET_SCALE = 0.3
SCATTER_SCALE = 0.6
MAG_FACTOR = 5 / np.log(10)


@dataclass(frozen=True)
class Distance:
    """Gauss-Legendre rule for luminosity distances at fixed redshifts."""

    def __init__(self, redshift: Array, order: int) -> None:
        nodes, weights = special.roots_legendre(order)
        log_redshift = np.log1p(redshift[:, None] * (nodes + 1) / 2)
        object.__setattr__(self, "log_redshift", log_redshift)
        object.__setattr__(self, "log_weights", np.log(weights / 2))
        object.__setattr__(self, "log_factor", np.log((1 + redshift) * redshift))

    log_redshift: Array
    log_weights: Array
    log_factor: Array

    def _terms(self, logit: Array, w: Array) -> tuple[Array, Array, Array, Array]:
        q = logit[..., None, None]
        matter = -np.logaddexp(0, -q) + 3 * self.log_redshift
        dark = -np.logaddexp(0, q) + 3 * (1 + w[..., None, None]) * self.log_redshift
        expansion = np.logaddexp(matter, dark)
        return self.log_weights - 0.5 * expansion, expansion, matter, dark

    def magnitude(self, logit: Array, w: Array) -> Array:
        """Return distance magnitudes of shape (..., n)."""
        terms, _, _, _ = self._terms(logit, w)
        return np.asarray(MAG_FACTOR * (self.log_factor + special.logsumexp(terms, axis=-1)))

    def evaluate(self, logit: Array, w: Array) -> tuple[Array, Array, Array]:
        """Return magnitudes and derivatives with respect to logit and w."""
        terms, expansion, matter, dark = self._terms(logit, w)
        integral = special.logsumexp(terms, axis=-1)
        derivative_weight = -0.5 * np.exp(terms - integral[..., None])
        logit_score = np.exp(matter - expansion) - special.expit(logit)[..., None, None]
        w_score = np.exp(dark - expansion) * (3 * self.log_redshift)
        return (
            MAG_FACTOR * (self.log_factor + integral),
            MAG_FACTOR * (derivative_weight * logit_score).sum(axis=-1),
            MAG_FACTOR * (derivative_weight * w_score).sum(axis=-1),
        )


def _logit(coordinate: Array) -> Array:
    return np.asarray(OMEGA_LOGIT_CENTER + OMEGA_LOGIT_SCALE * coordinate)


def _omega(coordinate: Array) -> Array:
    return special.expit(_logit(coordinate))


def _w(coordinate: Array) -> Array:
    return -1 + W_SCALE * coordinate


def _scatter_variance(coordinate: Array) -> Array:
    return 0.01 * np.exp(2 * SCATTER_SCALE * coordinate)


def _data() -> tuple[Array, Array, Array]:
    redshift = np.geomspace(0.02, 1.1, 32)
    sigma = np.linspace(0.09, 0.14, len(redshift))
    magnitude = Distance(redshift, 64).magnitude(
        np.asarray(special.logit(0.31)), np.asarray(-0.93)
    )
    y = (
        magnitude
        + 0.06
        + np.sqrt(sigma**2 + 0.07**2)
        * np.random.default_rng(DATA_SEED).standard_normal(len(redshift))
    )
    return redshift, y, sigma**2


def _quadrature(
    distances: Distance,
    y: Array,
    measurement_variance: Array,
    order: int,
    center: Array,
    factor: Array,
) -> tuple[Array, Array, Array, Array]:
    nodes, weights = special.roots_hermitenorm(order)
    log_weights = np.log(weights)
    first = np.repeat(nodes, order)
    second = np.tile(nodes, order)
    cosmology_weights = np.repeat(log_weights, order) + np.tile(log_weights, order)
    block_weights = []
    block_moments = []
    block_second_moments = []
    for begin in range(0, len(first), 128):
        selection = slice(begin, begin + 128)
        first_block, second_block = first[selection], second[selection]
        u = center[0] + factor[0, 0] * first_block
        v = center[1] + factor[1, 0] * first_block + factor[1, 1] * second_block
        t = (
            center[2]
            + (factor[2, 0] * first_block + factor[2, 1] * second_block)[:, None]
            + factor[2, 2] * nodes
        )
        omega, w = _omega(u), _w(v)
        magnitude = distances.magnitude(_logit(u), w)
        residual = y - magnitude
        variance = measurement_variance + _scatter_variance(t[..., None])
        precision = 1 / variance
        offset_variance = 1 / (1 + OFFSET_SCALE**2 * precision.sum(axis=-1))
        offset_mean = OFFSET_SCALE * (residual[:, None, :] * precision).sum(axis=-1)
        offset_mean *= offset_variance
        adjusted = residual[:, None, :] - OFFSET_SCALE * offset_mean[..., None]
        quadratic = (adjusted**2 * precision).sum(axis=-1) + offset_mean**2
        # The constant affine Jacobian cancels in the normalized moments.
        prior_ratio = -0.5 * (
            (u**2 + v**2 - first_block**2 - second_block**2)[:, None] + t**2 - nodes**2
        )
        log_mass = (
            -0.5 * (np.log(variance).sum(axis=-1) - np.log(offset_variance) + quadratic)
            + cosmology_weights[selection, None]
            + log_weights
            + prior_ratio
        ).ravel()
        block_log_mass = special.logsumexp(log_mass)
        normalized = np.exp(log_mass - block_log_mass)
        coords = np.stack(
            np.broadcast_arrays(
                u[:, None], v[:, None], offset_mean, t, omega[:, None], w[:, None]
            ),
            axis=-1,
        ).reshape(-1, 6)
        second_moments = (coords.T * normalized) @ coords
        second_moments[2, 2] += normalized @ offset_variance.ravel()
        block_weights.append(block_log_mass)
        block_moments.append(normalized @ coords)
        block_second_moments.append(second_moments)
    block_log_weights = np.asarray(block_weights)
    normalized = np.exp(block_log_weights - special.logsumexp(block_log_weights))
    mean = normalized @ np.stack(block_moments)
    second_moments = np.einsum("n,nij->ij", normalized, np.stack(block_second_moments))
    covariance = second_moments - np.outer(mean, mean)
    return mean[:4], covariance[:4, :4], mean[4:], np.diag(covariance)[4:]


@cache
def _reference(order: int, distance_order: int) -> tuple[Array, Array, Array, Array]:
    redshift, y, measurement_variance = _data()
    distances = Distance(redshift, distance_order)
    center, covariance, _, _ = _quadrature(
        distances, y, measurement_variance, 36, np.zeros(3), np.eye(3)
    )
    axes = np.array([0, 1, 3])
    factor = np.linalg.cholesky(1.5 * covariance[np.ix_(axes, axes)])
    return _quadrature(distances, y, measurement_variance, order, center[axes], factor)


def supernova(
    dimension: int, *, order: int = 128, distance_order: int = 24
) -> targets.BenchmarkCase:
    """Build the four-coordinate flat wCDM posterior and quadrature reference.

    The coordinates map standard normal priors to matter density in (0, 1),
    equation of state w, an additive magnitude offset and positive intrinsic
    scatter. Hubble-constant and zero-point factors are absorbed in the offset.
    """
    if dimension != 4:
        raise ValueError("supernova dimension must be 4")
    redshift, y, measurement_variance = _data()
    distances = Distance(redshift, distance_order)

    log_measurement_variance = np.log(measurement_variance)

    def log_prob(points: Array) -> Array:
        with np.errstate(over="ignore", invalid="ignore"):
            prior_energy = (points**2).sum(axis=-1)
            usable = np.isfinite(prior_energy)
            bounded = np.where(usable[..., None], points, 0.0)
            magnitude = distances.magnitude(_logit(bounded[..., 0]), _w(bounded[..., 1]))
            residual = y - magnitude - OFFSET_SCALE * bounded[..., 2, None]
            log_intrinsic = np.log(0.01) + 2 * SCATTER_SCALE * bounded[..., 3, None]
            log_variance = np.logaddexp(log_measurement_variance, log_intrinsic)
            scaled = residual * np.exp(-0.5 * log_variance)
            return np.asarray(-0.5 * ((log_variance + scaled**2).sum(axis=-1) + prior_energy))

    def grad(points: Array) -> Array:
        magnitude, logit_gradient, w_gradient = distances.evaluate(
            _logit(points[..., 0]), _w(points[..., 1])
        )
        residual = y - magnitude - OFFSET_SCALE * points[..., 2, None]
        log_intrinsic = np.log(0.01) + 2 * SCATTER_SCALE * points[..., 3, None]
        log_variance = np.logaddexp(log_measurement_variance, log_intrinsic)
        root_precision = np.exp(-0.5 * log_variance)
        scaled = residual * root_precision
        weighted = scaled * root_precision
        intrinsic_fraction = np.exp(log_intrinsic - log_variance)
        score = np.stack(
            [
                (weighted * logit_gradient).sum(axis=-1) * OMEGA_LOGIT_SCALE,
                (weighted * w_gradient).sum(axis=-1) * W_SCALE,
                weighted.sum(axis=-1) * OFFSET_SCALE,
                (intrinsic_fraction * (scaled**2 - 1)).sum(axis=-1) * SCATTER_SCALE,
            ],
            axis=-1,
        )
        return score - points

    mean, covariance, physical_mean, physical_variance = _reference(order, distance_order)
    return targets._case(
        "supernova",
        log_prob,
        grad,
        None,
        mean,
        covariance,
        (
            targets.Estimand(
                "omega_m",
                lambda points: _omega(points[..., 0]),
                float(physical_mean[0]),
                float(physical_variance[0]),
            ),
            targets.Estimand(
                "w",
                lambda points: _w(points[..., 1]),
                float(physical_mean[1]),
                float(physical_variance[1]),
            ),
        ),
        covariance_source="gauss_hermite_quadrature",
    )
