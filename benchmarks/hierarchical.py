"""An eight-schools-style target with a one-dimensional posterior reference."""

import numpy as np
from scipy import integrate, special

from benchmarks import targets
from telescopic._types import integer

Array = targets.Array
DATA_SEED = 91
MEAN_SCALE = 5.0
GROUP_SCALE = 5.0
LOG_GROUP_SCALE = np.log(GROUP_SCALE)


def normal_means(dimension: int, *, quadrature_rtol: float = 1e-9) -> targets.BenchmarkCase:
    """Build the centered normal-means posterior in coordinates (mean, log tau, effects).

    The dimensionless synthetic observations have known unequal measurement
    standard deviations from 0.7 to 2.0. The population mean has a normal
    prior and tau has a half-normal prior, both with standard deviation
    parameter 5. Reference moments integrate over tau after collapsing the
    normal coordinates. Exact draws use a separate rejection sampler.
    """
    dimension = integer("dimension", dimension, minimum=3)
    groups = dimension - 2
    sigma = np.linspace(0.7, 2.0, groups)
    sigma_squared = sigma**2
    effect_indices = np.arange(2, dimension)
    data_rng = np.random.default_rng(DATA_SEED)
    y = 1 + np.sqrt(4 + sigma_squared) * data_rng.standard_normal(groups)

    def log_prob(points: Array) -> Array:
        mean, log_tau = points[..., 0], points[..., 1]
        effects = points[..., 2:]
        with np.errstate(divide="ignore", over="ignore"):
            log_residual = np.log(np.abs(effects - mean[..., None])) - log_tau[..., None]
            energy = np.sum(((effects - y) / sigma) ** 2 + np.exp(2 * log_residual), axis=-1)
            energy += (mean / MEAN_SCALE) ** 2 + np.exp(2 * (log_tau - LOG_GROUP_SCALE))
        return -0.5 * energy - (groups - 1) * log_tau

    def grad(points: Array) -> Array:
        mean, log_tau = points[..., 0], points[..., 1]
        effects = points[..., 2:]
        residual = effects - mean[..., None]
        output = np.empty_like(points)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            log_residual = np.log(np.abs(residual)) - log_tau[..., None]
            precision_residual = np.copysign(np.exp(log_residual - log_tau[..., None]), residual)
            output[..., 0] = precision_residual.sum(axis=-1) - mean / MEAN_SCALE**2
            output[..., 1] = (
                np.sum(np.exp(2 * log_residual), axis=-1)
                - (groups - 1)
                - np.exp(2 * (log_tau - LOG_GROUP_SCALE))
            )
            output[..., 2:] = (y - effects) / sigma_squared - precision_residual
        return output

    def collapsed(tau: Array) -> tuple[Array, Array, Array, Array, Array]:
        variances = tau[..., None] ** 2 + sigma_squared
        precision = 1 / variances
        mean_variance = 1 / (1 / MEAN_SCALE**2 + precision.sum(axis=-1))
        mean = mean_variance * (precision * y).sum(axis=-1)
        log_det = np.log(variances).sum(axis=-1) + np.log(MEAN_SCALE**2 / mean_variance)
        quadratic = np.sum((y - mean[..., None]) ** 2 * precision, axis=-1)
        quadratic += (mean / MEAN_SCALE) ** 2
        return log_det, quadratic, mean, mean_variance, sigma_squared * precision

    # A fixed likelihood factor cancels when the integrated moments are normalized.
    baseline_det, baseline_quadratic, *_ = collapsed(np.asarray(1.0))

    def moments(tau: float) -> Array:
        log_det, quadratic, mean, mean_variance, effect_mean_weight = collapsed(np.asarray(tau))
        mass = np.exp(
            -0.5
            * (log_det - baseline_det + quadratic - baseline_quadratic + (tau / GROUP_SCALE) ** 2)
        )
        conditional_mean = np.concatenate(
            [[mean, np.log(tau)], y + effect_mean_weight * (mean - y)]
        )
        direction = np.concatenate([[1.0, 0.0], effect_mean_weight])
        second = np.outer(conditional_mean, conditional_mean)
        second += mean_variance * np.outer(direction, direction)
        second[effect_indices, effect_indices] += tau**2 * effect_mean_weight
        return np.asarray(
            mass * np.concatenate([[1.0], conditional_mean, second.ravel()]), dtype=np.float64
        )

    integral, _, info = integrate.quad_vec(
        moments, 0.0, np.inf, epsrel=quadrature_rtol, full_output=True
    )
    if not info.success:
        raise RuntimeError(f"normal-means reference quadrature failed: {info.message}")
    mean = integral[1 : 1 + dimension] / integral[0]
    covariance = integral[1 + dimension :].reshape(dimension, dimension) / integral[0]
    covariance -= np.outer(mean, mean)

    # The collapsed covariance increases with tau in positive semidefinite order,
    # so the log determinant increases and the quadratic form decreases.
    # Bound the log likelihood on each bin with these two endpoint values.
    edges = np.concatenate([np.linspace(0.0, 8.0, 321), np.linspace(8.25, 40.0, 128)])
    log_det, quadratic, *_ = collapsed(edges)
    log_bound = -0.5 * (log_det + np.append(quadratic[1:], 0.0))
    survival = np.append(2 * special.ndtr(-edges / GROUP_SCALE), 0.0)
    log_mass = log_bound + np.log(survival[:-1] - survival[1:])
    envelope_weights = np.exp(log_mass - special.logsumexp(log_mass))

    def draw_exact(rng: np.random.Generator, size: int) -> Array:
        """Draw dimensionless posterior points of shape (size, dimension)."""
        output = np.empty((size, dimension))
        filled = 0
        while filled < size:
            count = min(size - filled, 2048)
            bins = rng.choice(len(edges), size=count, p=envelope_weights)
            top = survival[bins]
            probability = top - rng.random(count) * (top - survival[bins + 1])
            tau = -GROUP_SCALE * special.ndtri(probability / 2)
            det, quadratic, center, variance, effect_mean_weight = collapsed(tau)
            accept = rng.random(count) < np.exp(-0.5 * (det + quadratic) - log_bound[bins])
            accept &= tau > 0
            width = int(accept.sum())
            row = output[filled : filled + width]
            row[:, 0] = center[accept] + np.sqrt(variance[accept]) * rng.standard_normal(width)
            row[:, 1] = np.log(tau[accept])
            conditional_weight = effect_mean_weight[accept]
            row[:, 2:] = y + conditional_weight * (row[:, 0, None] - y)
            row[:, 2:] += (
                tau[accept, None]
                * np.sqrt(conditional_weight)
                * rng.standard_normal((width, groups))
            )
            filled += width
        return output

    return targets._case(
        "normal_means",
        log_prob,
        grad,
        draw_exact,
        mean,
        covariance,
        (
            targets.Estimand(
                "log_tau_mean", lambda x: x[..., 1], float(mean[1]), float(covariance[1, 1])
            ),
            targets.Estimand(
                "first_effect_mean", lambda x: x[..., 2], float(mean[2]), float(covariance[2, 2])
            ),
        ),
    )
