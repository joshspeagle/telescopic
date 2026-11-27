"""A synthetic parallax hierarchy with nested one-dimensional quadrature."""

from functools import cache

import numpy as np
from scipy import integrate, optimize, special

from benchmarks import targets

Array = targets.Array
DATA_SEED = 85
LOG_SCALE_SD = 0.5


def _data(count: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(DATA_SEED)
    distance = rng.gamma(3.0, 1.25, count)
    error = np.linspace(0.12, 0.32, count)
    observed = 1 / distance + error * rng.standard_normal(count)
    return observed, error


def _inner_rule(order: int) -> tuple[Array, Array]:
    nodes, weights = special.roots_legendre(order)
    edges = np.linspace(-8.0, 8.0, 129)
    width = np.diff(edges) / 2
    centers = (edges[1:] + edges[:-1]) / 2
    return (
        (centers[:, None] + width[:, None] * nodes).ravel(),
        (width[:, None] * weights).ravel(),
    )


@cache
def _reference(
    count: int, order: int, outer_limit: float, outer_rtol: float
) -> tuple[Array, Array, Array, Array]:
    observed, error = _data(count)
    log_r, weights = _inner_rule(order)
    distance = np.exp(log_r)
    log_likelihood = -0.5 * ((observed[:, None] - 1 / distance) / error[:, None]) ** 2
    base = log_likelihood + 3 * log_r + np.log(weights)
    second_log_r = log_r**2

    def conditional(log_scale: float) -> tuple[Array, Array]:
        # dr/dlog_r is included in the power three.
        log_mass = base - 3 * log_scale - distance * np.exp(-log_scale)
        normalizers = special.logsumexp(log_mass, axis=1)
        return log_mass, normalizers

    def marginal(log_scale: float) -> float:
        _, normalizers = conditional(log_scale)
        return float(normalizers.sum() - 0.5 * (log_scale / LOG_SCALE_SD) ** 2)

    fit = optimize.minimize_scalar(
        lambda value: -marginal(value), bounds=(-outer_limit, outer_limit), method="bounded"
    )
    if not fit.success:
        raise RuntimeError("failed to locate the parallax scale posterior mode")
    anchor = -fit.fun

    def integrand(log_scale: float) -> Array:
        log_mass, normalizers = conditional(log_scale)
        marginal_log_mass = normalizers.sum() - 0.5 * (log_scale / LOG_SCALE_SD) ** 2
        marginal_mass = np.exp(marginal_log_mass - anchor)
        conditional_mass = np.exp(log_mass - normalizers[:, None])
        scale = np.exp(log_scale)
        return np.asarray(
            marginal_mass
            * np.concatenate(
                [
                    [1.0, log_scale, log_scale**2, scale, scale**2],
                    conditional_mass @ log_r,
                    conditional_mass @ second_log_r,
                    [conditional_mass[0] @ distance, conditional_mass[0] @ (distance**2)],
                ]
            ),
            dtype=np.float64,
        )

    integral, _, info = integrate.quad_vec(
        integrand,
        -outer_limit,
        outer_limit,
        epsrel=outer_rtol,
        points=[fit.x],
        full_output=True,
    )
    if not info.success:
        raise RuntimeError(f"parallax reference quadrature failed: {info.message}")
    moments = integral / integral[0]
    mean = np.concatenate([[moments[1]], moments[5 : 5 + count]])
    second = np.concatenate([[moments[2]], moments[5 + count : 5 + 2 * count]])
    physical_mean = moments[[3, -2]]
    physical_var = moments[[4, -1]] - physical_mean**2
    return mean, second - mean**2, physical_mean, physical_var


def parallax(
    dimension: int, *, order: int = 8, outer_limit: float = 5.0, outer_rtol: float = 1e-8
) -> targets.BenchmarkCase:
    """Build the parallax posterior in (log population scale, log distances).

    Distances and scale are in kpc, with a Gamma(shape=3, scale) distance
    prior. Measurements and errors are in mas. The log scale prior is
    Normal(0, 0.5**2). Log coordinates preserve positive physical support.
    """
    if dimension not in (51, 501):
        raise ValueError("parallax dimension must be 51 or 501")
    count = dimension - 1
    observed, error = _data(count)
    inverse_variance = 1 / error**2

    def log_prob(points: Array) -> Array:
        scale, log_r = points[..., 0], points[..., 1:]
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_log_r = log_r - scale[..., None]
            residual = observed - np.exp(-log_r)
            energy = np.exp(scaled_log_r) + 0.5 * residual**2 * inverse_variance
            terms = np.where(np.isposinf(energy), -np.inf, 3 * scaled_log_r - energy)
            return np.sum(terms, axis=-1) - 0.5 * (scale / LOG_SCALE_SD) ** 2

    def grad(points: Array) -> Array:
        scale, log_r = points[..., 0], points[..., 1:]
        score = np.empty_like(points)
        with np.errstate(over="ignore", invalid="ignore"):
            relative_distance = np.exp(log_r - scale[..., None])
            predicted = np.exp(-log_r)
            score[..., 0] = relative_distance.sum(axis=-1) - 3 * count - scale / LOG_SCALE_SD**2
            score[..., 1:] = (
                3 - relative_distance - (observed - predicted) * predicted * inverse_variance
            )
        return score

    mean, variance, physical_mean, physical_var = _reference(count, order, outer_limit, outer_rtol)
    return targets._case(
        "parallax",
        log_prob,
        grad,
        None,
        mean,
        np.diag(variance),
        (
            targets.Estimand(
                "population_scale_kpc",
                lambda points: np.exp(points[..., 0]),
                float(physical_mean[0]),
                float(physical_var[0]),
            ),
            targets.Estimand(
                "first_distance_kpc",
                lambda points: np.exp(points[..., 1]),
                float(physical_mean[1]),
                float(physical_var[1]),
            ),
        ),
        covariance_source="nested_quadrature_marginal_variances",
    )
