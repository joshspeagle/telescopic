"""Independent-chain checks for batch means without exact stationary draws.

The warmed reference path uses the optional arviz-stats dependency.
"""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from benchmarks import targets
from telescopic._random import Streams
from telescopic._types import integer, real
from telescopic.initial import Normal
from telescopic.kernels import Kernel
from telescopic.target import Target


@dataclass(frozen=True)
class Reference:
    """Candidate variances, costs, and acceptance diagnostics for one kernel."""

    variances: dict[str, float]
    evaluations: int
    gradient_evaluations: int
    max_rhat: float | None
    min_bulk_ess: float | None
    min_tail_ess: float | None
    estimand_z: dict[str, float | None]
    batch_variance_ratio: dict[str, float | None]
    passed: bool
    diagnostics_version: str
    start_scale: float | None = None


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _batch_variance(values: targets.Array, batch_size: int) -> float:
    means = values.reshape(values.shape[0], -1, batch_size).mean(axis=-1)
    return float(batch_size * means.var(ddof=1))


def diagnose(
    case: targets.BenchmarkCase,
    samples: targets.Array,
    batch_size: int,
    density_evaluations: int,
    gradient_evaluations: int,
    start_scale: float | None = None,
) -> Reference:
    """Gate asymptotic variances from arrays of shape (chains, draws, dimension)."""
    arviz_stats = import_module("arviz_stats")
    # Constant chains can yield undefined diagnostics. The gate rejects them.
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = _finite(np.max(arviz_stats.rhat(samples, method="rank")))
        bulk = _finite(np.min(arviz_stats.ess(samples, method="bulk")))
        tail = _finite(np.min(arviz_stats.ess(samples, method="tail", prob=(0.05, 0.95))))
    variances = {}
    z_scores = {}
    ratios = {}
    for item in case.estimands:
        values = item.function(samples)
        variance = _batch_variance(values, batch_size)
        shorter = _batch_variance(values, batch_size // 2)
        uncertainty = np.hypot(np.sqrt(variance / values.size), item.mean_standard_error)
        z_scores[item.name] = (
            _finite((values.mean() - item.mean) / uncertainty) if uncertainty else None
        )
        ratios[item.name] = _finite(variance / shorter) if shorter > 0 else None
        variances[item.name] = variance

    passed = (
        rhat is not None
        and rhat <= 1.01
        and bulk is not None
        and bulk >= 400
        and tail is not None
        and tail >= 400
        and all(value is not None and abs(value) <= 5 for value in z_scores.values())
        and all(value is not None and 0.7 <= value <= 1.4 for value in ratios.values())
    )
    return Reference(
        variances if passed else {},
        density_evaluations + gradient_evaluations,
        gradient_evaluations,
        rhat,
        bulk,
        tail,
        z_scores,
        ratios,
        passed,
        getattr(arviz_stats, "__version__", "unknown"),
        start_scale,
    )


def run(
    case: targets.BenchmarkCase,
    kernel: Kernel,
    *,
    warmup: int,
    batches: int,
    batch_size: int,
    seed: int,
    chains: int = 4,
    start_scale: float | None = None,
) -> Reference:
    """Warm independent chains from the wide start or a scaled matched covariance."""
    warmup = integer("warmup", warmup, minimum=1)
    batches = integer("batches", batches, minimum=20)
    batch_size = integer("batch_size", batch_size, minimum=2)
    chains = integer("chains", chains, minimum=4)
    if batch_size % 2:
        raise ValueError("batch_size must be even so the shorter batches use the same draws")
    if not kernel.is_tuned:
        raise ValueError("uncoupled reference requires a frozen kernel")
    # Fail before transitions if the diagnostics dependency is unavailable.
    import_module("arviz_stats")
    initial = case.wide
    if start_scale is not None:
        start_scale = real("start_scale", start_scale, 0, np.inf)
        initial = Normal(case.matched.mean, start_scale**2 * case.matched.cov)
    streams = Streams(chains)
    positions = []
    for lane, lineage in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        start, transitions = lineage.spawn(2)
        positions.append(initial.rvs(random_state=np.random.default_rng(start)))
        streams.assign(lane, np.random.default_rng(transitions))
    draws = streams.view(np.arange(chains))
    target = Target(case.log_prob, grad=case.grad, vectorize=True)
    state = kernel.init_state(target, np.stack(positions))
    for _ in range(warmup):
        state, _ = kernel.step(target, state, draws)
    samples = np.empty((chains, batches * batch_size, case.dimension))
    for index in range(samples.shape[1]):
        state, _ = kernel.step(target, state, draws)
        samples[:, index] = state.x
    return diagnose(case, samples, batch_size, target.n_log_prob, target.n_grad, start_scale)
