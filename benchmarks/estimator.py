"""Measure fixed-replicate coverage and cost for a frozen random walk.

Run with PYTHONPATH=src python -m benchmarks.estimator. Fix the averaging
window and iteration cap before evaluation. When any pair is censored, retain
its cost and warnings and withhold the affected run's intervals. Reference
chains start at stationarity when the benchmark supplies exact draws.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from benchmarks import research, targets
from benchmarks.run import setup
from telescopic import diagnostics, testing
from telescopic._types import integer
from telescopic.engine import Settings, check_settings, run
from telescopic.initial import Normal
from telescopic.kernels import Kernel
from telescopic.results import Result
from telescopic.target import Target


@dataclass(frozen=True)
class IntervalRecord:
    """One estimand's replicate values and whether its 95% interval covers."""

    replicates: list[float]
    covered: bool


@dataclass(frozen=True)
class RunRecord:
    """Costs and warnings for one run, including runs with censored pairs."""

    evaluations: int
    censored: int
    warnings: list[str]
    assessed_warnings: list[str]
    estimates: dict[str, IntervalRecord]
    gradient_evaluations: int = 0


def random_walk_runs(
    case: targets.BenchmarkCase,
    settings: Settings,
    *,
    start: Literal["matched", "wide"],
    proposal: Literal["oracle", "identity"],
    runs: int,
    replicates: int,
    seed: int,
) -> list[RunRecord]:
    """Run independent fixed-size experiments with the frozen random walk."""
    initial, kernel = research.random_walk_setup(case, start, proposal)
    return run_frozen(case, settings, initial, kernel, runs=runs, replicates=replicates, seed=seed)


def run_frozen(
    case: targets.BenchmarkCase,
    settings: Settings,
    initial: Normal,
    kernel: Kernel,
    *,
    runs: int,
    replicates: int,
    seed: int,
) -> list[RunRecord]:
    """Run independent fixed-size experiments with an explicit iteration cap."""
    if not kernel.is_tuned:
        raise ValueError("estimator benchmarks require a frozen kernel")
    runs = integer("runs", runs, minimum=1)
    replicates = integer("replicates", replicates, minimum=2)
    settings = check_settings(settings)
    if settings.keep is not None or not settings.store or settings.max_iter is None:
        raise ValueError("estimator benchmarks require full storage and an iteration cap")
    target = Target(case.log_prob, grad=case.grad, vectorize=True)
    indices = np.arange(replicates)
    records = []
    for child in np.random.SeedSequence(seed).spawn(runs):
        output = run(
            target,
            kernel,
            initial,
            indices,
            k=settings.k,
            m=settings.m,
            lag=settings.lag,
            thin=settings.thin,
            max_iter=settings.max_iter,
            rng=child,
        )
        result = Result(output, kernel=kernel, initial=initial)
        warnings = [issue.kind for issue in diagnostics.issues(result)]
        assessed_warnings = []
        if result.n_replicates >= 100:
            assessed_warnings.append("variance")
        if result.n_replicates >= 500 and not np.isnan(result.diagnostics()["tail_index"]):
            assessed_warnings.append("tail")
        estimates = {}
        if not result.n_censored:
            for estimand in case.estimands:
                estimate = result.estimate(estimand.function)
                lower, upper = estimate.interval
                estimates[estimand.name] = IntervalRecord(
                    estimate.replicates.tolist(), bool(lower <= estimand.mean <= upper)
                )
        records.append(
            RunRecord(
                result.n_log_prob + result.n_grad,
                result.n_censored,
                warnings,
                assessed_warnings,
                estimates,
                result.n_grad,
            )
        )
    return records


def batch_means_reference(
    case: targets.BenchmarkCase,
    kernel: Kernel,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, float], int]:
    """Estimate asymptotic variances from an independently seeded stationary chain.

    Each batch consists of consecutive uncoupled transitions. The returned
    cost is the number of density evaluations for this reference calculation.
    """
    batches = integer("batches", batches, minimum=2)
    batch_size = integer("batch_size", batch_size, minimum=1)
    rng = np.random.default_rng(seed)
    target = Target(case.log_prob, grad=case.grad, vectorize=True)
    if case.draw_exact is None:
        raise ValueError("stationary batch means require a target with exact draws")
    point = case.draw_exact(rng, 1)
    state = kernel.init_state(target, point)
    draws = testing.random_draws(1, rng)
    block = np.empty((batch_size, case.dimension))
    means = np.empty((batches, len(case.estimands)))
    for batch in range(batches):
        for index in range(batch_size):
            state, _ = kernel.step(target, state, draws)
            block[index] = state.x[0]
        means[batch] = [estimand.function(block).mean() for estimand in case.estimands]
    variance = batch_size * means.var(axis=0, ddof=1)
    return dict(
        zip((e.name for e in case.estimands), variance.tolist(), strict=True)
    ), target.n_log_prob + target.n_grad


def summarize(
    records: list[RunRecord],
    estimand: targets.Estimand,
    *,
    asymptotic_variance: float | None = None,
    setup_evaluations: int = 0,
) -> dict[str, object]:
    """Score available intervals and report full-experiment metrics if all runs completed."""
    if not records:
        raise ValueError("records must contain at least one run")
    intervals = [record.estimates[estimand.name] for record in records if not record.censored]
    total_evaluations = sum(record.evaluations for record in records)
    covered = sum(interval.covered for interval in intervals)
    warning_tables = {}
    for warning in ("variance", "tail"):
        table = {"covered": 0, "missed": 0, "unavailable": 0}
        for record in records:
            if warning in record.warnings:
                if record.censored:
                    table["unavailable"] += 1
                else:
                    outcome = "covered" if record.estimates[estimand.name].covered else "missed"
                    table[outcome] += 1
        warning_tables[warning] = table
    report: dict[str, object] = {
        "runs": len(records),
        "available_intervals": len(intervals),
        "censored_runs": len(records) - len(intervals),
        "censored_pairs": sum(record.censored for record in records),
        "target_mean": estimand.mean,
        "target_mean_standard_error": estimand.mean_standard_error,
        "coverage_target": "reference_center" if estimand.mean_standard_error else "exact_mean",
        "target_variance": estimand.variance,
        "covered_intervals": covered,
        "sampling_evaluations": total_evaluations,
        "sampling_gradient_evaluations": sum(record.gradient_evaluations for record in records),
        "setup_evaluations": setup_evaluations,
        "total_evaluations": total_evaluations + setup_evaluations,
        "warning_counts_by_interval_outcome": warning_tables,
        "warning_eligible_runs": {
            warning: sum(warning in record.assessed_warnings for record in records)
            for warning in warning_tables
        },
        "coverage": None,
        "coverage_standard_error": None,
        "pooled_mean": None,
        "pooled_standard_error": None,
        "pooled_z": None,
        "inefficiency_over_target_variance": None,
        "inefficiency_over_asymptotic_variance": None,
        "inefficiency_over_target_variance_without_setup": None,
        "inefficiency_over_asymptotic_variance_without_setup": None,
    }
    if len(intervals) == len(records):
        values = np.concatenate([interval.replicates for interval in intervals])
        variance = float(values.var(ddof=1))
        standard_error = float(np.sqrt(variance / len(values)))
        sampling_times_variance = total_evaluations / len(values) * variance
        cost_times_variance = (total_evaluations + setup_evaluations) / len(values) * variance
        combined_error = np.hypot(standard_error, estimand.mean_standard_error)
        coverage = covered / len(records)
        report.update(
            coverage=coverage,
            coverage_standard_error=float(np.sqrt(coverage * (1 - coverage) / len(records))),
            pooled_mean=float(values.mean()),
            pooled_standard_error=standard_error,
            pooled_z=float((values.mean() - estimand.mean) / combined_error)
            if combined_error > 0
            else None,
            inefficiency_over_target_variance_without_setup=(
                sampling_times_variance / estimand.variance if estimand.variance > 0 else None
            ),
            inefficiency_over_asymptotic_variance_without_setup=(
                sampling_times_variance / asymptotic_variance
                if asymptotic_variance is not None and asymptotic_variance > 0
                else None
            ),
            inefficiency_over_asymptotic_variance=(
                cost_times_variance / asymptotic_variance
                if asymptotic_variance is not None and asymptotic_variance > 0
                else None
            ),
            inefficiency_over_target_variance=(
                cost_times_variance / estimand.variance if estimand.variance > 0 else None
            ),
        )
    return report


def main() -> None:
    """Print a JSON experiment record with independently seeded runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=research.TARGETS)
    parser.add_argument("dimension", type=int)
    parser.add_argument("--start", choices=("matched", "wide"), default="matched")
    parser.add_argument("--proposal", choices=("oracle", "identity"), default="oracle")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    for setting in ("k", "m", "lag", "max_iter"):
        parser.add_argument("--" + setting.replace("_", "-"), type=int, required=True)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--reference-batches", type=int, default=100)
    parser.add_argument("--reference-batch-size", type=int, default=1000)
    args = parser.parse_args()
    research.require_source_checkout()
    if args.reference_seed == args.seed:
        parser.error("reference and estimator seeds must differ")
    settings = Settings(args.k, args.m, args.lag, 1, args.max_iter, True)
    with research.threadpoolctl.threadpool_limits(limits=1):
        case = research.TARGETS[args.target](args.dimension)
        if args.reference_seed is not None and case.draw_exact is None:
            parser.error("reference-seed requires a benchmark with exact stationary draws")
        records = random_walk_runs(
            case,
            settings,
            start=args.start,
            proposal=args.proposal,
            runs=args.runs,
            replicates=args.replicates,
            seed=args.seed,
        )
        reference: dict[str, float] = {}
        reference_cost = 0
        if args.reference_seed is not None:
            _, kernel = research.random_walk_setup(case, args.start, args.proposal)
            reference, reference_cost = batch_means_reference(
                case,
                kernel,
                batches=args.reference_batches,
                batch_size=args.reference_batch_size,
                seed=args.reference_seed,
            )
        report = {
            **setup(),
            **vars(args),
            "method": "random_walk",
            "setup_evaluations": 0,
            "matched_covariance_source": case.covariance_source,
            "reference_asymptotic_variance": reference,
            "reference_evaluations": reference_cost,
            "summary": {
                estimand.name: summarize(
                    records, estimand, asymptotic_variance=reference.get(estimand.name)
                )
                for estimand in case.estimands
            },
            "records": [asdict(record) for record in records],
        }
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
