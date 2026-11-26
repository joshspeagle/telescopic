"""Compare frozen random walk and automatically tuned jittered HMC on benchmark cases.

Run with PYTHONPATH=src python -m benchmarks.compare. Each pointwise density
evaluation and each pointwise gradient evaluation counts as one unit. The
count matches target calls, and different costs per call need not imply
comparable elapsed time. Tuning is charged once per case and method.
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Literal

from benchmarks import estimator, research, targets
from benchmarks.run import setup
from telescopic import CoupledSampler, TelescopicWarning
from telescopic.engine import Settings
from telescopic.kernels import HMC, Kernel

Method = Literal["random_walk", "hmc"]
Start = Literal["matched", "wide"]


@dataclass(frozen=True)
class EstimatorPlan:
    """Fixed averaging settings, run sizes and a separate coverage seed."""

    settings: Settings
    runs: int
    replicates: int
    seed: int


@dataclass(frozen=True)
class ReferencePlan:
    """Settings for an independent uncoupled batch-means reference."""

    batches: int
    batch_size: int
    seed: int


def compare_case(
    case: targets.BenchmarkCase,
    *,
    methods: list[Method],
    start: Start,
    pairs: int,
    budget: int,
    tuning_seed: int,
    seed: int,
    additional_meeting_seeds: tuple[int, ...] = (),
    estimation: EstimatorPlan | None = None,
    reference: ReferencePlan | None = None,
) -> dict[str, object]:
    """Return raw pair records and separately counted tuning for each method."""
    meeting_seeds = (seed, *additional_meeting_seeds)
    if tuning_seed in meeting_seeds or len(set(meeting_seeds)) != len(meeting_seeds):
        raise ValueError(
            "tuning and evaluation seeds must differ and meeting seeds must be unique"
        )
    if estimation is not None and estimation.seed in (*meeting_seeds, tuning_seed):
        raise ValueError("estimator, meeting and tuning seeds must differ")
    if reference is not None:
        if estimation is None or case.draw_exact is None:
            raise ValueError("batch-means comparison requires estimator settings and exact draws")
        if reference.seed in (*meeting_seeds, tuning_seed, estimation.seed):
            raise ValueError("the uncoupled reference seed must be distinct")
    initial, walk = research.random_walk_setup(case, start, "oracle")
    comparisons = {}
    for method in methods:
        if method not in ("random_walk", "hmc"):
            raise ValueError(f"unknown benchmark method: {method}")
        kernel: Kernel = walk
        density_setup = grad_setup = 0
        messages = []
        if method == "hmc":
            sampler = CoupledSampler(
                case.log_prob,
                initial,
                grad=case.grad,
                vectorize=True,
                kernel=HMC(),
                rng=tuning_seed,
                check_grad=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", TelescopicWarning)
                sampler.tune()
            for warning in caught:
                if issubclass(warning.category, TelescopicWarning):
                    messages.append(str(warning.message))
                else:
                    warnings.warn_explicit(
                        warning.message, warning.category, warning.filename, warning.lineno
                    )
            kernel = sampler.kernel
            density_setup = sampler.target.n_log_prob
            grad_setup = sampler.target.n_grad
        records = []
        per_seed = []
        for meeting_seed in meeting_seeds:
            batch = research.meeting_pilot(
                case, initial, kernel, pairs=pairs, budget=budget, seed=meeting_seed
            )
            per_seed.append({"seed": meeting_seed, **research.summarize_pilot(batch)})
            records.extend(batch)
        summary = research.summarize_pilot(records)
        density_count = sum(row.evaluations - row.gradient_evaluations for row in records)
        gradient_count = sum(row.gradient_evaluations for row in records)
        comparison: dict[str, object] = {
            "kernel": repr(kernel),
            "setup_density_evaluations": density_setup,
            "setup_gradient_evaluations": grad_setup,
            "setup_warnings": messages,
            "summary": summary,
            "meeting_summary_by_seed": per_seed,
            "records": [asdict(row) for row in records],
        }
        if estimation is not None:
            runs = estimator.run_frozen(
                case,
                estimation.settings,
                initial,
                kernel,
                runs=estimation.runs,
                replicates=estimation.replicates,
                seed=estimation.seed,
            )
            variances: dict[str, float] = {}
            reference_cost = 0
            if reference is not None:
                variances, reference_cost = estimator.batch_means_reference(
                    case,
                    kernel,
                    batches=reference.batches,
                    batch_size=reference.batch_size,
                    seed=reference.seed,
                )
            comparison["estimation"] = {
                "reference_asymptotic_variance": variances,
                "reference_evaluations": reference_cost,
                "summary": {
                    item.name: estimator.summarize(
                        runs,
                        item,
                        setup_evaluations=density_setup + grad_setup,
                        asymptotic_variance=variances.get(item.name),
                    )
                    for item in case.estimands
                },
                "mode_probabilities": {
                    item.name: estimator.summarize(
                        runs, item, setup_evaluations=density_setup + grad_setup
                    )
                    for item in case.mode_probabilities
                },
                "records": [asdict(row) for row in runs],
            }
            density_count += sum(row.evaluations - row.gradient_evaluations for row in runs)
            gradient_count += sum(row.gradient_evaluations for row in runs)
        comparison.update(
            sampling_density_evaluations=density_count,
            sampling_gradient_evaluations=gradient_count,
            total_evaluations=density_setup + grad_setup + density_count + gradient_count,
        )
        comparisons[method] = comparison
    return {
        "target": case.name,
        "dimension": case.dimension,
        "start": start,
        "matched_covariance_source": case.covariance_source,
        "methods": comparisons,
    }


def main() -> None:
    """Compare selected methods across one or more named benchmark cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", metavar="TARGET:DIMENSION")
    parser.add_argument("--method", action="append", choices=("random_walk", "hmc"))
    parser.add_argument("--start", choices=("matched", "wide"), default="matched")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--tuning-seed", type=int, required=True)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--replicates", type=int, default=100)
    for name in ("k", "m", "lag", "max-iter", "estimator-seed"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--reference-batches", type=int, default=100)
    parser.add_argument("--reference-batch-size", type=int, default=1000)
    args = parser.parse_args()
    reference = (
        ReferencePlan(args.reference_batches, args.reference_batch_size, args.reference_seed)
        if args.reference_seed is not None
        else None
    )
    settings = (args.k, args.m, args.lag, args.max_iter, args.estimator_seed)
    estimation = None
    if args.runs is not None:
        if any(setting is None for setting in settings):
            parser.error("--runs requires --k, --m, --lag, --max-iter, and --estimator-seed")
        estimation = EstimatorPlan(
            Settings(args.k, args.m, args.lag, 1, args.max_iter, True),
            args.runs,
            args.replicates,
            args.estimator_seed,
        )
    elif any(setting is not None for setting in settings):
        parser.error("estimator settings require --runs")
    research.require_source_checkout()
    methods = list(dict.fromkeys(args.method or ("random_walk", "hmc")))
    cases = []
    with research.threadpoolctl.threadpool_limits(limits=1):
        provenance = setup()
        for item in args.cases:
            name, _, raw_dimension = item.partition(":")
            if name not in research.TARGETS or not raw_dimension.isdecimal():
                parser.error(f"invalid case {item!r}; expected a known target and its dimension")
            case = research.TARGETS[name](int(raw_dimension))
            result = compare_case(
                case,
                methods=methods,
                start=args.start,
                pairs=args.pairs,
                budget=args.budget,
                tuning_seed=args.tuning_seed,
                seed=args.seed[0],
                additional_meeting_seeds=tuple(args.seed[1:]),
                estimation=estimation,
                reference=reference,
            )
            result["case"] = item
            cases.append(result)
    print(
        json.dumps(
            {
                **provenance,
                **vars(args),
                "cost_unit": "pointwise_density_plus_gradient",
                "comparisons": cases,
            },
            allow_nan=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
