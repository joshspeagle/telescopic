"""Run an evaluation-capped #90 smoke pilot with a frozen random-walk kernel.

Run from the checkout root with PYTHONPATH=src python -m benchmarks.research. A null meeting
cost means the pair was censored. Cost is measured in pointwise log-density
evaluations, including initial states. A pair is censored when its next
log-density request would exceed its allowance.
"""

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import telescopic
from benchmarks import targets
from benchmarks.run import setup
from telescopic._types import FloatArray, integer
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

threadpoolctl = importlib.import_module("threadpoolctl")

TARGETS = {
    "rotated_gaussian": targets.rotated_gaussian,
    "banana": targets.banana,
    "funnel": targets.funnel,
    "mixture": targets.mixture,
}


@dataclass(frozen=True)
class PairRecord:
    """Actual log-density evaluations and the cost at meeting, if observed."""

    evaluations: int
    meeting_cost: int | None
    meeting_iteration: int | None


class _BudgetExhausted(Exception):
    pass


class _BudgetTarget(Target):
    def __init__(self, case: targets.BenchmarkCase, budget: int) -> None:
        super().__init__(case.log_prob, vectorize=True)
        self.budget = budget

    def log_prob(self, x: FloatArray) -> FloatArray:
        if self.n_log_prob + len(x) > self.budget:
            raise _BudgetExhausted
        return super().log_prob(x)


def random_walk_setup(
    case: targets.BenchmarkCase,
    start: Literal["matched", "wide"],
    proposal: Literal["oracle", "identity"],
) -> tuple[Normal, RandomWalk]:
    """Build a matched or wide start and a frozen random-walk kernel."""
    if start not in ("matched", "wide"):
        raise ValueError("start must be 'matched' or 'wide'")
    if proposal not in ("oracle", "identity"):
        raise ValueError("proposal must be 'oracle' or 'identity'")
    initial = case.matched if start == "matched" else case.wide
    covariance = case.matched.cov if proposal == "oracle" else np.eye(case.dimension)
    return initial, RandomWalk(covariance)


def random_walk_pilot(
    case: targets.BenchmarkCase,
    *,
    start: Literal["matched", "wide"],
    proposal: Literal["oracle", "identity"],
    pairs: int,
    budget: int,
    seed: int,
) -> list[PairRecord]:
    """Run lag-one random-walk pairs with a pointwise evaluation allowance.

    Each pair stops at meeting or before its first unaffordable density
    request. With an oracle proposal the covariance is known analytically.
    """
    pairs = integer("pairs", pairs, minimum=1)
    budget = integer("budget", budget, minimum=5)
    initial, kernel = random_walk_setup(case, start, proposal)
    records = []
    for index in range(pairs):
        target = _BudgetTarget(case, budget)
        try:
            result = run(
                target,
                kernel,
                initial,
                [index],
                k=0,
                m=0,
                lag=1,
                rng=seed,
                store=False,
                lanes=1,
            )
        except _BudgetExhausted:
            records.append(PairRecord(target.n_log_prob, None, None))
        else:
            cost = result.n_log_prob
            records.append(PairRecord(cost, cost, int(result.meeting_time[0])))
    return records


def summarize_pilot(records: list[PairRecord]) -> dict[str, float | int | None]:
    """Summarize meeting costs with censored pairs ranked at positive infinity."""
    if not records:
        raise ValueError("records must contain at least one pair")
    costs = np.array(
        [np.inf if record.meeting_cost is None else record.meeting_cost for record in records]
    )
    quantiles = np.quantile(costs, [0.5, 0.9, 0.99, 1.0], method="inverted_cdf")
    return {
        "pairs": len(records),
        "censored_fraction": float(np.isinf(costs).mean()),
        "evaluations": sum(record.evaluations for record in records),
        **{
            key: int(value) if np.isfinite(value) else None
            for key, value in zip(("q50", "q90", "q99", "max"), quantiles, strict=True)
        },
    }


def require_source_checkout() -> None:
    """Ensure the benchmark metadata identifies the sampling code being run."""
    source = Path(__file__).resolve().parents[1] / "src" / "telescopic" / "__init__.py"
    if Path(telescopic.__file__).resolve() != source:
        raise RuntimeError("Run with PYTHONPATH=src so the benchmark uses this checkout")


def main() -> None:
    """Print a JSON record for one seed and benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("dimension", type=int)
    parser.add_argument("--start", choices=("matched", "wide"), default="matched")
    parser.add_argument("--proposal", choices=("oracle", "identity"), default="oracle")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--budget", type=int, required=True, help="evaluation ceiling per pair")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require_source_checkout()
    with threadpoolctl.threadpool_limits(limits=1):
        case = TARGETS[args.target](args.dimension)
        records = random_walk_pilot(
            case,
            start=args.start,
            proposal=args.proposal,
            pairs=args.pairs,
            budget=args.budget,
            seed=args.seed,
        )
        report = {
            **setup(),
            **vars(args),
            "method": "random_walk",
            "setup_evaluations": 0,
            "summary": summarize_pilot(records),
            "records": [asdict(record) for record in records],
        }
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
