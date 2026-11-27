"""Run selected #90 meeting cells with the registered case settings."""

import argparse
import json
from dataclasses import dataclass

from benchmarks import compare, research
from benchmarks.run import setup

TUNING_SEED = 20261101
MEETING_SEEDS = tuple(range(20261201, 20261206))
PAIRS_PER_SEED = 100


@dataclass(frozen=True)
class Case:
    """A registered target ID, variant, dimension, and pointwise evaluation budget."""

    target_id: int
    name: str
    dimension: int
    budget: int

    @property
    def label(self) -> str:
        """Return the CLI case selector."""
        return f"{self.name}:{self.dimension}"


def _cases() -> dict[str, Case]:
    families = (
        (1, "rotated_gaussian", (100, 1000), 1_000_000),
        (2, "banana", (2, 20, 100), 200_000),
        (3, "funnel", (10, 100), 200_000),
        (4, "mixture_equal_6", (2, 10, 32), 100_000),
        (4, "mixture_equal_12", (2, 10, 32), 100_000),
        (4, "mixture", (2, 10, 32), 100_000),
        (5, "four_mode", (10,), 100_000),
        (6, "normal_means", (10, 102), 200_000),
        (7, "logistic", (6, 300), 200_000),
        (9, "supernova", (4,), 100_000),
        (10, "parallax", (51, 501), 200_000),
    )
    cases = {}
    for target_id, name, dimensions, budget in families:
        for dimension in dimensions:
            case = Case(target_id, name, dimension, budget)
            cases[case.label] = case
    return cases


def main() -> None:
    """List the registered cells or run the selected ones."""
    cases = _cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=cases)
    parser.add_argument("--start", choices=("matched", "wide"), default="matched")
    parser.add_argument("--method", action="append", choices=("random_walk", "hmc"))
    args = parser.parse_args()
    if args.case is None:
        print(
            json.dumps(
                {
                    name: {"target_id": case.target_id, "budget": case.budget}
                    for name, case in cases.items()
                },
                indent=2,
            )
        )
        return
    research.require_source_checkout()
    results = []
    methods = list(dict.fromkeys(args.method or ("random_walk", "hmc")))
    with research.threadpoolctl.threadpool_limits(limits=1):
        provenance = setup()
        for label in dict.fromkeys(args.case):
            cell = cases[label]
            case = research.TARGETS[cell.name](cell.dimension)
            result = compare.compare_case(
                case,
                methods=methods,
                start=args.start,
                pairs=PAIRS_PER_SEED,
                budget=cell.budget,
                tuning_seed=TUNING_SEED,
                seed=MEETING_SEEDS[0],
                additional_meeting_seeds=MEETING_SEEDS[1:],
            )
            result.update(case=label, target_id=cell.target_id, budget=cell.budget)
            results.append(result)
    print(
        json.dumps(
            {
                **provenance,
                "tuning_seed": TUNING_SEED,
                "meeting_seeds": MEETING_SEEDS,
                "pairs_per_seed": PAIRS_PER_SEED,
                "cost_unit": "pointwise_density_plus_gradient",
                "comparisons": results,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
