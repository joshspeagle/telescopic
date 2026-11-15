"""Times of the levers of the guide Make a run faster, as JSON; run by hand.

Run from the repository on an otherwise idle machine: python benchmarks/run_faster.py. It
writes benchmarks/results/run-faster.json, which docs/guide/faster.md reads, with the
commit, machine, the CPUs this process may use, BLAS libraries and versions (setup() of
run.py). BLAS runs on one thread in this process, as it does in telescopic's workers.

Every case runs telescopic.sample on examples.gaussian(10) with a random walk whose
covariance is the target's and a fixed k, so that no pilot runs, except the last case,
which compares that run with the automatic pipeline at the default settings. Times are
the best of REPEATS runs in this process and of one run on a pool, whose workers start
within the call.
"""

import argparse
import json
import os
import warnings
from functools import partial
from pathlib import Path

import threadpoolctl
from run import ROOT, best, setup

import telescopic
from telescopic import examples
from telescopic.kernels import RandomWalk

GAUSSIAN = examples.gaussian(10)
K = 200
REPEATS = 2


def timed(n, **options):
    # Seconds of the best run, and the run's result.
    call = partial(telescopic.sample, GAUSSIAN.log_prob, GAUSSIAN.initial, n, rng=1, **options)
    return best(call, 1 if options.get("pool") else REPEATS)


def log_density_and_pool(n):
    # Seconds of R = n replicates with the log-density evaluated point by point and
    # vectorized, in this process and on integer pools of up to the CPUs available, and the
    # kernel applications of a run, the same in every case.
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    )
    out = {}
    for pool in (None, *(w for w in (2, 4, 8) if w <= available)):
        row = "in this process" if pool is None else f"{pool} workers"
        out[row] = {}
        for name, vectorize in (("point by point", False), ("vectorized", True)):
            seconds, result = timed(
                n, kernel=RandomWalk(GAUSSIAN.cov), k=K, vectorize=vectorize, pool=pool
            )
            out[row][name] = round(seconds, 2)
    return out, int(result.costs.sum())


def pilots(n):
    # Seconds of the whole call with the automatic pipeline, tuning and the meeting-time
    # pilot included, and of the same number of replicates with a given kernel and k.
    automatic, result = timed(n, vectorize=True)
    given, _ = timed(n, kernel=RandomWalk(GAUSSIAN.cov), k=result.k, vectorize=True)
    return {
        "automatic": round(automatic, 2),
        "of which the replicates": round(result.wall_time, 2),
        "kernel and k given": round(given, 2),
        "automatic k": result.k,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=1024, help="R of each case")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks" / "results" / "run-faster.json"
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore", telescopic.TelescopicWarning)
    with threadpoolctl.threadpool_limits(1):
        report = setup()
        report["case"] = (
            f"telescopic.sample on examples.gaussian(10), R = {args.replicates}, rng=1; "
            f"RandomWalk(cov) with the target's cov and k = {K} unless automatic"
        )
        seconds, applications = log_density_and_pool(args.replicates)
        report["seconds by pool and log-density"] = seconds
        report["kernel applications of each run"] = applications
        report["seconds with and without the pilots"] = pilots(args.replicates)
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.read_text())
