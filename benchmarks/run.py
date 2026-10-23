"""Timings and efficiency of telescopic, as JSON (test plan section 2.14); run by hand.

Run from the repository: python benchmarks/run.py. It prints the JSON and writes it to
benchmarks/results/<commit>.json, with the commit, machine, thread settings and versions;
figures from different setups are not comparable (CLAUDE.md, Measurements).
"""

import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import scipy

import telescopic
from telescopic import examples
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

ROOT = Path(__file__).resolve().parent.parent


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def setup():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {
        "commit": commit,
        "machine": f"{cpu_model()}, {os.cpu_count()} CPUs, {platform.system()} {platform.release()}",
        "threads": {name: os.environ.get(name) for name in threads},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "telescopic": telescopic.__version__,
    }


def best(function, repeats=3):
    # The best of a few repeats, in seconds, and the last result.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return min(times), result


def gaussian_run(target, lanes, n=1024, **options):
    example = examples.gaussian(10)
    kernel = RandomWalk(example.cov, scale=2.38 / np.sqrt(10))
    settings = {"k": 10, "m": 100, "lag": 10} | options
    return run(target, kernel, example.initial, range(n), rng=1, lanes=lanes, **settings)


def lockstep():
    # Coupled random walk on examples.gaussian(10), vectorised: microseconds per kernel
    # application (a coupled step counts two) against the lanes advanced together.
    target = Target(examples.gaussian(10).log_prob, vectorize=True)
    out = {}
    for lanes in (1, 8, 64, 256, 1024):
        seconds, replicates = best(lambda lanes=lanes: gaussian_run(target, lanes))
        out[lanes] = round(1e6 * seconds / replicates.cost.sum(), 3)
    return out


def point_by_point():
    # The same run at 64 lanes with a log-density evaluated point by point.
    out = {}
    for vectorize in (True, False):
        target = Target(examples.gaussian(10).log_prob, vectorize=vectorize)
        seconds, replicates = best(lambda target=target: gaussian_run(target, 64, n=256))
        out["vectorised" if vectorize else "point by point"] = round(
            1e6 * seconds / replicates.cost.sum(), 3
        )
    return out


def slow_log_prob(x):
    # A log-density that costs 1 ms of CPU per point.
    end = time.perf_counter() + 1e-3
    while time.perf_counter() < end:
        pass
    return -0.5 * float(x @ x)


def pool_speedup():
    # 256 replicates of a fixed random walk on N(0, I_2) with a 1 ms log-density, in this
    # process and on 4 spawned workers, whose start is included.
    initial = Normal(np.zeros(2), np.eye(2))
    out = {}
    for pool in (None, 4):

        def call(pool=pool):
            return telescopic.sample(
                slow_log_prob,
                initial,
                256,
                kernel=RandomWalk(np.eye(2), scale=1.7),
                k=5,
                m=20,
                lag=5,
                pool=pool,
                rng=2,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seconds, _ = best(call, repeats=1)
        out["in process" if pool is None else f"{pool} workers"] = round(seconds, 2)
    out["speed-up"] = round(out["in process"] / out["4 workers"], 2)
    return out


def pipeline():
    # sample() on examples.gaussian(10), 100 replicates: the whole call, which tunes the
    # kernel and runs the meeting-time pilot, against the engine alone with the same
    # kernel and settings.
    example = examples.gaussian(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole, result = best(
            lambda: telescopic.sample(
                example.log_prob, example.initial, 100, vectorize=True, rng=3
            ),
            repeats=1,
        )
    target = Target(example.log_prob, vectorize=True)
    alone, _ = best(
        lambda: run(
            target,
            result.kernel,
            example.initial,
            range(100),
            k=result.k,
            m=result.m,
            lag=result.lag,
            rng=3,
        ),
        repeats=1,
    )
    return {"sample": round(whole, 2), "engine alone": round(alone, 2), "k": result.k}


def efficiency():
    # Meeting times and inefficiency (mean cost over m - k + 1) of the automatic kernels.
    out = {}
    for example in (examples.gaussian(10), examples.banana()):
        for kernel in ("rwmh", "mala", "hmc"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seconds, result = best(
                    lambda example=example, kernel=kernel: telescopic.sample(
                        example.log_prob,
                        example.initial,
                        200,
                        grad=example.grad,
                        vectorize=True,
                        kernel=kernel,
                        rng=4,
                    ),
                    repeats=1,
                )
            diagnostics = result.diagnostics()
            out[f"{example.name}, {kernel}"] = {
                "k": result.k,
                "meeting time 50%": diagnostics["meeting_time_50"],
                "meeting time 99%": diagnostics["meeting_time_99"],
                "inefficiency": round(diagnostics["inefficiency"], 3),
                "seconds": round(seconds, 2),
            }
    return out


if __name__ == "__main__":
    report = setup()
    for name, case in (
        ("lockstep, microseconds per kernel application, by lanes", lockstep),
        ("point by point, microseconds per kernel application at 64 lanes", point_by_point),
        ("pool, seconds", pool_speedup),
        ("pipeline, seconds", pipeline),
        ("efficiency", efficiency),
    ):
        report[name] = case()
    text = json.dumps(report, indent=2)
    print(text)
    (ROOT / "benchmarks" / "results" / f"{report['commit'][:7]}.json").write_text(text + "\n")
