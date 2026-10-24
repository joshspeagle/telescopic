"""Timings and efficiency of telescopic on the setups of spec section 10, as JSON; run by hand.

Run from the repository on an otherwise idle machine: python benchmarks/run.py. It writes
benchmarks/results/<commit>.json after each section, with the commit, machine, the CPUs
this process may use, BLAS libraries and versions; figures from different setups are not
comparable (CLAUDE.md, Measurements). Every case runs with BLAS on one thread, as "blas"
records. A pair-step is an iteration of one replicate, max(tau, m) per replicate.
"""

import json
import multiprocessing
import os
import platform
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import scipy
import threadpoolctl

import telescopic
from telescopic import examples
from telescopic._random import MAIN, Streams, replicate_generator, root_seed
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

ROOT = Path(__file__).resolve().parent.parent
# Coupled random-walk Metropolis on N(0, I_10) from N(3, I), as in the M3 review.
STANDARD = Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True)
START = Normal(np.full(10, 3.0), np.eye(10))
WALK = RandomWalk(np.eye(10))
MAIN_RUN = {"k": 141, "m": 1410, "lag": 141}
PILOT = {"k": 0, "m": 0, "lag": 1, "store": False}


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def setup():
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "machine": f"{cpu_model()}, {os.cpu_count()} CPUs, {platform.system()} {platform.release()}",
        "available CPUs": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "blas": [
            {key: library[key] for key in ("internal_api", "version", "num_threads")}
            for library in threadpoolctl.threadpool_info()
        ],
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


def timed_run(target, lanes, n, settings):
    # Seconds of the best of 3 runs, the last run's replicates, and its pair-steps.
    seconds, replicates = best(
        partial(run, target, WALK, START, range(n), rng=1, lanes=lanes, **settings)
    )
    return seconds, replicates, np.maximum(replicates.meeting_time, settings["m"]).sum()


def lockstep():
    # Microseconds per pair-step of the vectorised target, by lanes, in the main run and in
    # the meeting-time pilot.
    out = {}
    for name, settings, lane_counts, minimum in (
        ("main run", MAIN_RUN, (1, 8, 64, 256, 1024), 16),
        ("pilot", PILOT, (1, 256), 64),
    ):
        out[name] = {}
        for lanes in lane_counts:
            seconds, _, steps = timed_run(STANDARD, lanes, max(minimum, 2 * lanes), settings)
            out[name][lanes] = round(1e6 * seconds / steps, 2)
    return out


def point_by_point():
    # One lane and a log-density evaluated point by point: microseconds per pair-step beyond
    # the log-density's own time.
    def log_prob(x):
        return -0.5 * float(x @ x)

    point, calls = np.full(10, 3.0), 100_000
    start = time.perf_counter()
    for _ in range(calls):
        log_prob(point)
    per_call = (time.perf_counter() - start) / calls
    out = {"log-density call": round(1e6 * per_call, 2)}
    for name, settings, n in (("main run", MAIN_RUN, 16), ("pilot", PILOT, 64)):
        seconds, replicates, steps = timed_run(Target(log_prob), 1, n, settings)
        out[name] = round(1e6 * (seconds - replicates.n_log_prob * per_call) / steps, 1)
    return out


def streams():
    # Per-replicate random streams at 256 lanes: microseconds per lane-step for the draws of
    # a coupled random-walk step in 10 dimensions (10 normals and two log-uniforms), and
    # per replicate to create and assign its streams.
    root, lanes, steps = root_seed(1), 256, 2000
    start = time.perf_counter()
    lane_streams = Streams(lanes)
    for lane in range(lanes):
        lane_streams.assign(lane, replicate_generator(root, MAIN, lane))
    set_up = (time.perf_counter() - start) / lanes
    draws = lane_streams.view(np.arange(lanes))
    start = time.perf_counter()
    for _ in range(steps):
        draws.normal(10)
        draws.log_uniform()
        draws.log_uniform()
    per_lane_step = (time.perf_counter() - start) / (steps * lanes)
    return {
        "draws per lane-step": round(1e6 * per_lane_step, 3),
        "set-up per replicate": round(1e6 * set_up, 1),
    }


def slow_log_prob(x):
    # A log-density that costs 1 ms of CPU per point.
    end = time.perf_counter() + 1e-3
    while time.perf_counter() < end:
        pass
    return -0.5 * float(x @ x)


def pool_speedup():
    # 256 replicates of a fixed random walk on N(0, I_2) with a 1 ms log-density, in this
    # process, on 4 workers that the call spawns, and on an executor's 4 workers started by
    # an earlier run, whose time is then the throughput.
    call = partial(
        telescopic.sample,
        slow_log_prob,
        Normal(np.zeros(2), np.eye(2)),
        256,
        kernel=RandomWalk(np.eye(2), scale=1.7),
        k=5,
        m=20,
        lag=5,
        rng=2,
    )
    out = {
        "in process": round(best(call, 1)[0], 2),
        "4 workers": round(best(partial(call, pool=4), 1)[0], 2),
    }
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as executor:
        call(pool=executor)
        out["4 started workers"] = round(best(partial(call, pool=executor), 1)[0], 2)
    out["speed-up"] = round(out["in process"] / out["4 workers"], 2)
    out["speed-up, started"] = round(out["in process"] / out["4 started workers"], 2)
    return out


def pipeline():
    # sample() at the default R = 100 on examples.gaussian(d): the whole call, its main run,
    # and the overhead of the tuning and meeting-time pilots relative to the main run.
    # Thinning by d / 10 keeps the stored atoms under about 0.5 GB.
    out = {}
    for d in (10, 50, 100):
        example = examples.gaussian(d)
        call = partial(
            telescopic.sample, example.log_prob, example.initial, vectorize=True, thin=d // 10
        )
        whole, result = best(partial(call, rng=3), 1)
        out[f"d = {d}"] = {
            "whole": round(whole, 2),
            "main run": round(result.wall_time, 2),
            "overhead": round(whole / result.wall_time - 1, 2),
            "k": result.k,
        }
    return out


def efficiency():
    # sample() with each automatic kernel on two examples, R = 200: k, meeting-time
    # quantiles and relative cost (mean cost over m - k + 1), or why k could not be chosen.
    out = {}
    for example in (examples.gaussian(10), examples.banana()):
        for kernel in ("rwmh", "mala", "hmc"):
            call = partial(
                telescopic.sample,
                example.log_prob,
                example.initial,
                200,
                grad=example.grad,
                vectorize=True,
                kernel=kernel,
                rng=4,
            )
            start = time.perf_counter()
            try:
                result = call()
            except ValueError as error:  # too many pilot pairs unmet at their cap
                out[f"{example.name}, {kernel}"] = {
                    "error": str(error),
                    "seconds": round(time.perf_counter() - start, 2),
                }
                continue
            diagnostics = result.diagnostics()
            out[f"{example.name}, {kernel}"] = {
                "k": result.k,
                "meeting time 50%": diagnostics["meeting_time_50"],
                "meeting time 99%": diagnostics["meeting_time_99"],
                "relative cost": round(diagnostics["relative_cost"], 3),
                "seconds": round(time.perf_counter() - start, 2),
            }
    return out


if __name__ == "__main__":
    warnings.simplefilter("ignore", telescopic.TelescopicWarning)
    with threadpoolctl.threadpool_limits(1):
        report = setup()
        path = ROOT / "benchmarks" / "results" / f"{report['commit']}.json"
        path.parent.mkdir(exist_ok=True)
        for name, case in (
            ("lockstep, microseconds per pair-step by lanes", lockstep),
            ("point by point, microseconds per pair-step beyond the log-density", point_by_point),
            ("random streams, microseconds", streams),
            ("pool, seconds", pool_speedup),
            ("pipeline, seconds", pipeline),
            ("efficiency", efficiency),
        ):
            report[name] = case()
            path.write_text(json.dumps(report, indent=2) + "\n")
    print(path.read_text())
