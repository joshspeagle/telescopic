"""Seconds of a run on 4 started workers by the size of the pool's chunks (M11, DECISIONS #52).

_parallel.CHUNK caps a chunk's replicates while four chunks per worker allow: 64, as at
M11's lanes change, against 256, the engine's lanes. 1,000 replicates of three
vectorised targets and 400 of a log-density of 0.2 ms evaluated point by point; the best
of 3 runs of each size, alternated. Run from the repository on an idle machine.
"""

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from telescopic import _parallel, engine, examples
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from run import setup

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)
LINE_COV = np.linalg.inv(
    [[np.sum(x**2 / yerr**2), np.sum(x / yerr**2)], [np.sum(x / yerr**2), np.sum(1 / yerr**2)]]
)


def line(theta):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)


def slow(theta):
    # About 0.2 ms per point, point by point.
    end = time.perf_counter() + 2e-4
    while time.perf_counter() < end:
        pass
    return -0.5 * float(theta @ theta)


def never(status):
    return False


def main():
    gaussian, banana = examples.gaussian(10), examples.banana()
    cases = {
        "gaussian(10), k = 222": (
            functools.partial(
                engine.run,
                Target(gaussian.log_prob, vectorize=True),
                RandomWalk(gaussian.cov),
                gaussian.initial,
                rng=2,
                k=222,
                m=2220,
                lag=222,
            ),
            1000,
        ),
        "line fit, k = 87": (
            functools.partial(
                engine.run,
                Target(line, vectorize=True),
                RandomWalk(LINE_COV),
                Normal([-0.9, 4.3], [1e-2, 0.25]),
                rng=2,
                k=87,
                m=870,
                lag=87,
            ),
            1000,
        ),
        "banana, pilot settings": (
            functools.partial(
                engine.run,
                Target(banana.log_prob, vectorize=True),
                RandomWalk(banana.cov),
                banana.initial,
                rng=2,
                k=0,
                m=0,
                lag=1,
                store=False,
            ),
            1000,
        ),
        "0.2 ms point by point, k = 10": (
            functools.partial(
                engine.run,
                Target(slow),
                RandomWalk(np.eye(2)),
                Normal([0.0, 0.0], [4.0, 4.0]),
                rng=2,
                k=10,
                m=50,
                lag=10,
            ),
            400,
        ),
    }
    with threadpool_limits(1):
        print(json.dumps(setup()), flush=True)
        with _parallel.opened(4) as workers:
            first = next(iter(cases.values()))[0]
            _parallel.execute(first, np.arange(8), workers, never)  # starts the workers
            for name, (job, n) in cases.items():
                best = {}
                for _ in range(3):
                    for chunk in (64, 256):
                        _parallel.CHUNK = chunk
                        start = time.perf_counter()
                        _parallel.execute(job, np.arange(n), workers, never)
                        best[chunk] = min(best.get(chunk, np.inf), time.perf_counter() - start)
                print(
                    f"{name}, {n} replicates: chunks of at most 64 {best[64]:.3f} s, "
                    f"of at most 256 {best[256]:.3f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
