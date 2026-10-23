"""Seconds of a main run by lanes, for 100 and 1,000 replicates (M11).

sample()'s tuning and pilot fix each case's settings; the main run is then timed alone,
the best of 3, with BLAS on one thread. Run from the repository on an idle machine.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import telescopic
from telescopic import engine, examples

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from run import setup

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def point(theta, x, y, yerr):
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)


def batch(theta, x, y, yerr):
    slope, intercept = theta[..., :1], theta[..., 1:]
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2, axis=-1)


line = telescopic.initial.Normal([-0.9, 4.3], [1e-2, 0.25])
gaussian = examples.gaussian(10)
cases = {
    "line fit, point by point": (point, line, (x, y, yerr), False),
    "line fit, vectorised": (batch, line, (x, y, yerr), True),
    "gaussian(10), vectorised": (gaussian.log_prob, gaussian.initial, (), True),
}
with threadpool_limits(1), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(json.dumps(setup()), flush=True)
    for name, (log_prob, initial, args, vectorize) in cases.items():
        sampler = telescopic.CoupledSampler(
            log_prob, initial, args=args, vectorize=vectorize, rng=1
        )
        sampler.run(1)  # tunes, runs the pilot and fixes the settings
        settings = sampler.settings._asdict()
        for n, lane_counts in ((100, (64, 100, 128, 256)), (1000, (64, 128, 256, 512, 1024))):
            times = []
            for lanes in lane_counts:
                best = np.inf
                for _ in range(3):
                    start = time.perf_counter()
                    engine.run(
                        sampler.target,
                        sampler.kernel,
                        sampler.initial,
                        range(n),
                        rng=2,
                        lanes=lanes,
                        **settings,
                    )
                    best = min(best, time.perf_counter() - start)
                times.append(f"{lanes} lanes {best:.3f} s")
            print(f"{name}, k = {settings['k']}, {n} replicates: " + "; ".join(times), flush=True)
