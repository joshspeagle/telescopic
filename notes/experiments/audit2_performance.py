"""Time and memory of the second audit's efficiency changes (DECISIONS #66).

Each figure is from one process, with BLAS limited to one thread:
- the default adaptive Metropolis pilot on `examples.gaussian(d)` at d = 50 and 100;
- `sample()` of the Gibbs tutorial: the pump model, 1,000 replicates, the maximal coupling;
- a run of 400 replicates of a 10-dimensional Gaussian: its peak memory, with its
  diagnostics, and the time and peak memory of a joint histogram of 3 coordinates of 40
  bins (tracemalloc).

Run: python notes/experiments/audit2_performance.py
"""

import os
import platform
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import scipy
import threadpoolctl

import telescopic
from telescopic import engine, examples, tuning
from telescopic.initial import Normal
from telescopic.kernels import Gibbs, RandomWalk
from telescopic.target import Target

warnings.simplefilter("ignore", telescopic.TelescopicWarning)
info = Path("/proc/cpuinfo").read_text().splitlines()
cpu = next((line.split(":")[1].strip() for line in info if "model name" in line), "")
print(f"{cpu}, {os.cpu_count()} CPUs; Python {platform.python_version()}")
print(f"NumPy {np.__version__}, SciPy {scipy.__version__}; BLAS limited to 1 thread")

with threadpoolctl.threadpool_limits(1):
    for d in (50, 100):
        gaussian = examples.gaussian(d)
        target = Target(gaussian.log_prob, vectorize=True)
        start = time.perf_counter()
        tuning.adaptive_metropolis(target, gaussian.initial, 1)
        print(f"pilot at d = {d}: {time.perf_counter() - start:.2f} s")

    pumps = examples.pumps()
    start = time.perf_counter()
    telescopic.sample(
        None, pumps.initial, 1000, kernel=Gibbs(pumps.conditional, pumps.blocks), rng=2
    )
    print(f"Gibbs tutorial run: {time.perf_counter() - start:.2f} s")

    d = 10
    target = Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True)
    tracemalloc.start()
    replicates = engine.run(
        target,
        RandomWalk(np.eye(d)),
        Normal(np.zeros(d), 4 * np.eye(d)),
        range(400),
        k=60,
        m=600,
        lag=60,
        rng=1,
    )
    result = telescopic.Result(replicates)
    result.diagnostics()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"run of 400 replicates, {len(replicates.atoms)} atoms: peak {peak / 1e6:.0f} MB")
    tracemalloc.start()
    start = time.perf_counter()
    result.histogram([0, 1, 2], bins=40)
    seconds = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"joint histogram of 3 coordinates, 40 bins: {seconds:.3f} s, peak {peak / 1e6:.0f} MB")
