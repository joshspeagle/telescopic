"""Exploratory: evaluations and peak memory of the prototype at d = 1,000 and 10,000 (seed 219500).

Design: N(0, I_d) from N(0, 4 I) drawn by a callable, vectorized; tracemalloc peak during prototype.N,
with P = 8; evaluations from the check's own Target. Also a float32 gradient, which adds P of each.
Result: see cost.out.
"""

import time
import tracemalloc

import numpy as np
import prototype


def log_prob(x):
    return -0.5 * np.sum(x * x, axis=-1)


for d in (1000, 10_000):
    for name, g in (
        ("float64", lambda x: -x),
        ("float32 grad", lambda x: (-x).astype(np.float32)),
    ):

        def init(generator, d=d):
            return 2.0 * generator.standard_normal(d)

        tracemalloc.start()
        t = time.perf_counter()
        v = prototype.N(log_prob, g, init, 219500)
        dt = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(
            f"d = {d}, {name}: {v.n_log_prob} log_prob + {v.n_grad} grad evaluations, peak {peak / 1e6:.2f} MB "
            f"({peak / (8 * d):.0f} vectors of d float64), {1e3 * dt:.1f} ms, warned {v.warned}"
        )
