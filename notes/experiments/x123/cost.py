"""Criterion 5 of experiment x123: the evaluations and the peak memory of the check.

Question: at d = 1,000 and 10,000, does the check make at most 60 evaluations of
log_prob and 25 of grad, and hold at most 20 vectors of d float64 at its peak?

Design: N(0, I_d) from N(0, 4 I) drawn by a callable, vectorized; three precisions: a
float64 target, a float32 gradient of a float64 log-density, and a target computed
wholly in float32. Seeds 390000 to 390004 (PREREGISTRATION.md, section 6). The peak is
tracemalloc's over the call of gradcheck.check, which traces every allocation of Python
and NumPy during it, including those of the user's functions and of the draws; it is
given in MB and in vectors of d float64 (8 d bytes). The time is the median over the
seeds of the call without tracemalloc. (N) and (H) are the same computation (they differ
only in the comparison of numbers that the check returns), so one call measures both.

Run: python notes/experiments/x123/cost.py [--smoke]

Result: see cost.out.
"""

import time
import tracemalloc
import warnings

import families as F
import gradcheck as G
import numpy as np
from runner import header, smoke

SEEDS = F.SMOKE if smoke() else range(390_000, 390_005)


def targets(d):
    lp = lambda x: -0.5 * np.sum(x * x, axis=-1)
    lp32 = lambda x: -np.float32(0.5) * np.sum(x.astype(np.float32) ** 2, axis=-1)
    return {
        "float64": (lp, lambda x: -x),
        "float32 grad": (lp, lambda x: (-x).astype(np.float32)),
        "float32 throughout": (lp32, lambda x: -x.astype(np.float32)),
    }


def callable_start(d):
    # N(0, 4 I_d), drawn by a callable as a user would pass it
    return lambda generator: 2.0 * generator.standard_normal(d)


def main():
    header()
    print(f"seeds {list(SEEDS)}; bounds: 60 log_prob, 25 grad, 20 vectors of d float64")
    for d in (1000, 10_000):
        init = callable_start(d)
        for name, (lp, g) in targets(d).items():
            peaks, times, counts = [], [], set()
            for seed in SEEDS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tracemalloc.start()
                    v = G.check(lp, g, init, seed, vectorize=True)
                    peaks.append(tracemalloc.get_traced_memory()[1])
                    tracemalloc.stop()
                    start = time.perf_counter()
                    G.check(lp, g, init, seed, vectorize=True)
                    times.append(time.perf_counter() - start)
                counts.add((v.n_log_prob, v.n_grad, v.dtype_f, v.dtype_g))
            peak = max(peaks)
            within = all(c[0] <= 60 and c[1] <= 25 for c in counts) and peak <= 20 * 8 * d
            print(
                f"d = {d:>6}, {name:<18}: evaluations (log_prob, grad, dtypes) {sorted(counts)}; peak {peak / 1e6:.3f} MB "
                f"= {peak / (8 * d):.1f} vectors of d; median time {1e3 * np.median(times):.1f} ms; within the bounds: {within}",
                flush=True,
            )


if __name__ == "__main__":
    main()
