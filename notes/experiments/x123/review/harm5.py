"""Review of x123: the harm, by the registered measure, of the wrong gradients of cases5.py.

Question: by how much does each wrong gradient that the check misses slow the meeting of
coupled MALA, against the correct gradient from the same start?

Design: the measure of PREREGISTRATION.md, section 4, with the package's sampler:
Lambda = mean tau(given gradient) / mean tau(derivative) over 5 seeds of 100 pairs,
CoupledSampler(..., kernel="mala", rng=seed).meeting_times(100, max_iter=10_000) with
default tuning, a pair unmet at 10,000 counted at 10,000, se = sd of the 5 per-seed
ratios / sqrt(5); harmful iff Lambda - 2 se >= 2, harmless iff Lambda + 2 se <= 1.5,
undetermined when the derivative leaves more than 5% of its 500 pairs unmet. Each
argument is a text that one name of cases5.M contains; the family at position i of
cases5.M runs on seeds 510000 + 10 i + (0..4).

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/harm5.py TEXT [TEXT ...] > harm5_<part>.out

The four outputs were made with these arguments, in this order:

- harm5_a.out: "point start at the mode: wrong sign"
  "point start at the mode: sign of coordinate 0" "held at 0: sign"
- harm5_b.out: "right for |x| < 1" "right for |x| < 3" "only for |x| < 3"
  "start sd 0.1: gradient clipped" "all coordinates equal" "2-dimensional subspace"
  "- 1e+10 d=5" "- 1e+09 d=5" "d=5, start sd 2: float32 gradient returned"
  "matched: float32 gradient returned"
- harm5_c.out: "posterior mode: wrong sign" "posterior mode: twice"
  "every group effect started at mu: sumsq sign"
  "every group effect started at mu: pooling sign" "spread start: jacobian"
  "on the boundary: wrong sign"
- harm5_d.out: "float32 throughout, N(0,I) d=50, start sd 2: sign of coordinate 0"

Result: see harm5_a.out to harm5_d.out and REVIEW.md, objections 4 and 5. Harmful: the
wrong sign from a point start at the mode (Lambda 6.71 on N(0, I), 7.11 on the logistic
regression), a coordinate held at 0 (3.58), right for |x| < 1 only (4.19), wrong only
for |x| < 3 (3.26), every coordinate equal (17.8), a subspace (11.3), a constant of 1e9
or 1e10 (14.9, 14.1), a start on a boundary (4.02), a float32 target at d = 50 (5.94).
Harmless: the two float32 gradients returned as float64 (1.000), a clipped gradient,
right for |x| < 3, twice the gradient from the posterior mode. Unclassed: the sign of one
coordinate from a point start at the mode (2.28). Undetermined: the hierarchical model.
"""

import sys
import time
import warnings

import cases5
import numpy as np

from telescopic import CoupledSampler

PAIRS, MAX_ITER = 100, 10_000


def package_run(case, seed):
    lp, g, init = case(seed) if callable(case) else case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = CoupledSampler(lp, init, grad=g, vectorize=True, kernel="mala", rng=seed)
        tau = s.meeting_times(PAIRS, max_iter=MAX_ITER)
    tau = np.asarray(tau, float)
    return (
        float(np.mean(np.minimum(tau, MAX_ITER))),
        int(np.sum(~np.isfinite(tau) | (tau > MAX_ITER))),
        s.kernel.step_size,
    )


for key in sys.argv[1:]:
    name = [n for n in cases5.M if key in n]
    assert len(name) == 1, (key, name)
    name = name[0]
    right, wrong = cases5.M[name]
    idx = list(cases5.M).index(name)
    seeds = [510_000 + 10 * idx + k for k in range(5)]
    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        r = package_run(right, seed)
        w = package_run(wrong, seed)
        rows.append((w[0], r[0], w[1], r[1], w[2], r[2]))
        print(
            f"  ROW | {name} | {seed} | given {w[0]:.2f} unmet {w[1]} step {w[2]:.4f} | derivative {r[0]:.2f} unmet {r[1]} step {r[2]:.4f} | ratio {w[0] / r[0]:.3f} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    r = np.array(rows)
    ratios = r[:, 0] / r[:, 1]
    Lam = r[:, 0].mean() / r[:, 1].mean()
    se = ratios.std(ddof=1) / np.sqrt(5)
    cls = (
        "undetermined"
        if r[:, 3].sum() > 0.05 * 500
        else (
            "harmful"
            if Lam - 2 * se >= 2
            else ("harmless" if Lam + 2 * se <= 1.5 else "unclassed")
        )
    )
    print(
        f"RESULT | {name} | Lambda {Lam:.3f} | se {se:.3f} | {cls} | unmet given {int(r[:, 2].sum())}/500 derivative {int(r[:, 3].sum())}/500 | step ratio {np.mean(r[:, 4] / r[:, 5]):.3f} | seeds {seeds}",
        flush=True,
    )
