"""Replicates for the unbiasedness tests of tests/test_statistics.py (M9, test plan 2.12).

For each case, a pilot of 1,000 replicates gives the per-replicate estimates H of
h(x) = (x, x^2) and, from the X paths, the plain averages P over the averaged times: the
estimator without its bias correction. A case's replicates R make the standard error of
H at most 1/5 of the bias of P in the component where that ratio is largest, so that
dropping the correction shows at |z| >= 5 there; R is at least 1,000. Also recorded:
meeting times, the excess kurtosis of H and the largest share of its sample variance
from one replicate (heavy-tail statistics), and the cost.

Run: PYTHONPATH=tests python notes/experiments/m9_unbiasedness.py [case ...]
"""

import sys
import time

import numpy as np
from scipy import stats
from test_statistics import CASES, moments, run_case

from telescopic.estimators import averaging_times
from telescopic.results import Result

PILOT = 1000

for name, case in CASES.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    start = time.perf_counter()
    replicates = run_case(case, PILOT, np.random.default_rng(12345), keep_paths=True)
    seconds = (time.perf_counter() - start) / PILOT
    example = case.example
    truth = np.concatenate([example.mean, np.diag(example.cov) + example.mean**2])
    estimates = Result(replicates).measure.per_replicate(moments)
    times = averaging_times(case.k, case.m, case.thin)
    plain = np.array([moments(replicates.paths[i][0][times]).mean(axis=0) for i in range(PILOT)])
    bias = plain.mean(axis=0) - truth
    spread = estimates.std(axis=0, ddof=1)
    best = int(np.argmax(bias**2 / spread**2))
    needed = int(np.ceil(25 * spread[best] ** 2 / bias[best] ** 2))
    chosen = max(1000, needed)
    z_plain = bias[best] / (plain[:, best].std(ddof=1) / np.sqrt(chosen))
    centred = estimates - estimates.mean(axis=0)
    share = np.max(centred**2 / np.sum(centred**2, axis=0))
    kurtosis = np.max(stats.kurtosis(estimates, axis=0))
    tau = replicates.meeting_time - case.lag
    print(
        f"{name}: tau - L q50/q99/max {np.quantile(tau, 0.5):.0f}/{np.quantile(tau, 0.99):.0f}/"
        f"{tau.max():.0f}; component {best}: plain bias {bias[best]:+.4f}, sd of H "
        f"{spread[best]:.4f}; R needed {needed}, chosen {chosen}, plain |z| at R "
        f"{abs(z_plain):.1f}; excess kurtosis of H at most {kurtosis:.1f}; largest variance "
        f"share {share:.3f}; {seconds * 1e3:.2f} ms per replicate",
        flush=True,
    )
