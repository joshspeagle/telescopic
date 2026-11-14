"""The TV bound with censored pairs against a transcription of its definition (#82).

The engine censors a pair after the comparison at `max_iter` fails, so a censored pair's
meeting time is at least the cap plus 1. The question is whether `Result.tv_bound` and
`tv_bound_at_k` of `Result.diagnostics()` equal Biswas, Jacob and Vanetti's bound with
that meeting time, counted pair by pair as #{j >= 1 : tau > t + jL} and averaged, on runs
with many censored pairs. 20 runs of 60 replicates of a 2-D standard normal, a random walk
of proposal covariance 0.3 I from N((4, -4), 9 I), k = 2, m = 12, lags 1 to 4 and caps 13
to 17. Result: 16 to 39 pairs censored per run, and the largest difference over every
run, every t from 0 to cap + 4 and the bound at k is 1.1e-16.

Run: python notes/experiments/i82_tv_censored.py
"""

import warnings

import numpy as np

from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target

warnings.simplefilter("ignore")  # the censoring warnings
target = Target(lambda x: -0.5 * np.sum(x**2, axis=-1), vectorize=True)
worst = 0.0
for seed in range(20):
    lag, cap = 1 + seed % 4, 13 + seed % 5
    replicates = run(
        target,
        RandomWalk(0.3 * np.eye(2)),
        Normal([4.0, -4.0], [3.0, 3.0]),
        range(60),
        rng=seed,
        k=2,
        m=12,
        lag=lag,
        max_iter=cap,
    )
    result = Result(replicates)
    tau = np.where(np.isfinite(result.meeting_times), result.meeting_times, cap + 1)
    t = np.arange(cap + 5)
    counts = np.array([[sum(1 for j in range(1, 100) if s > u + j * lag) for u in t] for s in tau])
    expected = np.minimum(1, counts.mean(axis=0))
    at_k = min(1.0, counts[:, result.k].mean())
    worst = max(
        worst,
        float(np.max(np.abs(result.tv_bound(t) - expected))),
        abs(result.diagnostics()["tv_bound_at_k"] - at_k),
    )
    print(f"seed {seed}: lag {lag}, cap {cap}, {result.n_censored} of 60 censored")
print(f"largest difference from the transcription: {worst:.2g}")
