"""The k the automatic pipeline chooses at larger d, against the exact covariance's (M9).

DECISIONS #7 measured, for d <= 20, a median k at most 1.3 times the one from the exact
covariance, and 4.4 times at d = 50. On examples.gaussian(d) (Sigma_ij = 0.9^|i - j|) from
N(0, 4I), for d = 20, 50 and 100 and 3 seeds: the sampler tunes a random walk on its
adaptive Metropolis pilot, then runs its meeting-time pilot of 100 lag-1 pairs, and k is
their 99% quantile. The reference runs the same pilot with the exact covariance and the
scale 2.38/sqrt(d).

Run: python notes/experiments/m9_tuning_dimension.py
"""

import time
import warnings

import numpy as np

from telescopic import CoupledSampler, examples
from telescopic.engine import run
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.tuning import choose_k

for d in (20, 50, 100):
    example = examples.gaussian(d)
    target = Target(example.log_prob, vectorize=True)
    for seed in range(3):
        sampler = CoupledSampler(example.log_prob, example.initial, vectorize=True, rng=seed)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler.tune()
            tuned = choose_k(sampler.meeting_times(100))
        seconds = time.perf_counter() - start
        exact = RandomWalk(example.cov, scale=2.38 / np.sqrt(d))
        reference = run(
            target, exact, example.initial, range(100), k=0, m=0, lag=1, rng=seed, store=False
        )
        k = choose_k(reference.meeting_time)
        print(
            f"d = {d}, seed {seed}: k {tuned} tuned, {k} exact covariance, ratio "
            f"{tuned / k:.2f}; tuning and pilot {seconds:.1f} s",
            flush=True,
        )
