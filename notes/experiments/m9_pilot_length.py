"""The k chosen at d = 50 against the length of the adaptive Metropolis pilot (M9).

The study of m9_tuning_dimension.py found k 11 to 17 times the exact covariance's at
d = 50, with the default pilot of max(2000, 200 d) steps. On examples.gaussian(50), from
N(0, 4I), 2 seeds: the random walk tuned on pilots of 200 d, 500 d, 1000 d and 2000 d steps,
its 100-pair meeting-time pilot and k, against the exact covariance's k.

Run: python notes/experiments/m9_pilot_length.py
"""

import time

import numpy as np

from telescopic import examples
from telescopic.engine import run
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis, choose_k

d = 50
example = examples.gaussian(d)
target = Target(example.log_prob, vectorize=True)


def k_of(kernel, seed):
    pilot = run(
        target, kernel, example.initial, range(100), k=0, m=0, lag=1, rng=seed, store=False
    )
    return choose_k(pilot.meeting_time)


for seed in range(2):
    exact = k_of(RandomWalk(example.cov, scale=2.38 / np.sqrt(d)), seed)
    for factor in (200, 500, 1000, 2000):
        start = time.perf_counter()
        pilot = adaptive_metropolis(target, example.initial, seed, n_steps=factor * d)
        kernel, _ = RandomWalk().tune(target, example.initial, seed, pilot=pilot)
        seconds = time.perf_counter() - start
        tuned = k_of(kernel, seed)
        print(
            f"seed {seed}, pilot of {factor} d = {factor * d} steps: k {tuned} against {exact}, "
            f"ratio {tuned / exact:.2f}; pilot {seconds:.1f} s",
            flush=True,
        )
