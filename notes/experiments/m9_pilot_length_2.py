"""The pilot of 10 d^2 steps at d = 15 and d = 100 (M9), after m9_pilot_length.py.

At d = 20 and 50, about 10 d^2 adaptive Metropolis steps gave k within 1.4 times the exact
covariance's. On examples.gaussian(d) from N(0, 4I), 2 seeds: at d = 15, where 10 d^2 = 2250
is shorter than the current 200 d = 3000, both; at d = 100, the current 200 d = 20,000 and
10 d^2 = 100,000.

Run: python notes/experiments/m9_pilot_length_2.py
"""

import time

import numpy as np

from telescopic import examples
from telescopic.engine import run
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis, choose_k

for d, lengths in ((15, (200 * 15, 10 * 15**2)), (100, (200 * 100, 10 * 100**2))):
    example = examples.gaussian(d)
    target = Target(example.log_prob, vectorize=True)

    def k_of(kernel, seed, example=example, target=target):
        pilot = run(
            target, kernel, example.initial, range(100), k=0, m=0, lag=1, rng=seed, store=False
        )
        return choose_k(pilot.meeting_time)

    for seed in range(2):
        exact = k_of(RandomWalk(example.cov, scale=2.38 / np.sqrt(d)), seed)
        for n_steps in lengths:
            start = time.perf_counter()
            pilot = adaptive_metropolis(target, example.initial, seed, n_steps=n_steps)
            kernel, _ = RandomWalk().tune(target, example.initial, seed, pilot=pilot)
            seconds = time.perf_counter() - start
            tuned = k_of(kernel, seed)
            print(
                f"d = {d}, seed {seed}, pilot of {n_steps} steps: k {tuned} against {exact}, "
                f"ratio {tuned / exact:.2f}; pilot {seconds:.1f} s",
                flush=True,
            )
