"""The default tuning pilot after thinning its states and its factorisations (M9).

Beyond 20 dimensions the pilot now refactors its covariance and keeps its states every
ceil(d/20) steps (DECISIONS #34). On the settings of m9_pilot_length_2.py, its default of
10 d^2 steps on 4 chains at d = 50 (3 seeds) and d = 100 (2 seeds): k from 100 lag-1
meeting times capped at 100,000 against the exact covariance's k, the condition number
of the target whitened by the pilot's covariance, the pilot's seconds and the bytes of
its stored states.

Run: python notes/experiments/m9_pilot_length_3.py
"""

import time

import numpy as np

from telescopic import examples
from telescopic.engine import run
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis, choose_k

for d, seeds in ((50, range(3)), (100, range(2))):
    example = examples.gaussian(d)
    target = Target(example.log_prob, vectorize=True)

    def k_of(kernel, seed, example=example, target=target):
        pilot = run(
            target,
            kernel,
            example.initial,
            range(100),
            k=0,
            m=0,
            lag=1,
            max_iter=100_000,
            rng=seed,
            store=False,
        )
        try:
            return choose_k(pilot.meeting_time)
        except ValueError:
            return np.inf

    for seed in seeds:
        exact = k_of(RandomWalk(example.cov), seed)
        start = time.perf_counter()
        pilot = adaptive_metropolis(target, example.initial, seed)
        seconds = time.perf_counter() - start
        kernel, _ = RandomWalk().tune(target, example.initial, seed, pilot=pilot)
        whitening = np.linalg.inv(np.linalg.cholesky(pilot.cov))
        eigenvalues = np.linalg.eigvalsh(whitening @ example.cov @ whitening.T)
        tuned = k_of(kernel, seed)
        print(
            f"d = {d}, seed {seed}, {pilot.n_steps} steps: k {tuned} against {exact}, ratio "
            f"{tuned / exact:.2f}; condition number {eigenvalues.max() / eigenvalues.min():.1f}; "
            f"pilot {seconds:.1f} s; states {pilot.samples.nbytes / 1e6:.1f} MB",
            flush=True,
        )
