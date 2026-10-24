"""The tuning pilot of 10 d^2 steps at d = 15, 50 and 100, against more chains (M9).

At d = 20 and 50, about 10 d^2 adaptive Metropolis steps on 4 chains gave k within 1.4
times the exact covariance's (m9_pilot_length.out). On examples.gaussian(d) from
N(0, 4I), each line runs one pilot and gives k from 100 lag-1 meeting times capped at
100,000 iterations, against the exact covariance's k; the condition number of the
target whitened by the pilot's covariance; and the pilot's seconds:

- d = 15, 2 seeds: 200 d = 3000 steps and 10 d^2 = 2250 steps, on 4 chains;
- d = 50, 3 seeds: 200 d and 10 d^2 steps on 4 chains, and 200 d steps on 4 ceil(d/20)
  = 12 chains, the alternative of the M9 altitude review;
- d = 100, 2 seeds: 10 d^2 steps on 4 chains, and 200 d steps on 20 chains, the same
  number of evaluations. 200 d steps on 4 chains are left out: their meeting-time pilot
  left more than 1% of its pairs unmet at 100,000 (m9_tuning_dimension.out).

Run: python notes/experiments/m9_pilot_length_2.py
"""

import time

import numpy as np

from telescopic import examples, tuning
from telescopic.engine import run
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.tuning import adaptive_metropolis, choose_k

CASES = {
    15: (range(2), [(4, 200 * 15), (4, 10 * 15**2)]),
    50: (range(3), [(4, 200 * 50), (4, 10 * 50**2), (12, 200 * 50)]),
    100: (range(2), [(4, 10 * 100**2), (20, 200 * 100)]),
}

for d, (seeds, configurations) in CASES.items():
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
        for n_chains, n_steps in configurations:
            tuning.N_CHAINS = n_chains
            start = time.perf_counter()
            pilot = adaptive_metropolis(target, example.initial, seed, n_steps=n_steps)
            seconds = time.perf_counter() - start
            kernel, _ = RandomWalk().tune(target, example.initial, seed, pilot=pilot)
            tuning.N_CHAINS = 4
            whitening = np.linalg.inv(np.linalg.cholesky(pilot.cov))
            eigenvalues = np.linalg.eigvalsh(whitening @ example.cov @ whitening.T)
            tuned = k_of(kernel, seed)
            print(
                f"d = {d}, seed {seed}, {n_chains} chains of {n_steps} steps: k {tuned} against "
                f"{exact}, ratio {tuned / exact:.2f}; condition number "
                f"{eigenvalues.max() / eigenvalues.min():.1f}; pilot {seconds:.1f} s",
                flush=True,
            )
