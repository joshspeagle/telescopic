"""Does the unequal-scales failure of i137_unequal.py hold at the package's own cap? (#137)

Question: on N(0, diag(s^2)), d = 10, s = geomspace(1, 1/4, 10), started from
N(0, 4 I), can k be chosen by `CoupledSampler(..., kernel=HMC(cov=np.eye(10))).run(1)`,
with the package's pilot of 100 pairs capped at 100,000 iterations, at seeds 0 to 9?
And with the default HMC(), whose covariance the package tunes, as a control?

Design: the target is coded here, as the agent that reproduced the finding coded it from
the code alone (issue #137, second comment); i137_unequal.py builds the same density with
the package's example functions, whose rounding differs, so the tuned steps differ in
their third or fourth digit. For each seed and kernel: the tuned step and n_steps, the
pilot pairs unmet at the cap, the largest factor |cos(n theta)| over the exact whitened
modes of the tuned covariance (theta = 2 arcsin(eps w / 2)), and k or the error. Then:
- one trajectory of n leapfrog steps with a shared momentum on a one-dimensional
  Gaussian of frequency w, from two positions, against cos(n theta), at the tuned step
  of each seed where k is not chosen, on its slowest-contracting mode;
- the first unmet pilot pair of each such seed, rerun with the pilot's streams
  (`engine.run` with the pilot's seed and index) with a cap of 1,000,000 and with its
  paths kept: its meeting time, and the share of the squared whitened distance of the
  lag-1 pair at iteration 100,000 that lies along each mode of the target.

Result (i137_reproduce_unequal.out): with HMC(cov=np.eye(10)), k cannot be chosen at 3 of
10 seeds (3, 4 and 5: 54, 23 and 12 of 100 pairs unmet at 100,000), all with the step
near 0.30 and 4 leapfrog steps; the largest factor is 0.99993 to 0.99997 where k is not
chosen and at most 0.99858 where it is, and k is 1,566 to 10,134 where it is chosen. The
one-dimensional check gives the ratio of distances equal to cos(n theta) to the last
digits. The followed pairs meet at 110,807 (seed 3), 261,720 (seed 4) and 107,685
(seed 5) iterations: the three followed pairs all meet, as a heavy tail of meeting times
would have them, not pairs that cannot meet. At iteration 100,000 all of their whitened distance lies along the mode of
frequency 2.52, where n theta is near pi. The control HMC() gives k at every
seed (50 to 65), with 2 leapfrog steps and largest factors 0.24 to 0.42.

Run: PYTHONPATH=src:notes/experiments python notes/experiments/i137_reproduce_unequal.py
(needs nothing first; it runs in one go, on one thread)
"""

import warnings

import numpy as np
from i85_setup import setup
from scipy import stats

import telescopic
from telescopic import engine
from telescopic._random import PILOT, namespace_seed
from telescopic.kernels import HMC
from telescopic.sampler import PILOT_MAX_ITER

D = 10
SCALES = np.geomspace(1, 0.25, D)
PRECISION = 1 / SCALES**2
INITIAL = stats.multivariate_normal(np.zeros(D), 4 * np.eye(D))
SEEDS, FOLLOW_CAP = range(10), 1_000_000


def log_prob(x):
    return -0.5 * np.sum(x * x * PRECISION, axis=-1)


def grad(x):
    return -x * PRECISION


def modes(kernel):
    # Whitened frequencies w (sqrt eig of C^T P C, C C^T the kernel's covariance), their
    # eigenvectors and cos(n theta) of each.
    chol = np.linalg.cholesky(kernel.cov)
    w2, vectors = np.linalg.eigh(chol.T @ np.diag(PRECISION) @ chol)
    w = np.sqrt(w2)
    return w, vectors, np.cos(kernel.n_steps * 2 * np.arcsin(kernel.step_size * w / 2)), chol


def leapfrog_ratio(eps, n, w, x=1.0, y=0.2, p=0.37):
    # (end of x - end of y) / (x - y) after n leapfrog steps with the same momentum p.
    def end(q, m):
        m = m - 0.5 * eps * w**2 * q
        for i in range(n):
            q = q + eps * m
            m = m - (eps if i < n - 1 else 0.5 * eps) * w**2 * q
        return q

    return (end(x, p) - end(y, p)) / (x - y)


print(setup())
print(
    f"\nseed, kernel, tuned step, n_steps, pilot pairs unmet at {PILOT_MAX_ITER:,}, largest "
    "|cos(n theta)| over the exact modes, k"
)
failing = []
for name, make in (("HMC(cov=np.eye(10))", lambda: HMC(cov=np.eye(D))), ("HMC()", HMC)):
    for seed in SEEDS:
        sampler = telescopic.CoupledSampler(
            log_prob, INITIAL, grad=grad, vectorize=True, kernel=make(), rng=seed
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sampler.run(1)
                k, error = sampler.settings.k, None
            except ValueError as raised:
                k, error = None, str(raised)
        unmet = np.flatnonzero(np.isinf(sampler.pilot))
        w, vectors, cos, chol = modes(sampler.kernel)
        print(
            f"seed={seed} {name:20} step={sampler.kernel.step_size:.4f} "
            f"n_steps={sampler.kernel.n_steps} unmet={len(unmet):3} "
            f"largest={np.abs(cos).max():.7f} k={k if error is None else 'not chosen'}"
        )
        if error is not None:
            failing.append((seed, sampler, unmet[0], w, vectors, cos, chol))
            if len(failing) == 1:
                print(
                    f"  what a user sees: {[str(c.message) for c in caught]}; ValueError: {error}"
                )

print("\nOne trajectory on a one-dimensional mode: ratio of distances, cos(n theta)")
for seed, sampler, _, w, _, cos, _ in failing:
    mode = np.argmax(np.abs(cos))
    eps, n = sampler.kernel.step_size, sampler.kernel.n_steps
    print(
        f"seed={seed} step={eps!r} n={n} w={w[mode]:.4f}: ratio "
        f"{float(leapfrog_ratio(eps, n, w[mode]))!r}, cos(n theta) {float(cos[mode])!r}"
    )

print(
    f"\nThe first unmet pilot pair of each failing seed, followed with the pilot's streams "
    f"(cap {FOLLOW_CAP:,}): meeting time; share of the squared whitened distance at "
    f"{PILOT_MAX_ITER:,} along each mode, by increasing frequency"
)
for seed, sampler, index, w, vectors, _, chol in failing:
    replicates = engine.run(
        sampler.target,
        sampler.kernel,
        sampler.initial,
        [index],
        k=0,
        m=0,
        lag=1,
        max_iter=FOLLOW_CAP,
        rng=namespace_seed(sampler.seed, PILOT),
        store=False,
        keep_paths=True,
    )
    xs, ys = replicates.paths[index]
    # Lag 1: X_t is compared with Y_(t-1).
    along = vectors.T @ np.linalg.solve(chol, xs[PILOT_MAX_ITER] - ys[PILOT_MAX_ITER - 1])
    share = along**2 / np.sum(along**2)
    print(
        f"seed={seed} pair={index}: met at {replicates.meeting_time[0]:,.0f}; share "
        f"{np.round(share, 6).tolist()} (largest at w={w[np.argmax(share)]:.4f})"
    )
