"""Whitened MALA against the form it replaces (issue #168): the same kernel, moved bits.

Run from the repository's root, at a clean commit, with a checkout of the commit before
the change (3c8d5b2) as the argument, for example one made by
`git worktree add <path> 3c8d5b2`:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -W ignore notes/experiments/i168_whitened_mala.py <path>

Each part runs each package, this checkout's `src` and the argument's, in a subprocess of
its own, so that no module of one tree runs with the other.

Part 1 steps MALA, RandomWalk and HMC from the same states and draws, with each package,
on a banana in d = 1, 2, 10 and 50 with a dense covariance, point by point and vectorized,
with pairs that start equal, 1e-3 apart and 1 apart. MALA is compared to rounding, in its
acceptance probabilities, its positions and every accept and meeting decision;
RandomWalk and HMC, whose `reflection_maximal` was refactored, bit for bit.

Part 2 runs `sample` with MALA on `examples.gaussian(d)`, d = 2, 10 and 50, 128 replicates,
seeds 0 to 4, and compares k, the meeting times and the estimates of the mean in
standard errors.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np

STEPS = """
import sys
import numpy as np
from telescopic.kernels import HMC, MALA, RandomWalk, State
from telescopic.target import Target
from telescopic.testing import random_draws

def log_prob(x):
    x = np.atleast_2d(x)
    return -0.5 * (x[:, 0] ** 2 / 2 + ((x[:, 1:] - x[:, :1] ** 2) ** 2 / 0.1).sum(1))

def grad(x):
    x = np.atleast_2d(x)
    g, r = np.zeros_like(x), (x[:, 1:] - x[:, :1] ** 2) / 0.1
    g[:, 1:] = -r
    g[:, 0] = -x[:, 0] / 2 + 2 * x[:, 0] * r.sum(1)
    return g

rng = np.random.default_rng(1)
out = {}
for d in (1, 2, 10, 50):
    a = rng.standard_normal((d, d))
    cov = a @ a.T / d + 0.1 * np.eye(d)
    for vectorize in (False, True):
        target = Target(log_prob, grad=grad, vectorize=vectorize)
        kernels = {"mala": MALA(0.3, cov=cov), "rw": RandomWalk(cov=0.25 * cov),
                   "hmc": HMC(0.1, 5, cov=cov)}
        for seed in range(10):
            n = 64
            px = 0.5 * rng.standard_normal((n, d))
            py = px + rng.choice([0.0, 1e-3, 1.0], (n, 1)) * rng.standard_normal((n, d))
            sx, sy = (State(p, log_prob(p), grad(p)) for p in (px, py))
            for name, kernel in kernels.items():
                key = f"{name}-{d}-{vectorize}-{seed}"
                state, accept = kernel.step(target, sx, random_draws(n, seed))
                cx, cy = kernel.coupled_step(target, sx, sy, random_draws(n, 1000 + seed))
                out[key + "-step"], out[key + "-accept"] = state.x, accept
                out[key + "-x"], out[key + "-y"] = cx.x, cy.x
                out[key + "-sx"], out[key + "-sy"] = sx.x, sy.x
np.savez(sys.argv[1], **out)
"""

SEEDED = """
import json, sys
import telescopic
from telescopic import examples, kernels
out = {}
for d in (2, 10, 50):
    ex = examples.gaussian(d)
    for vectorize in (False, True):
        for seed in range(5):
            r = telescopic.sample(ex.log_prob, ex.initial, grad=ex.grad, kernel=kernels.MALA(),
                                  n_replicates=128, vectorize=vectorize, rng=seed, progress=False)
            e = r.mean()
            out[f"{d}-{vectorize}-{seed}"] = [r.k, r.meeting_times.tolist(),
                                              e.value.tolist(), e.standard_error.tolist()]
json.dump(out, sys.stdout)
"""


def run(code, src, *args):
    env = dict(os.environ, PYTHONPATH=src)
    command = [sys.executable, "-W", "ignore", "-c", code, *args]
    return subprocess.run(command, env=env, capture_output=True, text=True, check=True).stdout


def steps(previous):
    with tempfile.TemporaryDirectory() as folder:
        paths = [os.path.join(folder, f"{label}.npz") for label in ("previous", "whitened")]
        run(STEPS, previous, paths[0])
        run(STEPS, "src", paths[1])
        old, new = (dict(np.load(path)) for path in paths)
    worst_accept = worst_x = 0.0
    differing = total = 0
    bitwise = {"rw": True, "hmc": True}
    for key in old:
        name = key.split("-")[0]
        if name != "mala":
            bitwise[name] &= np.array_equal(old[key], new[key], equal_nan=True)
            continue
        if key.endswith("-accept"):
            worst_accept = max(worst_accept, np.max(np.abs(old[key] - new[key])))
        elif not key.endswith(("-sx", "-sy")):
            worst_x = max(worst_x, np.max(np.abs(old[key] - new[key])))
            start = old[key[: key.rindex("-")] + ("-sy" if key.endswith("-y") else "-sx")]
            moved_old, moved_new = ((a != start).any(1) for a in (old[key], new[key]))
            differing += np.sum(moved_old != moved_new)
            total += len(start)
            if key.endswith("-y"):
                base = key[: key.rindex("-")]
                met_old = (old[base + "-x"] == old[key]).all(1)
                met_new = (new[base + "-x"] == new[key]).all(1)
                differing += np.sum(met_old != met_new)
                total += len(start)
    print(
        f"MALA from the same states and draws: largest |difference| of acceptance"
        f" {worst_accept:.1e}, of positions {worst_x:.1e}; differing accept or meeting"
        f" decisions {differing} of {total}"
    )
    print(
        f"bitwise equal: RandomWalk {bitwise['rw']}, HMC {bitwise['hmc']}"
        f" (steps and coupled steps, {len(old) // 18} configurations each)"
    )


def seeded(previous):
    old, new = (json.loads(run(SEEDED, src)) for src in (previous, "src"))
    moved_k = moved_tau = 0
    worst = 0.0
    for key, (k, tau, value, se) in old.items():
        k2, tau2, value2, _ = new[key]
        moved_k += k != k2
        moved_tau += sum(a != b for a, b in zip(tau, tau2, strict=True))
        worst = max(worst, np.max(np.abs(np.subtract(value, value2)) / np.asarray(se)))
    print(
        f"seeded runs: {len(old)}; k moved in {moved_k}; meeting times moved:"
        f" {moved_tau} of {128 * len(old)}; largest move of an estimate"
        f" {worst:.1e} standard errors"
    )


def git(*args):
    command = ["git", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    previous = os.path.join(sys.argv[1], "src")
    clean = "clean" if not git("status", "--porcelain", "src") else "src modified"
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    print(
        f"whitened at {git('rev-parse', '--short', 'HEAD')} ({clean}), previous at"
        f" {git('-C', sys.argv[1], 'rev-parse', '--short', 'HEAD')}; NumPy {np.__version__},"
        f" Python {platform.python_version()}, {platform.machine()}, {threads}"
    )
    steps(previous)
    seeded(previous)
