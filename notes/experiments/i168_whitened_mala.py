"""Whitened MALA against the form it replaces (issue #168): the same kernel, moved bits.

Run from the repository's root with a checkout of the commit before the change (3c8d5b2)
as the argument, for example one made by `git worktree add <path> 3c8d5b2`:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
        python -W ignore notes/experiments/i168_whitened_mala.py <path>

Part 1 steps both kernels from the same states and draws, on a banana in d = 2, 10 and 50
with a dense covariance, point by point and vectorized, and compares the acceptance
probabilities, the positions and every accept and meeting decision. Part 2 runs `sample`
with MALA on `examples.gaussian(d)`, d = 2, 10 and 50, 128 replicates, seeds 0 to 4, with
each package, in a subprocess of its own, and compares k, the meeting times and the
estimates of the mean in standard errors.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys

import numpy as np

import telescopic
from telescopic.kernels import MALA, State
from telescopic.target import Target
from telescopic.testing import random_draws

RUN = """
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


def banana(x):
    return -0.5 * (x[:, 0] ** 2 / 2 + ((x[:, 1:] - x[:, :1] ** 2) ** 2 / 0.1).sum(1))


def banana_grad(x):
    g = np.zeros_like(x)
    r = (x[:, 1:] - x[:, :1] ** 2) / 0.1
    g[:, 1:] = -r
    g[:, 0] = -x[:, 0] / 2 + 2 * x[:, 0] * r.sum(1)
    return g


def previous_mala(checkout):
    path = os.path.join(checkout, "src", "telescopic", "kernels", "mala.py")
    spec = importlib.util.spec_from_file_location("previous_mala", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MALA


def same_kernel(previous):
    rng = np.random.default_rng(1)
    worst_acceptance = worst_x = 0.0
    differing = total = 0
    for d in (2, 10, 50):
        a = rng.standard_normal((d, d))
        cov = a @ a.T / d + 0.1 * np.eye(d)
        for vectorize in (False, True):
            target = Target(
                lambda x: banana(np.atleast_2d(x)),
                vectorize=vectorize,
                grad=lambda x: banana_grad(np.atleast_2d(x)),
            )
            for eps in (0.1, 0.4):
                old, new = previous(eps, cov=cov), MALA(eps, cov=cov)
                for seed in range(20):
                    n = 64
                    sx, sy = (
                        State(p, banana(p), banana_grad(p))
                        for p in (0.5 * rng.standard_normal((n, d)) for _ in range(2))
                    )
                    (o, ao), (w, aw) = (
                        k.step(target, sx, random_draws(n, seed)) for k in (old, new)
                    )
                    worst_acceptance = max(worst_acceptance, np.max(np.abs(ao - aw)))
                    pairs = [(o, w, sx)]
                    ox, oy = old.coupled_step(target, sx, sy, random_draws(n, 1000 + seed))
                    wx, wy = new.coupled_step(target, sx, sy, random_draws(n, 1000 + seed))
                    pairs += [(ox, wx, sx), (oy, wy, sy)]
                    for p, q, start in pairs:
                        moved_p, moved_q = ((s.x != start.x).any(1) for s in (p, q))
                        differing += np.sum(moved_p != moved_q)
                        total += n
                        worst_x = max(worst_x, np.max(np.abs(p.x - q.x)))
                    differing += np.sum((ox.x == oy.x).all(1) != (wx.x == wy.x).all(1))
                    total += n
    print(
        f"same states and draws: largest |difference| of acceptance {worst_acceptance:.1e},"
        f" of positions {worst_x:.1e}; differing accept or meeting decisions"
        f" {differing} of {total}"
    )


def seeded(checkout):
    runs = {}
    for label, src in (("previous", os.path.join(checkout, "src")), ("whitened", "src")):
        env = dict(os.environ, PYTHONPATH=src)
        done = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", RUN],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        runs[label] = json.loads(done.stdout)
    moved_k = moved_tau = 0
    worst = 0.0
    for key, (k, tau, value, se) in runs["previous"].items():
        k2, tau2, value2, _ = runs["whitened"][key]
        moved_k += k != k2
        moved_tau += sum(a != b for a, b in zip(tau, tau2, strict=True))
        worst = max(worst, np.max(np.abs(np.subtract(value, value2)) / np.asarray(se)))
    print(
        f"seeded runs: {len(runs['previous'])}; k moved in {moved_k}; meeting times moved:"
        f" {moved_tau} of {128 * len(runs['previous'])}; largest move of an estimate"
        f" {worst:.1e} standard errors"
    )


if __name__ == "__main__":
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(
        f"telescopic at {head} (working tree), NumPy {np.__version__}, Python"
        f" {platform.python_version()}, {platform.machine()}, OMP_NUM_THREADS="
        f"{os.environ.get('OMP_NUM_THREADS')}; telescopic {telescopic.__version__}"
    )
    same_kernel(previous_mala(sys.argv[1]))
    seeded(sys.argv[1])
