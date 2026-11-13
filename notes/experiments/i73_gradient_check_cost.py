"""The cost of the gradient check against a default run (issue #73, #82 ruling 5).

Question: what does the check that `CoupledSampler` runs at construction, for a kernel
that needs gradients, cost next to the run it precedes? For two targets with MALA, each
from `sample()`'s defaults (R = 100, tuning, the pilot and the run): the evaluations of
`log_prob` and `grad` that the check makes, counted by the target after construction,
against those of the whole run, counted by the target after it; and the check's time,
the best of 20 constructions with the check less the best of 20 without, against the
best of 3 runs. One thread for BLAS.

Result (in the .out file beside this script): the check makes 3 (4d + 1) = 63
evaluations of `log_prob` and 3 of `grad` at d = 5, 0.3% and 0.02% of the run's, and
takes 0.1% to 0.2% of its time.

Run: python notes/experiments/i73_gradient_check_cost.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
from scipy import special

from telescopic import CoupledSampler
from telescopic.initial import Normal


def gaussian(d=5):
    return (
        lambda x: -0.5 * np.sum(x * x, axis=-1),
        lambda x: -x,
        Normal(np.zeros(d), np.full(d, 4.0)),
    )


def logistic(n=10_000, d=5):
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, d))
    labels = rng.random(n) < special.expit(features @ np.linspace(-1, 1, d))
    signs = np.where(labels, 1.0, -1.0)

    def log_prob(beta):
        margins = signs * (features @ beta.T).T
        return -np.sum(np.logaddexp(0, -margins), axis=-1) - 0.5 * np.sum(beta**2, axis=-1)

    def grad(beta):
        margins = signs * (features @ beta.T).T
        return (signs * special.expit(-margins)) @ features - beta

    return log_prob, grad, Normal(np.zeros(d), np.full(d, 0.01))


def best(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
)
print(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
print(f"machine: {platform.machine()}, {os.cpu_count()} cpus; OMP_NUM_THREADS=1")
print(f"commit {commit.stdout.strip()} with the working tree's changes")
warnings.simplefilter("ignore")
for name, (log_prob, grad, initial) in {
    "N(0, I), d = 5": gaussian(),
    "logistic, n = 10^4, d = 5": logistic(),
}.items():

    def build(check, log_prob=log_prob, grad=grad, initial=initial):
        return CoupledSampler(
            log_prob, initial, grad=grad, vectorize=True, kernel="mala", check_grad=check, rng=1
        )

    checked = build(True).target
    sampler = build(False)
    sampler.run(100)
    run = sampler.target
    check_time = best(lambda build=build: build(True), 20) - best(
        lambda build=build: build(False), 20
    )
    run_time = best(lambda build=build: build(False).run(100), 3)
    print(name)
    print(
        f"  check: {checked.n_log_prob} log_prob, {checked.n_grad} grad; run: "
        f"{run.n_log_prob} log_prob, {run.n_grad} grad; ratio of evaluations "
        f"{checked.n_log_prob / run.n_log_prob:.2g} and {checked.n_grad / run.n_grad:.2g}"
    )
    print(f"  check time over run time: {check_time / run_time:.2g}")
