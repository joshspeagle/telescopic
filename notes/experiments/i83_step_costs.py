"""What the per-step changes of issue #83 gain, and which seeded outputs they move.

Issue #83 proposed five changes to the costs of a step, which the rulings comment
approved: (p1-04) the reflection-maximal coupling reflects with one product with C
instead of two, reusing C z; (p1-05) a coupled Gibbs sweep counts the coordinates where
the chains differ, where it tested every coordinate outside each block; (p1-06) the
tuning pilot draws its normals and uniforms in blocks of 256 steps, where it made two
requests per step; (p1-07) Independence splits its one call of the proposal's
log-density by slicing, where it called `np.split`; (p1-08) `MultivariateT` computes its
Student-t normalizing constant once, at construction. The issue measured them before the
kernels' products moved to BLAS for vectorized targets; this script measures those that
are made, a section each, on the code as it is now. Two questions: how much faster is each changed function, and which
seeded outputs move, and how far?

The source of commit `BEFORE` (the base of this change) is extracted with `git archive`
into a temporary folder; "after" is this checkout's `src`. Each measurement is a child
interpreter (`python -I`, the source folder first on its path) with one BLAS and OpenMP
thread, and each child times every configuration of its section. Rounds alternate which
source goes first, after one uncounted round; a child times each configuration as the
minimum of 5 repeats of processor time. The ratio before/after of each round is
summarized by its median and its range over the rounds. The targets are N(0, Sigma),
Sigma_ij = 0.9^|i-j| (`examples.gaussian(d)`), whose log-density is a product with a
precomputed factor: vectorized with BLAS, as in `i83_blas_products.py`, or evaluated
point by point.

1. Fixed seeds. Runs with rng = 0 from each source, of a point-by-point and a
   vectorized target: `sample` at d = 10 with RandomWalk, MALA and Independence, all
   tuned (pilot, then k from 100 pilot pairs), 64 replicates, and with Gibbs on the pumps
   example (`log_prob=None`, and its log-density vectorized), 64 replicates; `engine.run`
   with RandomWalk given Sigma at d = 60, 32 replicates, k = 10, m = 50, max_iter =
   20000, and with HMC given Sigma at d = 2, step size 2^(-1/4), 64 replicates, the same
   k, m and max_iter (its coupled trajectories contract until the chains are equal, an
   event that turns on the last bits); `sample` with MALA (rng = 2) and HMC (rng = 3)
   on the logistic regression of the gradients tutorial (d = 6, 100 replicates, its
   initial distribution); `adaptive_metropolis` at d = 20 (its default
   4,000 steps); and `engine.run` with Independence given a Student-t proposal of scale
   1.5 Sigma at d = 20, one lane, 16 replicates, k = 10, m = 100. For each: a hash of
   the meeting times, atoms and weights (of the samples, log-densities and acceptance
   rates for the pilot), before and after, the number of meeting times that differ, and
   the largest change of a coordinate of the pooled estimate of E[x] in units of its
   standard error.
2. Reflection (p1-04). `reflection_maximal` at 256 rows, d = 2, 10, 20, 50 and 200, with
   einsum's products and with BLAS's, the means 0.5 apart per coordinate so that most
   rows are not identical: microseconds per call, and from the same streams (seed 5)
   whether `x` and the flags are bitwise equal, and the largest relative difference of
   `y` on the rows that are not identical.
3. Gibbs (p1-05). Three coupled sweeps of `Gibbs.coupled_step` with the maximal
   coupling, one-coordinate blocks of the conditionals of an AR(1) chain
   (x_i | rest ~ N(0.5 (x_(i-1) + x_(i+1)) / 1.25, 1 / 1.25), a small class with `rvs`
   and `logpdf`), at d = 20 and 200, 16 and 256 rows, with pairs that differ in every
   coordinate, in half of them, or nowhere: milliseconds per three sweeps, and whether
   x and y are bitwise equal before and after.

Result (`i83_step_costs.out`, at a load average of 1.3 to 2.0 on 4 cores shared with
other jobs).
1. RandomWalk's, MALA's and HMC's runs changed their hashes. Their meeting times were
   equal and their pooled estimates of E[x] moved by at most 5.2e-15 standard errors
   (RandomWalk at d = 60), except HMC's on the logistic regression, where one of 100
   meeting times moved and with it the estimates, by up to 0.075 standard errors (point
   by point) and 0.068 (vectorized): HMC's chains meet in a random-walk step of scale
   1e-3 once their trajectories have brought them close, where a difference of rounding
   is a larger share of the distance between them. Independence's, Gibbs's and the
   pilot's hashes did not change, point by point or vectorized.
2. From the same streams `x` and the flags were bitwise equal, and `y` differed by at
   most 9.5e-12 relative (d = 200, einsum). Before/after per round, medians: with einsum
   1.10, 1.10, 1.22, 1.26 and 1.36 at d = 2, 10, 20, 50 and 200; with BLAS 1.09, 1.06,
   1.10, 1.10 and 1.24. No median is below 1, so one form serves every d.
3. Bitwise equal everywhere. At d = 20 no gain: medians 0.93 to 1.02. At d = 200, 1.06
   to 1.10 where the pairs differ and 1.20 to 1.28 where they are equal, at 16 and 256
   rows.

Run from the repository root: OMP_NUM_THREADS=1 PYTHONPATH=src python notes/experiments/i83_step_costs.py
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from i85_setup import setup

ROOT = Path(__file__).resolve().parents[2]
BEFORE = "6b1c75d"
ROUNDS = 7
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# Shared by the children: the targets, a hash, and a timer.
COMMON = """
import hashlib, json, sys, time, warnings
sys.path.insert(0, sys.argv[1])
import numpy as np
from telescopic import examples
from telescopic.target import Target

warnings.simplefilter("ignore")


def gaussian_log_prob(d, vectorize):
    whiten = np.linalg.cholesky(np.linalg.inv(examples.gaussian(d).cov))
    if vectorize:
        return lambda x: -0.5 * np.sum((x @ whiten) ** 2, axis=1)
    return lambda x: -0.5 * float(np.sum((x @ whiten) ** 2))


def gaussian_target(d, vectorize):
    return Target(gaussian_log_prob(d, vectorize), vectorize=vectorize)


def digest(*arrays):
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()[:12]


def timed(call, number):
    times = []
    for _ in range(5):
        start = time.process_time()
        for _ in range(number):
            call()
        times.append((time.process_time() - start) / number)
    return min(times)
"""

# Section 1: hashes and estimates of fixed-seed runs.
HASHES = (
    COMMON
    + """
from scipy import optimize, special

from telescopic import kernels, sample
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.results import Result
from telescopic.tuning import adaptive_metropolis


def summary(result):
    estimate = result.estimate(lambda x: x)
    return {
        "hash": digest(result.meeting_times, result.measure.atoms, result.measure.weights),
        "meeting_time": result.meeting_times.tolist(),
        "value": estimate.value.tolist(),
        "error": estimate.standard_error.tolist(),
    }


# The logistic regression of the gradients tutorial, whose MALA and HMC runs it repeats.
generator = np.random.default_rng(1)
X = np.column_stack([np.ones(200), generator.normal(size=(200, 5))])
y = (generator.random(200) < special.expit(X @ [0.5, 1.0, -0.5, 0.25, 0.0, -1.0])).astype(float)


def logistic_log_prob(beta, X, y):
    eta = beta @ X.T
    return np.sum(y * eta - np.logaddexp(0, eta), axis=-1) - np.sum(beta**2, axis=-1) / 50


def logistic_grad(beta, X, y):
    return (y - special.expit(beta @ X.T)) @ X - beta / 25


fit = optimize.minimize(
    lambda beta: -logistic_log_prob(beta, X, y), np.zeros(6),
    jac=lambda beta: -logistic_grad(beta, X, y),
)
logistic_initial = Normal(mean=fit.x, cov=100 * fit.hess_inv)

outputs = {}
for vectorize in (False, True):
    label = "vectorized" if vectorize else "point by point"
    example = examples.gaussian(10)
    precision = np.linalg.inv(example.cov)
    for name, kernel in (
        ("RandomWalk", kernels.RandomWalk),
        ("MALA", kernels.MALA),
        ("Independence", kernels.Independence),
    ):
        grad = (lambda x: -x @ precision) if name == "MALA" else None
        result = sample(
            gaussian_log_prob(10, vectorize), example.initial,
            n_replicates=64, grad=grad, vectorize=vectorize, kernel=kernel(), rng=0,
        )
        outputs[f"sample {name}, d = 10, {label}"] = summary(result)
    pumps = examples.pumps()
    result = sample(
        pumps.log_prob if vectorize else None, pumps.initial, n_replicates=64,
        vectorize=vectorize, kernel=kernels.Gibbs(pumps.conditional, pumps.blocks), rng=0,
    )
    outputs[f"sample Gibbs, pumps, {label}"] = summary(result)
    example = examples.gaussian(60)
    replicates = run(
        gaussian_target(60, vectorize), kernels.RandomWalk(example.cov), example.initial,
        range(32), k=10, m=50, lag=1, max_iter=20000, rng=0,
    )
    outputs[f"run RandomWalk, d = 60, {label}"] = summary(Result(replicates))
    example = examples.gaussian(2)
    precision = np.linalg.inv(example.cov)
    target = Target(
        gaussian_log_prob(2, vectorize), grad=lambda x: -x @ precision, vectorize=vectorize
    )
    replicates = run(
        target, kernels.HMC(2**-0.25, cov=example.cov), example.initial, range(64), k=10,
        m=50, lag=1, max_iter=20000, rng=0,
    )
    outputs[f"run HMC, d = 2, {label}"] = summary(Result(replicates))
    for name, seed in (("MALA", 2), ("HMC", 3)):
        result = sample(
            logistic_log_prob, logistic_initial, args=(X, y), grad=logistic_grad,
            vectorize=vectorize, kernel=name.lower(), rng=seed,
        )
        outputs[f"sample {name}, logistic regression, {label}"] = summary(result)
    example = examples.gaussian(20)
    pilot = adaptive_metropolis(gaussian_target(20, vectorize), example.initial, rng=0)
    outputs[f"adaptive_metropolis, d = 20, {label}"] = {
        "hash": digest(pilot.samples, pilot.log_prob, pilot.acceptance)
    }
    proposal = kernels.MultivariateT(np.zeros(20), 1.5 * example.cov)
    replicates = run(
        gaussian_target(20, vectorize), kernels.Independence(proposal), example.initial,
        range(16), k=10, m=100, lag=1, rng=0, lanes=1,
    )
    outputs[f"run Independence, d = 20, one lane, {label}"] = summary(Result(replicates))
print(json.dumps(outputs))
"""
)

# Section 2: reflection_maximal per call, and its output from fixed streams.
REFLECTION = (
    COMMON
    + """
from telescopic.couplings import reflection_maximal
from telescopic.testing import random_draws

outputs = {}
for d in (2, 10, 20, 50, 200):
    chol = np.linalg.cholesky(examples.gaussian(d).cov)
    inverse = np.linalg.inv(chol)
    mu_x = np.random.default_rng(4).standard_normal((256, d))
    mu_y = mu_x + 0.5
    for blas in (False, True):
        draws = random_draws(256, 6)
        seconds = timed(lambda: reflection_maximal(mu_x, mu_y, chol, inverse, draws, blas=blas), 200)
        x, y, identical = reflection_maximal(mu_x, mu_y, chol, inverse, random_draws(256, 5), blas=blas)
        outputs[f"{d} {blas}"] = {
            "time": 1e6 * seconds, "x": x.tolist(), "y": y.tolist(),
            "identical": identical.tolist(),
        }
print(json.dumps(outputs))
"""
)

# Section 3: coupled Gibbs sweeps.
GIBBS = (
    COMMON
    + """
from telescopic.kernels import Gibbs
from telescopic.kernels.base import State
from telescopic.testing import random_draws


class Normal:
    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd

    def rvs(self, random_state):
        return self.mean + self.sd * random_state.standard_normal()

    def logpdf(self, value):
        return -0.5 * ((value - self.mean) / self.sd) ** 2 - np.log(self.sd)


def conditional(x, b):
    left = x[b - 1] if b else 0.0
    right = x[b + 1] if b + 1 < len(x) else 0.0
    return Normal(0.5 * (left + right) / 1.25, 1.25**-0.5)


outputs = {}
for d in (20, 200):
    kernel = Gibbs(conditional, [np.array([i]) for i in range(d)])
    for n in (16, 256):
        x0 = np.random.default_rng(7).standard_normal((n, d))
        for differ, offset in (("every", np.ones(d)), ("half", np.arange(d) % 2), ("none", np.zeros(d))):
            y0 = x0 + offset

            def sweeps(seed=8):
                draws = random_draws(n, seed)
                x, y = State(x0, np.zeros(n)), State(y0, np.zeros(n))
                for _ in range(3):
                    x, y = kernel.coupled_step(Target(None), x, y, draws)
                return x, y

            x, y = sweeps()
            outputs[f"{d} {n} {differ}"] = {
                "time": 1e3 * timed(sweeps, 1 if n * d > 4000 else 5),
                "hash": digest(x.x, y.x),
            }
print(json.dumps(outputs))
"""
)


def child(script, source):
    output = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source)],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | THREADS,
    ).stdout
    return json.loads(output)


def extract(commit, folder):
    archive = subprocess.run(
        ["git", "archive", commit, "src/telescopic"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(folder, filter="data")
    return Path(folder) / "src"


def rounds(script, sources):
    # Each source's outputs over ROUNDS rounds, after one uncounted; the order alternates.
    outputs = {label: [] for label in sources}
    for round_ in range(ROUNDS + 1):
        order = list(sources) if round_ % 2 == 0 else list(reversed(sources))
        for label in order:
            result = child(script, sources[label])
            if round_:
                outputs[label].append(result)
    return outputs


def compare(outputs, key):
    # The times of key before and after, medians over the rounds, and the ratio
    # before/after of each round, by its median and range.
    times = {
        label: np.array([output[key]["time"] for output in outputs[label]]) for label in outputs
    }
    ratios = times["before"] / times["after"]
    return (
        f"{np.median(times['before']):8.2f} -> {np.median(times['after']):8.2f}, ratio "
        f"{np.median(ratios):.2f} ({ratios.min():.2f} to {ratios.max():.2f})"
    )


def same(outputs, key):
    return outputs["before"][0][key]["hash"] == outputs["after"][0][key]["hash"]


def hashes(sources):
    print(
        "\n1. Fixed seeds: hash before -> after, meeting times that differ, largest change of E[x]"
    )
    before, after = (child(HASHES, sources[label]) for label in ("before", "after"))
    for name, old in before.items():
        new = after[name]
        line = f"{name:48} {old['hash']} -> {new['hash']}"
        if "meeting_time" in old:
            differ = int(np.sum(np.array(old["meeting_time"]) != new["meeting_time"]))
            change = np.max(np.abs(np.subtract(new["value"], old["value"])) / old["error"])
            line += f"; {differ} meeting times differ; E[x] moved {change:.1e} standard errors"
        print(line)


def reflection(sources):
    print(
        "\n2. Reflection: 256 rows; microseconds per call, median of the rounds, and "
        "before/after per round: median (range)"
    )
    outputs = rounds(REFLECTION, sources)
    for key in outputs["after"][0]:
        d, blas = key.split()
        old, new = outputs["before"][0][key], outputs["after"][0][key]
        identical = np.array(old["identical"])
        equal = np.array_equal(old["x"], new["x"]) and old["identical"] == new["identical"]
        y_old, y_new = np.array(old["y"])[~identical], np.array(new["y"])[~identical]
        relative = np.max(np.abs(y_new - y_old) / np.abs(y_old)) if len(y_old) else 0.0
        print(
            f"d = {d:>3}, {'BLAS  ' if blas == 'True' else 'einsum'}: {compare(outputs, key)}; "
            f"x and flags {'equal' if equal else 'DIFFER'}, {int((~identical).sum())} rows "
            f"not identical, y largest relative difference {relative:.1e}"
        )


def gibbs(sources):
    print(
        "\n3. Gibbs: three coupled sweeps, one-coordinate blocks; milliseconds, median of "
        "the rounds, and before/after per round: median (range)"
    )
    outputs = rounds(GIBBS, sources)
    for key in outputs["after"][0]:
        d, n, differ = key.split()
        print(
            f"d = {d:>3}, {n:>3} rows, differ in {differ:5}: {compare(outputs, key)}; "
            f"x and y {'bitwise equal' if same(outputs, key) else 'DIFFER'}"
        )


def main():
    print(setup())
    print(f"before: commit {BEFORE}; after: this checkout; children run with {THREADS}")
    print(f"load average before: {tuple(round(x, 2) for x in os.getloadavg())}")
    with tempfile.TemporaryDirectory() as folder:
        sources = {"before": extract(BEFORE, folder), "after": ROOT / "src"}
        hashes(sources)
        reflection(sources)
        gibbs(sources)
    print(f"load average after: {tuple(round(x, 2) for x in os.getloadavg())}")


if __name__ == "__main__":
    main()
