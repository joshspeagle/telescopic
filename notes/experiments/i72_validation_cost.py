"""The cost of the input checks of issue #72 on the paths that run every iteration.

Most checks run once, at construction or at a public function's boundary. Three run
inside loops:

- `Target._call`, at every evaluation: it now converts what the user's function returns
  without a dtype, checks the kind, and converts to float64 only when the dtype differs
  (DECISIONS #109, #110); the scan for NaN that found `None` on the pointwise path
  is gone.
- the tuning pilot (`adaptive_metropolis`), at every factorization of the covariance:
  NumPy's error state and one `isfinite` of a d x d matrix (DECISIONS #112).
- `Gibbs(coupling="crn")`, which converts each conditional's mean and covariance with
  `_types.vector` and `_linalg.square_root` once per row and step (DECISIONS #103).

Each is compared with the code of the merge base of this checkout's HEAD and `origin/main`,
which the output names: the old module is loaded from git beside the new one, and each
round times both, in alternating order. The ratio new/old of each round is summarized by
its quartiles; a ratio whose quartiles span 1 is not distinguishable from 1, and the
output says so. The output also records the commit, which must be clean (`git describe`
without `-dirty`) for the figures to belong to it, the CPU model, the thread settings and
the versions of Python, NumPy and SciPy. Workloads: N(0, I_d)
evaluated point by point at n = 4 (the tuning pilot's batch) and n = 512 (two lanes per
pair of a full batch), and vectorized at n = 512; the pilot at d = 2 and d = 30 with 2000
steps on a vectorized N(0, I_d); one coupled Gibbs step of 512 rows on a two-dimensional
Gaussian with two blocks of one coordinate whose conditionals are normal with `mean` and
`cov` values.

Result (`i72_validation_cost.out`, the uncommitted code of this change against commit
6623397, so the commit reads `6623397-dirty`): `Target.log_prob` costs the same at
n = 512, pointwise or vectorized (medians 1.003 and 1.001), and less pointwise at n = 4
(0.903), without the scan for NaN; the pilot costs 5.4% more at d = 2, where the toy
log-density is almost free, so the error state and the finiteness test are a visible
share of a step, and at d = 30 its ratio (median 1.024) has quartiles that span 1, so
it is not distinguishable from 1; the Gibbs step with common random numbers costs 3.0%
more, from `real_array`'s conversion and dtype test on each conditional's mean and
covariance.

Run from the repository root: PYTHONPATH=src python notes/experiments/i72_validation_cost.py
"""

import functools
import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy

import telescopic
from telescopic import tuning
from telescopic.initial import Normal
from telescopic.kernels import Gibbs, gibbs
from telescopic.target import Target
from telescopic.testing import random_draws


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


BEFORE = git("merge-base", "HEAD", "origin/main").strip()


def load(path, name):
    source = git("show", f"{BEFORE}:src/telescopic/{path}")
    with tempfile.TemporaryDirectory() as folder:
        file_path = os.path.join(folder, f"{name}.py")
        with open(file_path, "w") as file:
            file.write(source)
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # for its dataclasses
        spec.loader.exec_module(module)
    return module


old_target = load("target.py", "old_target")
old_tuning = load("tuning.py", "old_tuning")
old_types = load("_types.py", "old_types")
old_linalg = load("_linalg.py", "old_linalg")


def point(x):
    return -0.5 * float(x @ x)


def batch(x):
    return -0.5 * np.einsum("ij,ij->i", x, x)


def ratios(calls, inner, rounds):
    # calls: {"old": f, "new": g}; the ratio of the two times of each round.
    out = []
    for r in range(rounds):
        seconds = {}
        for name in ("old", "new") if r % 2 else ("new", "old"):
            start = time.perf_counter()
            for _ in range(inner):
                calls[name]()
            seconds[name] = time.perf_counter() - start
        out.append(seconds["new"] / seconds["old"])
    return np.quantile(out, [0.25, 0.5, 0.75])


def report(label, q):
    span = "; quartiles span 1: not distinguishable from 1" if q[0] <= 1 <= q[2] else ""
    print(f"{label:42}: new/old median {q[1]:.3f}, quartiles {q[0]:.3f} to {q[2]:.3f}{span}")


def cpu_model():
    with open("/proc/cpuinfo") as file:
        for line in file:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


class Conditional:
    # The conditional of one coordinate of N(0, [[1, rho], [rho, 1]]) given the other.
    def __init__(self, mean, variance):
        self.mean, self.cov = mean, variance


def conditional(x, block, rho=0.5):
    return Conditional(rho * x[1 - block], 1 - rho**2)


def main():
    print(
        f"telescopic {telescopic.__version__}, Python {platform.python_version()}, NumPy "
        f"{np.__version__}, SciPy {scipy.__version__}"
    )
    print(
        f"{cpu_model()}, {platform.machine()}, {os.cpu_count()} CPUs, OMP_NUM_THREADS="
        f"{os.environ.get('OMP_NUM_THREADS')}, OPENBLAS_NUM_THREADS="
        f"{os.environ.get('OPENBLAS_NUM_THREADS')}"
    )
    commit = git("describe", "--always", "--dirty").strip()
    clean = (
        "" if not commit.endswith("-dirty") else " (uncommitted changes: not this commit's code)"
    )
    print(f"commit {commit}{clean}, against the merge base with origin/main, {BEFORE[:7]}")
    print("load average before:", tuple(round(v, 2) for v in os.getloadavg()))

    for n, function, vectorize, rounds in (
        (4, point, False, 2001),
        (512, point, False, 201),
        (512, batch, True, 2001),
    ):
        x = np.random.default_rng(0).standard_normal((n, 2))
        targets = {
            "old": old_target.Target(function, vectorize=vectorize),
            "new": Target(function, vectorize=vectorize),
        }
        calls = {name: functools.partial(t.log_prob, x) for name, t in targets.items()}
        kind = "vectorized" if vectorize else "pointwise"
        report(f"Target.log_prob, {kind}, n = {n}", ratios(calls, 20, rounds))

    for d in (2, 30):
        target = Target(batch, vectorize=True)
        initial = Normal(np.zeros(d), np.eye(d))
        calls = {
            name: functools.partial(module.adaptive_metropolis, target, initial, 0, n_steps=2000)
            for name, module in (("old", old_tuning), ("new", tuning))
        }
        report(f"adaptive_metropolis, d = {d}, 2000 steps", ratios(calls, 1, 41))

    kernel = Gibbs(conditional, [[0], [1]], coupling="crn")
    target = Target(None)
    n = 512
    start = kernel.init_state(target, np.random.default_rng(1).standard_normal((2 * n, 2)))
    x, y = start.take(slice(0, n)), start.take(slice(n, None))
    new_helpers = gibbs.vector, gibbs.square_root

    def step(helpers):
        gibbs.vector, gibbs.square_root = helpers
        kernel.coupled_step(target, x, y, random_draws(n, 2))

    calls = {
        "old": functools.partial(step, (old_types.vector, old_linalg.square_root)),
        "new": functools.partial(step, new_helpers),
    }
    report(f"Gibbs crn coupled_step, n = {n}", ratios(calls, 1, 101))
    gibbs.vector, gibbs.square_root = new_helpers
    print("load average after:", tuple(round(v, 2) for v in os.getloadavg()))


if __name__ == "__main__":
    main()
