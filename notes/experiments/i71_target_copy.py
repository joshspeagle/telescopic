"""The cost of copying what a vectorized target function returns (issue #71, DECISIONS #94).

`Target` now copies every output of `log_prob` and `grad`. This compares it with the
`Target` of commit c961089, which copied nothing, on a cheap vectorized target,
N(0, I_d) at d = 2 and d = 100, on n = 512 rows (the 256 lanes of a coupled batch):
`log_prob`, `log_prob_and_grad`, and one `MALA.step`. Each round times both versions, in
alternating order, and the ratio new/old of each round is summarized by its quartiles.

Run from the repository root: PYTHONPATH=src python notes/experiments/i71_target_copy.py
"""

import functools
import importlib.util
import os
import platform
import subprocess
import tempfile
import time

import numpy as np

import telescopic
from telescopic.kernels import MALA
from telescopic.target import Target
from telescopic.testing import random_draws

BEFORE = "c961089"
source = subprocess.run(
    ["git", "show", f"{BEFORE}:src/telescopic/target.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "old_target.py")
    with open(path, "w") as file:
        file.write(source)
    spec = importlib.util.spec_from_file_location("old_target", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)


def log_prob(x):
    return -0.5 * np.einsum("ij,ij->i", x, x)


def grad(x):
    return -x


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


def main():
    print(
        f"telescopic {telescopic.__version__}, Python {platform.python_version()}, NumPy "
        f"{np.__version__}, {platform.machine()}, {os.cpu_count()} CPUs, OMP_NUM_THREADS="
        f"{os.environ.get('OMP_NUM_THREADS')}, OPENBLAS_NUM_THREADS="
        f"{os.environ.get('OPENBLAS_NUM_THREADS')}"
    )
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"commit {commit}, {platform.processor() or platform.machine()}")
    print("load average before:", tuple(round(v, 2) for v in os.getloadavg()))
    n = 512
    for d in (2, 100):
        x = np.random.default_rng(0).standard_normal((n, d))
        targets = {
            "old": old.Target(log_prob, grad, vectorize=True),
            "new": Target(log_prob, grad, vectorize=True),
        }
        for method in ("log_prob", "log_prob_and_grad"):
            calls = {name: functools.partial(getattr(t, method), x) for name, t in targets.items()}
            q = ratios(calls, 20, 2001)
            print(
                f"d = {d:3}, {method:17}: new/old median {q[1]:.3f}, quartiles {q[0]:.3f} to {q[2]:.3f}"
            )
        kernel = MALA(d**-0.25, cov=np.eye(d))
        start = kernel.init_state(targets["new"], x)
        draws = random_draws(n, 0)
        calls = {
            name: functools.partial(kernel.step, t, start, draws) for name, t in targets.items()
        }
        q = ratios(calls, 10, 1001)
        print(
            f"d = {d:3}, {'MALA.step':17}: new/old median {q[1]:.3f}, quartiles {q[0]:.3f} to {q[2]:.3f}"
        )
    print("load average after:", tuple(round(v, 2) for v in os.getloadavg()))


if __name__ == "__main__":
    main()
