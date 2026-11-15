"""Times of Independence's step and coupled step at 256 lanes by dimension (issue #77).

Question: the second pull request of issue #77 caches log q in `Independence`. The cache
saves the evaluation of `MultivariateT.log_density` at the current states, whose cost
grows as d^2, and adds the bookkeeping of its fields (a bitwise comparison of the points
and two more fields through `select`), whose cost grows as d. Below which dimension does
the cache cost more than it saves?

Method. As `step_times.py`: `Independence(MultivariateT(0, 2 cov))` with cov that of
`examples.gaussian(d)`, on its vectorized log-density, 256 lanes, states from the
kernel's `init_state` (valid caches, as when the kernel alone moves the chains), the
median CPU time per call of 7 rounds, at d = 2, 3, 5, 10, 20 and 50. Run it alternately
with the base package and the change first on `PYTHONPATH`, with threads pinned
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), and compare the medians
over the runs.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from i85_setup import setup
from step_times import per_call

from telescopic import examples
from telescopic.kernels import Independence, MultivariateT
from telescopic.target import Target
from telescopic.testing import random_draws

LANES = 256


def main():
    print(setup())
    print(f"{'kernel':13} {'operation':12} {'d':>4} {'lanes':>5} {'median us':>10}")
    for d in (2, 3, 5, 10, 20, 50):
        example = examples.gaussian(d)
        target = Target(example.log_prob, vectorize=True)
        chol = np.linalg.cholesky(example.cov)
        rng = np.random.default_rng(1)
        kernel = Independence(MultivariateT(np.zeros(d), 2 * example.cov))
        x, y = (kernel.init_state(target, rng.standard_normal((LANES, d)) @ chol.T) for _ in "xy")
        step = per_call(partial(kernel.step, target, x, random_draws(LANES, 2)))
        coupled = per_call(partial(kernel.coupled_step, target, x, y, random_draws(LANES, 3)))
        print(f"{'Independence':13} {'step':12} {d:4} {LANES:5} {1e6 * step:10.1f}")
        print(f"{'Independence':13} {'coupled_step':12} {d:4} {LANES:5} {1e6 * coupled:10.1f}")


if __name__ == "__main__":
    main()
