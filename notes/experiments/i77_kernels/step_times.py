"""Times of a step and a coupled step of the three Metropolis-Hastings kernels (issue #77).

Question: the design of issue #77 re-bases `RandomWalk`, `MALA` and `Independence` on
one Metropolis-Hastings kernel, which adds a layer of calls, and asks for the median time
per step at 256 lanes before and after; a growth above 5% for any kernel goes to Josh
before merging.

Method. Each kernel, tuned by hand (`RandomWalk(cov)`, `MALA(1.65 d^(-1/6), cov=cov)`,
`Independence(MultivariateT(0, 2 cov))`, with cov that of `examples.gaussian(d)`), on
that example's log-density and gradient as a vectorized target, so that the kernels'
products are BLAS calls, at d = 2 and 100 and with 1 and 256 lanes. `step` starts from
fixed states drawn from the target, and `coupled_step` from fixed pairs of independent
draws; the returned states are discarded, so every call does the same work, and the
draws come from `testing.random_draws`. The states are those of the kernel's
`init_state`, so that, since the second pull request of issue #77, `Independence` starts
from states whose cache of log q is valid, as in a run where it alone moves the chains. For each configuration one uncounted round fixes
the number of calls that takes at least 0.2 s, then 7 rounds of that many calls give the
median CPU time per call. CPU time of the process (`time.process_time`), not wall time:
the machine of the recorded runs was shared with other jobs that kept its load above 5
on 4 logical cores, which made wall times vary by a factor of up to 3 between rounds.
Run as `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python
notes/experiments/i77_kernels/step_times.py`, with the package to time first on
`PYTHONPATH`; the first line gives the setup. To compare two commits, run it on each in
turn, alternating, several times, and compare the medians over the runs.
"""

import gc
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i85_setup import setup

from telescopic import examples
from telescopic.kernels import MALA, Independence, MultivariateT, RandomWalk
from telescopic.target import Target
from telescopic.testing import random_draws

ROUNDS = 7


def kernels(d, cov):
    return {
        "RandomWalk": RandomWalk(cov),
        "MALA": MALA(1.65 * d ** (-1 / 6), cov=cov),
        "Independence": Independence(MultivariateT(np.zeros(d), 2 * cov)),
    }


def per_call(call):
    number = 1
    while True:
        start = time.process_time()
        for _ in range(number):
            call()
        if time.process_time() - start >= 0.2:
            break
        number *= 2
    times = []
    for _ in range(ROUNDS):
        gc.collect()
        start = time.process_time()
        for _ in range(number):
            call()
        times.append((time.process_time() - start) / number)
    return float(np.median(times))


def main():
    print(setup())
    print(f"{'kernel':13} {'operation':12} {'d':>4} {'lanes':>5} {'median us':>10}")
    for d in (2, 100):
        example = examples.gaussian(d)
        target = Target(example.log_prob, example.grad, vectorize=True)
        chol = np.linalg.cholesky(example.cov)
        rng = np.random.default_rng(1)
        for lanes in (1, 256):
            x, y = (rng.standard_normal((lanes, d)) @ chol.T for _ in range(2))
            for name, kernel in kernels(d, example.cov).items():
                # The states of init_state, as a run starts from: with gradients for the
                # kernel that needs them, and, where the kernel has them, its own fields.
                state_x, state_y = (kernel.init_state(target, z) for z in (x, y))
                step = per_call(partial(kernel.step, target, state_x, random_draws(lanes, 2)))
                coupled = per_call(
                    partial(kernel.coupled_step, target, state_x, state_y, random_draws(lanes, 3))
                )
                print(f"{name:13} {'step':12} {d:4} {lanes:5} {1e6 * step:10.1f}")
                print(f"{name:13} {'coupled_step':12} {d:4} {lanes:5} {1e6 * coupled:10.1f}")


if __name__ == "__main__":
    main()
