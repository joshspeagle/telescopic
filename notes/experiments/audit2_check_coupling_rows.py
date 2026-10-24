"""`check_coupling` against couplings that read another row's state (audit 2).

`FirstRow` gives every X chain's acceptance the log-density of the first row's X, and
`ShiftTwo` that of the X two rows on, as a broadcasting or an indexing bug would;
`RandomWalk` is right. Each is checked at the defaults (n = 10,000, alpha = 1e-3) on
N(0, 1), from x, y = 0.3, 0.6 and from 0.0, 1.0, with seeds 0 to 19: the number that
pass. Then a run of `FirstRow`: estimates of E[x] and E[x^2], exactly 0 and 1, from
4,000 replicates started from N(3, 1).

Run: python notes/experiments/audit2_check_coupling_rows.py
"""

import warnings

import numpy as np

import telescopic
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk, State
from telescopic.target import Target
from telescopic.testing import check_coupling


class FirstRow(RandomWalk):
    def coupled_step(self, target, state_x, state_y, draws):
        return _read(self, target, state_x, state_y, draws, np.full(len(state_x), 0))


class ShiftTwo(RandomWalk):
    def coupled_step(self, target, state_x, state_y, draws):
        return _read(self, target, state_x, state_y, draws, np.roll(np.arange(len(state_x)), 2))


def _read(kernel, target, state_x, state_y, draws, rows):
    # A random-walk coupled step whose X chains take the log-densities of other rows; the
    # chains that stay keep their own.
    shared = State(state_x.x, state_x.log_prob[rows])
    new_x, new_y = RandomWalk.coupled_step(kernel, target, shared, state_y, draws)
    stayed = np.all(new_x.x == state_x.x, axis=1)
    return State(new_x.x, np.where(stayed, state_x.log_prob, new_x.log_prob)), new_y


def log_prob(x):
    return -0.5 * np.sum(x**2, axis=-1)


target = Target(log_prob, vectorize=True)
for x, y in ((0.3, 0.6), (0.0, 1.0)):
    for kernel in (RandomWalk(np.eye(1)), FirstRow(np.eye(1)), ShiftTwo(np.eye(1))):
        passed = sum(
            check_coupling(kernel, target, [x], [y], rng=seed).passed() for seed in range(20)
        )
        print(f"x, y = {x}, {y}: {type(kernel).__name__} passed {passed} of 20")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = telescopic.sample(
        log_prob,
        Normal([3.0], [[1.0]]),
        4000,
        vectorize=True,
        kernel=FirstRow(np.eye(1)),
        k=2,
        m=10,
        lag=1,
        rng=2,
    )
estimate = result.estimate(lambda x: np.column_stack([x[:, 0], x[:, 0] ** 2]))
z = (estimate.value - [0.0, 1.0]) / estimate.standard_error
print(
    f"FirstRow run: E[x] = {estimate.value[0]:.3f} +- {estimate.standard_error[0]:.3f} "
    f"(z = {z[0]:.1f}), E[x^2] = {estimate.value[1]:.3f} +- {estimate.standard_error[1]:.3f} "
    f"(z = {z[1]:.1f})"
)
