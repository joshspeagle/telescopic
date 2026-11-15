"""A frozen copy of HMC's leapfrog, and the helpers it calls.

The leapfrog of HMC as it stood before issue #91 jittered its step, copied verbatim but
for its docstrings, tuning and validation, so that a test can compare the new kernel with
it bit for bit.
"""

import math
from typing import cast

import numpy as np

from telescopic._linalg import covariance, matvec
from telescopic.kernels.base import Kernel, State, acceptance


def select(accepted, proposal, current):
    return State(
        np.where(accepted[:, None], proposal.x, current.x),
        np.where(accepted, proposal.log_prob, current.log_prob),
        None
        if current.grad is None
        else np.where(accepted[:, None], cast(np.ndarray, proposal.grad), current.grad),
    )


def evaluate(target, x, need_grad, *, start=False):
    if not need_grad:
        return State(x, target.log_prob(x))
    return State(x, *target.log_prob_and_grad(x, start=start))


def evaluate_proposal(target, x, live=None):
    if np.isfinite(x).all() and (live is None or live.all()):
        return evaluate(target, x, True)
    usable = np.isfinite(x).all(axis=1)
    if live is not None:
        usable &= live
    log_prob, grad = np.full(len(x), -np.inf), np.zeros_like(x)
    rows = usable.nonzero()[0]
    if len(rows):
        log_prob[rows], grad[rows] = target.log_prob_and_grad(x[rows])
    return State(x, log_prob, grad)


class Leapfrog(Kernel):
    def __init__(self, step_size, n_steps, cov):
        self.step_size, self._n_steps = step_size, n_steps
        self.cov, self._chol = covariance("cov", cov)

    @property
    def needs_grad(self):
        return True

    @property
    def n_steps(self):
        if self._n_steps is not None or self.step_size is None:
            return self._n_steps
        return math.ceil(1 / max(self.step_size, 1 / 1000))

    def step(self, target, state, draws):
        end, log_alpha = self._trajectory(target, state, draws.normal(state.x.shape[1]), 1)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, end, state), acceptance(log_alpha)

    def coupled_step(self, target, state_x, state_y, draws):
        n = len(state_x)
        momentum = draws.normal(state_x.x.shape[1])
        both = State(
            np.concatenate([state_x.x, state_y.x]),
            np.concatenate([state_x.log_prob, state_y.log_prob]),
            np.concatenate([state_x.grad, state_y.grad]),
        )
        end, log_alpha = self._trajectory(target, both, np.concatenate([momentum, momentum]), 2)
        log_u = draws.log_uniform()
        moved = select(np.concatenate([log_u, log_u]) < log_alpha, end, both)
        grad = moved.grad
        first = State(moved.x[:n], moved.log_prob[:n], grad[:n])
        return first, State(moved.x[n:], moved.log_prob[n:], grad[n:])

    def _trajectory(self, target, state, momentum, chains):
        chol, epsilon, length = self._chol, self.step_size, self.n_steps
        x, grad = state.x, state.grad

        def product(matrix, vectors):
            stacked = vectors.reshape(chains, -1, vectors.shape[1])
            return matvec(matrix, stacked, blas=target.vectorize).reshape(vectors.shape)

        going = np.ones(len(x), dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            start = state.log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum)
            momentum = momentum + 0.5 * epsilon * product(chol.T, grad)
            for leap in range(length):
                x = x + epsilon * product(chol, momentum)
                reached = evaluate_proposal(target, x, going)
                log_prob, grad = reached.log_prob, reached.grad
                going = np.isfinite(log_prob)
                kick = epsilon if leap < length - 1 else 0.5 * epsilon
                momentum = momentum + kick * product(chol.T, grad)
            log_alpha = log_prob - 0.5 * np.einsum("ij,ij->i", momentum, momentum) - start
        return State(x, log_prob, grad), np.where(going, log_alpha, -np.inf)
