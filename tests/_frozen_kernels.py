"""Frozen copies of RandomWalk, MALA and Independence and the helpers they call.

The steps of the three kernels as they stood before issue #77 re-based them on one
Metropolis-Hastings kernel, copied verbatim but for their docstrings, tuning and
validation, so that a test can compare the re-based kernels with them bit for bit. They
are removed once the re-basing has merged.
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import numpy as np

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._types import log_densities
from telescopic.couplings import maximal
from telescopic.kernels.base import Kernel, State, acceptance, log_ratio
from telescopic.tuning import OPTIMAL_SCALE


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


def evaluate_pair(evaluate_rows: Callable, x, y, identical):
    distinct = (~identical).nonzero()[0]
    n = len(x)
    both = evaluate_rows(np.concatenate([x, y[distinct]]))
    source = np.arange(n)
    source[distinct] = np.arange(n, len(both))
    first = State(both.x[:n], both.log_prob[:n], None if both.grad is None else both.grad[:n])
    return first, both.take(source)


def reflection_maximal(mu_x, mu_y, chol, inverse, draws, *, blas=False):
    with np.errstate(invalid="ignore"):
        difference = mu_x - mu_y
        delta = matvec(inverse, difference, blas=blas)
    z, identical, shift = _reflection(delta, draws, difference)
    step = matvec(chol, z, blas=blas)
    x = mu_x + step
    y = np.where(identical[:, None], x, mu_y + step - shift)
    return x, y, identical


def _reflection(delta, draws, direction):
    z = draws.normal(delta.shape[1])
    log_u = draws.log_uniform()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared = np.einsum("ij,ij->i", delta, delta)
        along = np.einsum("ij,ij->i", z, delta)
        identical = log_u < -along - 0.5 * squared
        coefficient = 2.0 * along / squared
    finite = (squared > 0) & (squared < np.inf)
    shift = np.multiply(
        coefficient[:, None], direction, out=np.zeros_like(direction), where=finite[:, None]
    )
    return z, identical, shift


class RandomWalk(Kernel):
    def __init__(self, cov, *, scale=None, coupling="reflection"):
        self.scale = scale
        self.coupling = coupling
        self.cov, chol = covariance("cov", cov)
        scale = OPTIMAL_SCALE / np.sqrt(len(chol)) if self.scale is None else self.scale
        scaled = scale * chol
        self._factors = scaled, inverse_lower(scaled)

    def step(self, target, state, draws):
        chol, _ = self._factors
        x = state.x + matvec(chol, draws.normal(len(chol)), blas=target.vectorize)
        proposal = evaluate(target, x, state.grad is not None)
        log_alpha = log_ratio(proposal.log_prob, state.log_prob)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(self, target, state_x, state_y, draws):
        chol, inverse = self._factors
        if self.coupling == "reflection":
            x, y, identical = reflection_maximal(
                state_x.x, state_y.x, chol, inverse, draws, blas=target.vectorize
            )
        else:
            x, y, identical, _ = maximal(
                [_Gaussian(mean, chol, inverse) for mean in state_x.x],
                [_Gaussian(mean, chol, inverse) for mean in state_y.x],
                draws,
            )
        proposal_x, proposal_y = evaluate_pair(
            partial(evaluate, target, need_grad=state_x.grad is not None), x, y, identical
        )
        log_u = draws.log_uniform()
        state_x = select(
            log_u < log_ratio(proposal_x.log_prob, state_x.log_prob), proposal_x, state_x
        )
        state_y = select(
            log_u < log_ratio(proposal_y.log_prob, state_y.log_prob), proposal_y, state_y
        )
        return state_x, state_y


class _Gaussian:
    def __init__(self, mean, chol, inverse):
        self.mean, self.chol, self.inverse = mean, chol, inverse

    def rvs(self, *, random_state):
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, x):
        z = self.inverse @ (x - self.mean)
        return float(-0.5 * z @ z)


class MALA(Kernel):
    def __init__(self, step_size, *, cov):
        self.step_size = step_size
        self.cov, chol = covariance("cov", cov)
        self._factors = step_size * chol, inverse_lower(chol) / step_size

    def step(self, target, state, draws):
        chol, _ = self._factors
        blas = target.vectorize
        drift = _drift(state, chol, blas)
        z = draws.normal(len(chol))
        proposal = evaluate_proposal(target, state.x + matvec(chol, 0.5 * drift + z, blas=blas))
        log_alpha = _log_alpha(state, proposal, z, drift, chol, blas)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(self, target, state_x, state_y, draws):
        chol, inverse = self._factors
        blas = target.vectorize
        drift_x, drift_y = _drift(state_x, chol, blas), _drift(state_y, chol, blas)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = matvec(inverse, state_x.x - state_y.x, blas=blas) + 0.5 * (drift_x - drift_y)
        z, identical, shift = _reflection(delta, draws, delta)
        residual_y = np.where(identical[:, None], z + delta, z - shift)
        x = state_x.x + matvec(chol, 0.5 * drift_x + z, blas=blas)
        y = np.where(
            identical[:, None], x, state_y.x + matvec(chol, 0.5 * drift_y + residual_y, blas=blas)
        )
        proposal_x, proposal_y = evaluate_pair(partial(evaluate_proposal, target), x, y, identical)
        log_u = draws.log_uniform()
        state_x = select(
            log_u < _log_alpha(state_x, proposal_x, z, drift_x, chol, blas), proposal_x, state_x
        )
        state_y = select(
            log_u < _log_alpha(state_y, proposal_y, residual_y, drift_y, chol, blas),
            proposal_y,
            state_y,
        )
        return state_x, state_y


def _drift(state, chol, blas):
    return matvec(chol.T, state.grad, blas=blas)


def _log_alpha(current, proposal, residual, drift, chol, blas):
    with np.errstate(over="ignore", invalid="ignore"):
        back = residual + 0.5 * (drift + _drift(proposal, chol, blas))
        log_q = 0.5 * (
            np.einsum("ij,ij->i", residual, residual) - np.einsum("ij,ij->i", back, back)
        )
        log_alpha = proposal.log_prob - current.log_prob + log_q
    return log_alpha


class Independence(Kernel):
    def __init__(self, proposal):
        self.proposal = proposal

    def step(self, target, state, draws):
        q = self.proposal
        proposal = evaluate(target, q.sample(draws), state.grad is not None)
        new, old = _log_weights(q, proposal, state)
        log_alpha = log_ratio(new, old)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(self, target, state_x, state_y, draws):
        q = self.proposal
        proposal = evaluate(target, q.sample(draws), state_x.grad is not None)
        log_u = draws.log_uniform()
        new, old_x, old_y = _log_weights(q, proposal, state_x, state_y)
        state_x = select(log_u < log_ratio(new, old_x), proposal, state_x)
        state_y = select(log_u < log_ratio(new, old_y), proposal, state_y)
        return state_x, state_y


def _log_weights(q, proposal, *states):
    x = np.concatenate([proposal.x, *(state.x for state in states)])
    log_q = log_densities("proposal log_density", q.log_density(x), x).reshape(1 + len(states), -1)
    new = proposal.log_prob - np.where(log_q[0] == -np.inf, np.inf, log_q[0])
    return [new] + [
        state.log_prob - np.where(state.log_prob == -np.inf, 0.0, part)
        for state, part in zip(states, log_q[1:], strict=True)
    ]
