"""MALA in whitened coordinates: the prototype of issue #82, item q1-05.

With L = epsilon C (the kernel's `_factors[0]`) and h = L^T grad log pi(x), the whitened
drift, MALA's proposal x' = x + (epsilon^2 / 2) Sigma grad + epsilon C z is
x' = x + L (h / 2 + z), one product with L. Its forward residual in whitened units is
r = L^{-1}(x' - mu(x)) = z, which needs no product, and its backward residual is

    L^{-1}(x - mu(x')) = L^{-1}(x - x') - h' / 2 = -(r + (h + h') / 2),

with h' = L^T grad log pi(x'). So

    log q(x | x') - log q(x' | x) = (|r|^2 - |r + (h + h') / 2|^2) / 2,

and a step costs 3 products (h, L(h/2 + z), h'), against 7 for `MALA.step`. The positions
stay in the target's coordinates, so `State` is unchanged; only the residuals and the drift
are whitened. Caching h with the state would save one more product per chain per step,
but needs a field that `State` does not have, so it is not prototyped.

The coupled step reflects in whitened coordinates. The whitened difference of the means
is delta = L^{-1}(mu_x - mu_y) = L^{-1}(x - y) + (h_x - h_y) / 2, one product with
L^{-1}; `reflect` then does what `couplings.reflection_maximal` does to z without its
products, and returns z, the reflected z_y and the flags. Then x' = x + L(h_x / 2 + z) and
y' = y + L(h_y / 2 + z_y), copied from x' on identical rows, where the forward residual
of y is r_y = z + delta (since x' = mu_y + L(z + delta)). With h_x, h_y, h'_x and h'_y a
coupled step costs 7 products, against 14 for `MALA.coupled_step` (4 for the two means,
2 in `reflection_maximal`, 4 in each log ratio).

The draws are read in MALA's order: a step reads d normals, then a uniform; a coupled step
d normals, the reflection's uniform, then the shared Metropolis uniform. So both forms,
given the same draws, make the same proposals and decisions up to rounding.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray
from telescopic.kernels import MALA
from telescopic.kernels.base import (
    State,
    acceptance,
    evaluate_pair,
    evaluate_proposal,
    select,
)
from telescopic.target import Target


class WhitenedMALA(MALA):
    """MALA with whitened residuals: the same kernel as `MALA`, in fewer products."""

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        proposal, log_alpha = self.propose(target, state, draws.normal(len(self._ready()[0])))
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def propose(self, target: Target, state: State, z: FloatArray) -> tuple[State, FloatArray]:
        # The proposal of a step from the normals z, and its log acceptance ratio; the
        # experiment calls it to compare the ratios of the two forms.
        chol, _ = self._ready()
        blas = target.vectorize
        drift = whitened_drift(state, chol, blas)
        proposal = evaluate_proposal(target, state.x + matvec(chol, 0.5 * drift + z, blas=blas))
        return proposal, log_alpha(state, proposal, z, drift, chol, blas)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        chol, inverse = self._ready()
        blas = target.vectorize
        drift_x, drift_y = whitened_drift(state_x, chol, blas), whitened_drift(state_y, chol, blas)
        # Infinite drifts, from a gradient that overflowed, can give inf - inf = NaN in
        # delta, which fails the meeting test as it does in reflection_maximal.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = matvec(inverse, state_x.x - state_y.x, blas=blas) + 0.5 * (drift_x - drift_y)
        z, z_y, identical = reflect(delta, draws)
        x = state_x.x + matvec(chol, 0.5 * drift_x + z, blas=blas)
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.where(
                identical[:, None], x, state_y.x + matvec(chol, 0.5 * drift_y + z_y, blas=blas)
            )
            residual_y = np.where(identical[:, None], z + delta, z_y)
        proposal_x, proposal_y = evaluate_pair(partial(evaluate_proposal, target), x, y, identical)
        log_u = draws.log_uniform()
        state_x = select(
            log_u < log_alpha(state_x, proposal_x, z, drift_x, chol, blas), proposal_x, state_x
        )
        state_y = select(
            log_u < log_alpha(state_y, proposal_y, residual_y, drift_y, chol, blas),
            proposal_y,
            state_y,
        )
        return state_x, state_y

    def _with_cov(self, cov: FloatArray) -> WhitenedMALA:
        return WhitenedMALA(self.step_size, cov=cov)


def whitened_drift(state: State, chol: FloatArray, blas: bool) -> FloatArray:
    # h = L^T grad log pi(x), with chol = L = epsilon C.
    with np.errstate(over="ignore", invalid="ignore"):
        return matvec(chol.T, cast(FloatArray, state.grad), blas=blas)


def log_alpha(
    current: State,
    proposal: State,
    residual: FloatArray,
    drift: FloatArray,
    chol: FloatArray,
    blas: bool,
) -> FloatArray:
    # log pi(x') - log pi(x) + (|r|^2 - |r + (h + h')/2|^2) / 2. A rejected proposal
    # (log pi = -inf) carries a zero gradient and may hold non-finite positions; a NaN
    # ratio rejects as -inf does.
    with np.errstate(over="ignore", invalid="ignore"):
        back = residual + 0.5 * (drift + whitened_drift(proposal, chol, blas))
        log_q = 0.5 * (
            np.einsum("ij,ij->i", residual, residual) - np.einsum("ij,ij->i", back, back)
        )
        ratio: FloatArray = proposal.log_prob - current.log_prob + log_q
    return ratio


def reflect(delta: FloatArray, draws: Draws) -> tuple[FloatArray, FloatArray, BoolArray]:
    # The reflection-maximal coupling of N(0, I) and N(-delta, I), shifted: z ~ N(0, I) for
    # x, and z_y = z - 2 (e.z) e, e = delta / |delta|, for y, unless log u < -z.delta -
    # |delta|^2 / 2, where the pair is identical. This is reflection_maximal with C = I
    # and without its products, reading the same draws (d normals, then 1 uniform).
    z = draws.normal(delta.shape[1])
    log_u = draws.log_uniform()
    with np.errstate(invalid="ignore", divide="ignore"):
        squared = np.einsum("ij,ij->i", delta, delta)
        along = np.einsum("ij,ij->i", z, delta)
        identical = log_u < -along - 0.5 * squared
        coefficient = 2.0 * along / squared
    finite = (squared > 0) & (squared < np.inf)
    shift = np.multiply(
        coefficient[:, None], delta, out=np.zeros_like(delta), where=finite[:, None]
    )
    return z, z - shift, identical
