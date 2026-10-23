"""The Metropolis-adjusted Langevin algorithm with the reflection-maximal coupling."""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, Seed
from telescopic.couplings import reflection_maximal
from telescopic.kernels.base import (
    Kernel,
    State,
    acceptance,
    evaluate_pair,
    evaluate_proposal,
    select,
)
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport, dual_averaging

TARGET_ACCEPTANCE = 0.574


class MALA(Kernel):
    r"""The Metropolis-adjusted Langevin algorithm (spec R-KER-MALA).

    The proposal is :math:`x' = x + \tfrac{\epsilon^2}{2}\Sigma\nabla\log\pi(x) + \epsilon Cz`,
    with :math:`CC^\top = \Sigma` (``cov``) and :math:`z \sim \mathcal N(0, I)`, a draw from
    :math:`q(\cdot \mid x)`. It is accepted when
    :math:`\log u < \log\pi(x') - \log\pi(x) + \log q(x \mid x') - \log q(x' \mid x)`; a
    proposal with a log-density of ``-inf`` or a non-finite gradient is rejected. Coupled
    chains draw their proposals from the reflection-maximal coupling of their proposal
    distributions, which share the covariance :math:`\epsilon^2\Sigma`, and share the
    uniform (Jacob, O'Leary & Atchade 2020, section 4.4).

    Parameters
    ----------
    step_size : float, optional
        :math:`\epsilon`. ``None`` leaves it to :meth:`tune`, which runs dual averaging
        toward an acceptance rate of 0.574 from the last states of the pilot chains
        (spec section 5.4).
    cov : array_like, shape (d, d), optional
        :math:`\Sigma`, usually an estimate of the target covariance. ``None`` leaves it to
        :meth:`tune`, which takes the covariance of adaptive Metropolis pilot chains.

    Notes
    -----
    A step uses ``d`` normals and 1 uniform, and a coupled step ``d`` normals and 2
    uniforms. Each evaluates the log-density and its gradient once per chain, once per
    pair when the proposals coincide.
    """

    def __init__(
        self, step_size: float | None = None, *, cov: npt.ArrayLike | None = None
    ) -> None:
        if step_size is not None and not step_size > 0:
            raise ValueError(f"step_size must be positive, not {step_size}")
        self.step_size = step_size
        self.cov: FloatArray | None = None
        self._factors: tuple[FloatArray, FloatArray] | None = None
        if cov is not None:
            self.cov, chol = covariance(cov, "cov")
            if step_size is not None:
                self._factors = step_size * chol, inverse_lower(step_size * chol)

    @property
    def needs_grad(self) -> bool:
        return True

    @property
    def is_tuned(self) -> bool:
        return self._factors is not None

    @property
    def ndim(self) -> int | None:
        return None if self.cov is None else len(self.cov)

    def _tuned(self, target: Target, pilot: PilotRun, rng: Seed) -> tuple[Kernel, TuningReport]:
        cov = pilot.cov if self.cov is None else self.cov
        step_size = self.step_size
        if step_size is None:
            # The optimal scale for a standard Gaussian, 1.65 d^(-1/6) (Roberts & Rosenthal 1998).
            step_size = dual_averaging(
                lambda epsilon: MALA(epsilon, cov=cov),
                target,
                pilot.samples[:, -1],
                rng,
                TARGET_ACCEPTANCE,
                1.65 * len(cov) ** (-1 / 6),
            )
        return MALA(step_size, cov=cov), pilot.report()

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        chol, inverse = self._ready()
        mean = _mean(state, chol)
        proposal = evaluate_proposal(target, mean + matvec(chol, draws.normal(len(chol))))
        log_alpha = _log_alpha(state, proposal, mean, chol, inverse)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        chol, inverse = self._ready()
        mean_x, mean_y = _mean(state_x, chol), _mean(state_y, chol)
        x, y, identical = reflection_maximal(mean_x, mean_y, chol, inverse, draws)
        proposal_x, proposal_y = evaluate_pair(partial(evaluate_proposal, target), x, y, identical)
        log_u = draws.log_uniform()
        state_x = select(
            log_u < _log_alpha(state_x, proposal_x, mean_x, chol, inverse), proposal_x, state_x
        )
        state_y = select(
            log_u < _log_alpha(state_y, proposal_y, mean_y, chol, inverse), proposal_y, state_y
        )
        return state_x, state_y

    def _ready(self) -> tuple[FloatArray, FloatArray]:
        if self._factors is None:
            raise ValueError(
                "MALA needs step_size and cov; give them, or tune the kernel on a pilot run"
            )
        return self._factors

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return f"MALA(step_size={self.step_size}, cov={cov})"


def _mean(state: State, chol: FloatArray) -> FloatArray:
    # The mean of q(. | x): x + (epsilon^2 / 2) Sigma grad, with chol = epsilon C.
    grad = cast(FloatArray, state.grad)
    mean: FloatArray = state.x + 0.5 * matvec(chol, matvec(chol.T, grad))
    return mean


def _log_alpha(
    current: State, proposal: State, forward: FloatArray, chol: FloatArray, inverse: FloatArray
) -> FloatArray:
    # log pi(x') - log pi(x) + log q(x | x') - log q(x' | x); the Gaussian constants cancel.
    # A rejected proposal (log pi = -inf) may hold non-finite positions: no warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        to = matvec(inverse, proposal.x - forward)
        back = matvec(inverse, current.x - _mean(proposal, chol))
        log_q = 0.5 * (np.einsum("ij,ij->i", to, to) - np.einsum("ij,ij->i", back, back))
        log_alpha: FloatArray = proposal.log_prob - current.log_prob + log_q
    return np.where(np.isfinite(proposal.log_prob), log_alpha, -np.inf)
