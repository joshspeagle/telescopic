"""Random-walk Metropolis-Hastings with its couplings."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, Seed, real
from telescopic.couplings import maximal, reflection_maximal
from telescopic.kernels.base import (
    Kernel,
    State,
    acceptance,
    evaluate,
    evaluate_pair,
    log_ratio,
    select,
)
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport


class RandomWalk(Kernel):
    r"""Random-walk Metropolis-Hastings with a Gaussian proposal (spec R-KER-RW).

    The proposal is :math:`x' = x + \lambda C z` with :math:`CC^\top = \Sigma` (``cov``),
    :math:`\lambda` (``scale``) and :math:`z \sim \mathcal N(0, I)`; it is accepted when
    :math:`\log u < \log\pi(x') - \log\pi(x)`.

    Parameters
    ----------
    cov : array_like, shape (d, d), optional
        Shape of the proposal, usually an estimate of the target covariance. ``None``
        leaves the kernel untuned: :meth:`~telescopic.kernels.Kernel.tune` sets it to the covariance of adaptive
        Metropolis pilot chains (spec section 5.3).
    scale : float, optional
        Scale :math:`\lambda`; defaults to :math:`2.38/\sqrt d`, the optimal scale for a
        Gaussian target whose covariance is ``cov`` (Roberts, Gelman & Gilks 1997).
    coupling : {"reflection", "maximal"}
        ``"reflection"`` couples the proposals by the reflection-maximal coupling and
        uses one uniform for both acceptances (Jacob, O'Leary & Atchade 2020; O'Leary
        2021). ``"maximal"`` uses the rejection maximal coupling with independent
        residuals, whose meeting times grow exponentially with the dimension; it is for
        comparison only.

    Notes
    -----
    A step uses ``d`` normals and 1 uniform; a coupled step with the reflection coupling
    uses ``d`` normals and 2 uniforms and at most two log-density evaluations per pair,
    one when the proposals coincide.
    """

    def __init__(
        self,
        cov: npt.ArrayLike | None = None,
        *,
        scale: float | None = None,
        coupling: Literal["reflection", "maximal"] = "reflection",
    ) -> None:
        if coupling not in ("reflection", "maximal"):
            raise ValueError(f"coupling must be 'reflection' or 'maximal', not {coupling!r}")
        self.scale = None if scale is None else real("scale", scale, 0, np.inf)
        self.coupling = coupling
        self.cov: FloatArray | None = None
        self._factors: tuple[FloatArray, FloatArray] | None = None
        if cov is not None:
            self.cov, chol = covariance("cov", cov)
            scaled = (2.38 / np.sqrt(len(chol)) if self.scale is None else self.scale) * chol
            self._factors = scaled, inverse_lower(scaled)

    @property
    def is_tuned(self) -> bool:
        return self.cov is not None

    @property
    def ndim(self) -> int | None:
        return None if self.cov is None else len(self.cov)

    def _tuned(
        self, target: Target, initial: Any, pilot: PilotRun, rng: Seed
    ) -> tuple[Kernel, TuningReport]:
        tuned = type(self)(pilot.cov, scale=self.scale, coupling=self.coupling)
        return tuned, pilot.report()

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        chol, _ = self._ready()
        x = state.x + matvec(chol, draws.normal(len(chol)))
        proposal = evaluate(target, x, state.grad is not None)
        log_alpha = log_ratio(proposal.log_prob, state.log_prob)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        chol, inverse = self._ready()
        if self.coupling == "reflection":
            x, y, identical = reflection_maximal(state_x.x, state_y.x, chol, inverse, draws)
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

    def _ready(self) -> tuple[FloatArray, FloatArray]:
        if self._factors is None:
            raise ValueError("RandomWalk needs cov; give it, or tune the kernel on a pilot run")
        return self._factors

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return f"RandomWalk(cov={cov}, scale={self.scale}, coupling={self.coupling!r})"


class _Gaussian:
    # A Gaussian proposal for the rejection coupling; both members of a pair share chol,
    # so the normalising constants cancel and are left out.
    def __init__(self, mean: FloatArray, chol: FloatArray, inverse: FloatArray) -> None:
        self.mean, self.chol, self.inverse = mean, chol, inverse

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, x: FloatArray) -> float:
        z = self.inverse @ (x - self.mean)
        return float(-0.5 * z @ z)
