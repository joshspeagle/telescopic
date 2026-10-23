"""Random-walk Metropolis-Hastings with its couplings."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws, FloatArray
from telescopic.couplings import maximal, reflection_maximal
from telescopic.kernels.base import (
    BoolArray,
    Kernel,
    State,
    equal_rows,
    evaluate,
    evaluate_pair,
    log_ratio,
    select,
)
from telescopic.target import Target


class RandomWalk(Kernel):
    r"""Random-walk Metropolis-Hastings with a Gaussian proposal (spec R-KER-RW).

    The proposal is :math:`x' = x + \lambda L z` with :math:`LL^\top = \Sigma` (``cov``),
    :math:`\lambda` (``scale``) and :math:`z \sim \mathcal N(0, I)`; it is accepted when
    :math:`\log u < \log\pi(x') - \log\pi(x)`.

    Parameters
    ----------
    cov : array_like, shape (d, d), optional
        Shape of the proposal, usually an estimate of the target covariance. ``None``
        leaves the kernel untuned: a sampler tunes it on a pilot run.
    scale : float, optional
        Scale :math:`\lambda`; defaults to :math:`2.38/\sqrt d`.
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
        if scale is not None and not scale > 0:
            raise ValueError(f"scale must be positive, not {scale}")
        self.cov = None if cov is None else _covariance(cov)
        self.scale = scale
        self.coupling = coupling
        self._chol = None if self.cov is None else self._proposal_chol(self.cov, scale)

    @property
    def is_tuned(self) -> bool:
        return self.cov is not None

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, BoolArray]:
        chol = self._ready()
        proposal = evaluate(target, state.x + draws.normal(len(chol)) @ chol.T, False)
        accepted = draws.log_uniform() < log_ratio(proposal.log_prob, state.log_prob)
        return select(accepted, proposal, state), accepted

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State, BoolArray]:
        chol = self._ready()
        if self.coupling == "reflection":
            x, y, identical = reflection_maximal(state_x.x, state_y.x, chol, draws)
        else:
            x, y, identical, _ = maximal(
                [_Gaussian(mean, chol) for mean in state_x.x],
                [_Gaussian(mean, chol) for mean in state_y.x],
                draws,
            )
        proposal_x, proposal_y = evaluate_pair(target, x, y, identical, False)
        log_u = draws.log_uniform()
        state_x = select(
            log_u < log_ratio(proposal_x.log_prob, state_x.log_prob), proposal_x, state_x
        )
        state_y = select(
            log_u < log_ratio(proposal_y.log_prob, state_y.log_prob), proposal_y, state_y
        )
        return state_x, state_y, equal_rows(state_x.x, state_y.x)

    def _ready(self) -> FloatArray:
        if self._chol is None:
            raise ValueError("RandomWalk needs cov; give it, or tune the kernel on a pilot run")
        return self._chol

    @staticmethod
    def _proposal_chol(cov: FloatArray, scale: float | None) -> FloatArray:
        return (2.38 / np.sqrt(len(cov)) if scale is None else scale) * np.linalg.cholesky(cov)

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return f"RandomWalk(cov={cov}, scale={self.scale}, coupling={self.coupling!r})"


class _Gaussian:
    # A Gaussian proposal for the rejection coupling; both members of a pair share chol,
    # so the normalising constants cancel and are left out.
    def __init__(self, mean: FloatArray, chol: FloatArray) -> None:
        self.mean, self.chol = mean, chol

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, x: FloatArray) -> float:
        z = np.linalg.solve(self.chol, x - self.mean)
        return float(-0.5 * z @ z)


def _covariance(cov: npt.ArrayLike) -> FloatArray:
    matrix = np.array(cov, dtype=np.float64, ndmin=2)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cov must be a square matrix, not shape {matrix.shape}")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("cov must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("cov must be positive definite") from None
    return matrix
