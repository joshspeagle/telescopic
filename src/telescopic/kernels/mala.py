"""The Metropolis-adjusted Langevin algorithm with the reflection-maximal coupling."""

from __future__ import annotations

from typing import cast

import numpy.typing as npt

from telescopic._linalg import inverse_lower
from telescopic._types import FloatArray
from telescopic.couplings import Coupling, Reflection
from telescopic.kernels.base import GradientKernel
from telescopic.kernels.metropolis import LangevinProposal, MetropolisHastings, Proposal


# Spec R-KER-MALA and section 5.4.
class MALA(GradientKernel, MetropolisHastings):
    r"""The Metropolis-adjusted Langevin algorithm.

    The proposal is :math:`x' = x + \tfrac{\epsilon^2}{2}\Sigma\nabla\log\pi(x) + \epsilon Cz`,
    with :math:`CC^\top = \Sigma` (``cov``) and :math:`z \sim \mathcal N(0, I)`, a draw from
    :math:`q(\cdot \mid x)`. It is accepted when
    :math:`\log u < \log\pi(x') - \log\pi(x) + \log q(x \mid x') - \log q(x' \mid x)`; a
    proposal with a log-density of ``-inf`` or a non-finite position is rejected. Coupled
    chains draw their proposals from the reflection-maximal coupling of their proposal
    distributions, which share the covariance :math:`\epsilon^2\Sigma`, and share the
    uniform (Jacob, O'Leary & Atchadé 2020, section 4.4): a
    :class:`~telescopic.kernels.MetropolisHastings` with a
    :class:`~telescopic.kernels.LangevinProposal` and
    :class:`~telescopic.couplings.Reflection`.

    Parameters
    ----------
    step_size : float, optional
        :math:`\epsilon`. ``None`` leaves it to :meth:`~telescopic.kernels.Kernel.tune`: dual averaging toward an
        acceptance rate of 0.574 from :math:`\epsilon_0 = 1.65\,d^{-1/6}` (Roberts &
        Rosenthal 1998).
    cov : array_like, shape (d, d), optional
        :math:`\Sigma`, usually an estimate of the target covariance. ``None`` leaves it to
        :meth:`~telescopic.kernels.Kernel.tune`, which takes the covariance of adaptive Metropolis pilot chains.

    Notes
    -----
    A step uses ``d`` normals and 1 uniform, and a coupled step ``d`` normals and 2
    uniforms. Each evaluates the log-density once per chain, once per pair when the
    proposals coincide, and the gradient where the log-density is finite.

    Both compute in whitened coordinates. With :math:`B = \epsilon C` and the whitened
    drift :math:`h = B^\top\nabla\log\pi(x)`, the proposal is :math:`x' = x + B(h/2 + z)`
    and the log ratio of the proposal densities is
    :math:`(\lVert z\rVert^2 - \lVert z + (h + h')/2\rVert^2)/2`, so a step costs 3
    products of a row with a :math:`d \times d` factor and a coupled step 7.

    Examples
    --------
    MALA needs the gradient, as ``grad`` or ``log_prob_and_grad``, which it evaluates
    with every finite log-density, so on a Gaussian the two counts agree:

    >>> from telescopic import examples, kernels, sample
    >>> example = examples.gaussian(2)
    >>> result = sample(
    ...     example.log_prob, example.initial, grad=example.grad,
    ...     kernel=kernels.MALA(), n_replicates=50, vectorize=True, rng=0,
    ... )
    >>> result.n_grad == result.n_log_prob
    True
    """

    target_acceptance = 0.574

    def __init__(
        self, step_size: float | None = None, *, cov: npt.ArrayLike | None = None
    ) -> None:
        super().__init__(step_size, cov)
        self._inverse = None if self._chol is None else inverse_lower(self._chol)
        self._coupling: Coupling = Reflection()
        self._proposal: Proposal | None = self._langevin()

    def with_step_size(self, step_size: float) -> MALA:
        kernel = cast(MALA, super().with_step_size(step_size))
        kernel._proposal = kernel._langevin()
        return kernel

    def _with_cov(self, cov: FloatArray) -> MALA:
        return MALA(self.step_size, cov=cov)

    def _first_step_size(self, d: int) -> float:
        return float(1.65 * d ** (-1 / 6))

    def _langevin(self) -> LangevinProposal | None:
        # The proposal with the factor of its covariance epsilon^2 Sigma, and its inverse,
        # computed once per step size: per step they would cost a third of a step at
        # d = 200.
        if self._chol is None or self._inverse is None or self.step_size is None:
            return None
        return LangevinProposal._trusted(
            self.step_size * self._chol, self._inverse / self.step_size
        )

    def _ready(self) -> tuple[Proposal, Coupling]:
        if self._proposal is None:
            raise ValueError(
                "MALA needs step_size and cov; give them, or tune the kernel on a pilot run"
            )
        return self._proposal, self._coupling

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return f"MALA(step_size={self.step_size}, cov={cov})"
