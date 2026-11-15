"""Random-walk Metropolis-Hastings with its couplings."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance, inverse_lower
from telescopic._types import FloatArray, Seed, real
from telescopic.couplings import Coupling, Maximal, Reflection
from telescopic.kernels.base import Kernel
from telescopic.kernels.metropolis import MetropolisHastings, Proposal, RandomWalkProposal
from telescopic.target import Target
from telescopic.tuning import OPTIMAL_SCALE, PilotRun, TuningReport


# Spec R-KER-RW; the tuning of cov is that of spec section 5.3.
class RandomWalk(MetropolisHastings):
    r"""Random-walk Metropolis-Hastings with a Gaussian proposal.

    The proposal is :math:`x' = x + \lambda C z` with :math:`CC^\top = \Sigma` (``cov``),
    :math:`\lambda` (``scale``) and :math:`z \sim \mathcal N(0, I)`; it is accepted when
    :math:`\log u < \log\pi(x') - \log\pi(x)`. It is a
    :class:`~telescopic.kernels.MetropolisHastings` with a
    :class:`~telescopic.kernels.RandomWalkProposal`.

    Parameters
    ----------
    cov : array_like, shape (d, d), optional
        Shape of the proposal, usually an estimate of the target covariance. ``None``
        leaves the kernel untuned: :meth:`~telescopic.kernels.Kernel.tune` sets it to the covariance of adaptive
        Metropolis pilot chains.
    scale : float, optional
        Scale :math:`\lambda`; defaults to :math:`2.38/\sqrt d`, the optimal scale for a
        Gaussian target whose covariance is ``cov`` (Roberts, Gelman & Gilks 1997).
    coupling : {"reflection", "maximal"}, Reflection or Maximal
        ``"reflection"``, or :class:`~telescopic.couplings.Reflection`, couples the
        proposals by the reflection-maximal coupling and uses one uniform for both
        acceptances (Jacob, O'Leary & Atchadé 2020; O'Leary 2021). ``"maximal"``, or
        :class:`~telescopic.couplings.Maximal`, uses the rejection maximal coupling with
        independent residuals, whose meeting times grow exponentially with the
        dimension; it is for comparison only. The attribute ``coupling`` returns what
        was given, and cannot be reassigned.

    Notes
    -----
    A step uses ``d`` normals and 1 uniform; a coupled step with the reflection coupling
    uses ``d`` normals and 2 uniforms and at most two log-density evaluations per pair,
    one when the proposals coincide. With the maximal coupling, a coupled step draws
    proposals of ``d`` normals and 1 uniform each from each pair's own generator
    (:meth:`~telescopic.kernels.Draws.generator`), two per pair on average for a pair
    apart and one for a pair that has met, then 1 uniform for both acceptances.

    Examples
    --------
    A kernel left without ``cov`` is tuned by the sampler, which keeps the tuned copy in
    the result and leaves the kernel passed to it as it was:

    >>> from telescopic import examples, kernels, sample
    >>> example = examples.gaussian(2)
    >>> kernel = kernels.RandomWalk()
    >>> result = sample(
    ...     example.log_prob, example.initial, kernel=kernel,
    ...     n_replicates=50, vectorize=True, rng=0,
    ... )
    >>> kernel.is_tuned, result.kernel.is_tuned
    (False, True)
    """

    def __init__(
        self,
        cov: npt.ArrayLike | None = None,
        *,
        scale: float | None = None,
        coupling: Literal["reflection", "maximal"] | Reflection | Maximal = "reflection",
    ) -> None:
        if isinstance(coupling, Reflection | Maximal):
            self._coupling: Coupling = coupling
        elif isinstance(coupling, str) and coupling in ("reflection", "maximal"):
            self._coupling = Reflection() if coupling == "reflection" else Maximal()
        else:
            raise ValueError(
                "coupling must be 'reflection', 'maximal', Reflection() or Maximal(), not "
                f"{coupling!r}"
            )
        self.scale = None if scale is None else real("scale", scale, 0, np.inf)
        self._given_coupling = coupling
        self.cov: FloatArray | None = None
        self._proposal: Proposal | None = None
        if cov is not None:
            self.cov, chol = covariance("cov", cov)
            scale = OPTIMAL_SCALE / np.sqrt(len(chol)) if self.scale is None else self.scale
            scaled = scale * chol
            self._proposal = RandomWalkProposal._trusted(scaled, inverse_lower(scaled))

    @property
    def coupling(self) -> Literal["reflection", "maximal"] | Reflection | Maximal:
        """The coupling as it was given: a name or a coupling object.

        Returns
        -------
        str, Reflection or Maximal
        """
        return self._given_coupling

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

    def _ready(self) -> tuple[Proposal, Coupling]:
        if self._proposal is None:
            raise ValueError("RandomWalk needs cov; give it, or tune the kernel on a pilot run")
        return self._proposal, self._coupling

    def __repr__(self) -> str:
        cov = "None" if self.cov is None else f"<{len(self.cov)}x{len(self.cov)}>"
        return f"RandomWalk(cov={cov}, scale={self.scale}, coupling={self.coupling!r})"
