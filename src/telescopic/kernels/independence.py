"""Independence Metropolis-Hastings and its proposal distributions."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, ProposalDistribution, Seed, integer, vector
from telescopic.couplings import Coupling, Shared
from telescopic.kernels.base import Kernel
from telescopic.kernels.metropolis import IndependenceProposal, MetropolisHastings, Proposal
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport


class MultivariateT:
    r"""Multivariate Student-:math:`t` proposal with location ``mean``, scale ``cov``, ``df``.

    Draws use only normals: :math:`x = \mu + Cz/\sqrt{w/\nu}` with :math:`w` the sum of
    :math:`\nu` squared normals, so ``df`` is a positive integer. ``df=None`` gives the
    Gaussian :math:`\mathcal N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like, shape (d,)
        The location :math:`\mu`.
    cov : array_like, shape (d, d)
        Scale matrix; the covariance is ``cov * df / (df - 2)`` for ``df > 2``.
    df : int or None
        The degrees of freedom :math:`\nu`, a positive integer, or ``None``.
    """

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike, df: int | None = 5) -> None:
        self.mean = vector("mean", mean)
        self.cov, self._chol = covariance("cov", cov, len(self.mean))
        expected = "a positive integer, or None for the Gaussian"
        self.df = None if df is None else integer("df", df, expected, minimum=1)
        self._inverse = inverse_lower(self._chol)
        self._log_det = 2.0 * np.log(np.diag(self._chol)).sum()
        if self.df is not None:
            # The Student-t normalizing constant, grouped as log_density adds it.
            d, nu = len(self.mean), self.df
            self._log_constant: float = (
                gammaln((nu + d) / 2)
                - gammaln(nu / 2)
                - 0.5 * (d * math.log(nu * math.pi) + self._log_det)
            )

    def sample(self, draws: Draws) -> FloatArray:
        """One draw per row of ``draws``; uses ``d + df`` normals per row.

        Parameters
        ----------
        draws : Draws
            The random numbers of ``n`` chains.

        Returns
        -------
        ndarray, shape (n, d)
        """
        z = draws.normal(len(self.mean))
        if self.df is not None:
            z = z / np.sqrt((draws.normal(self.df) ** 2).sum(axis=1) / self.df)[:, None]
        return self.mean + matvec(self._chol, z, blas=False)

    def log_density(self, x: FloatArray) -> FloatArray:
        """Normalized log-density at the rows of ``x``.

        Parameters
        ----------
        x : ndarray, shape (n, d)
            The points.

        Returns
        -------
        ndarray, shape (n,)
        """
        d = len(self.mean)
        centered = x - self.mean
        z = matvec(self._inverse, centered, blas=False)
        log_density: FloatArray
        if self.df is None:
            log_density = -0.5 * ((z**2).sum(axis=1) + self._log_det + d * math.log(2 * math.pi))
            return log_density
        nu = self.df
        # A squared distance beyond the float64 range is inf, or NaN where whitening
        # overflows to inf - inf, but the log-density is finite there. The sum of the
        # squared distances is finite only if each is.
        with np.errstate(over="ignore"):
            squared = (z**2).sum(axis=1)
            near = math.isfinite(squared.sum())
        log_kernel = np.log1p(squared / nu)
        if not near:
            # log1p(s / nu) = log s - log nu to within nu / s < 1e-307, and log s is
            # computed from the rows scaled by their largest coordinate. A row that is
            # not finite keeps its value.
            far = ~np.isfinite(squared) & np.isfinite(centered).all(axis=1)
            scale = np.abs(centered[far]).max(axis=1)
            whitened = matvec(self._inverse, centered[far] / scale[:, None], blas=False)
            log_kernel[far] = 2 * np.log(scale) + np.log((whitened**2).sum(axis=1)) - math.log(nu)
        log_density = self._log_constant - 0.5 * (nu + d) * log_kernel
        return log_density

    def __repr__(self) -> str:
        return f"MultivariateT(mean={self.mean.tolist()}, df={self.df})"


# Spec R-KER-IMH; the tuned proposal is that of spec section 5.6 and D-25.
class Independence(MetropolisHastings):
    r"""Independence Metropolis-Hastings.

    A proposal :math:`x' \sim q` is accepted when
    :math:`\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]`. The weight
    :math:`\pi/q` is 0 wherever :math:`\pi` is 0, even where :math:`q` is 0 too, so a chain
    at zero density accepts any proposal of finite density, as random-walk Metropolis
    does. A proposal where :math:`q` is 0 is rejected, even where :math:`\pi` is not:
    ``sample`` drew a point that ``log_density`` gives no mass, such as a draw clipped onto
    the edge of an open support. The coupled step shares the proposal and :math:`u`
    between the chains, which is a maximal kernel coupling (O'Leary, Wang & Jacob 2021):
    a pair meets when both chains accept. It is a
    :class:`~telescopic.kernels.MetropolisHastings` with an
    :class:`~telescopic.kernels.IndependenceProposal` and
    :class:`~telescopic.couplings.Shared`.

    Parameters
    ----------
    proposal : object, optional
        Any object with ``sample(draws)`` and ``log_density(x)``, the normalized density of
        what ``sample`` draws, positive wherever ``sample`` can land. A NaN or ``+inf``
        log-density raises ``ValueError`` with the point. Its ``mean``, if it has one that
        is not a method, fixes the dimension. ``None`` leaves the kernel untuned:
        :meth:`~telescopic.kernels.Kernel.tune` fits a Student-:math:`t_5` located at the
        mean of adaptive Metropolis pilot chains, with their covariance as its scale
        matrix. Its report gives the largest weight
        :math:`\pi/q` of the pilot's draws relative to their median.

    Notes
    -----
    A step and a coupled step use what the proposal's ``sample`` uses (``d + df``
    normals for :class:`MultivariateT`, ``d`` with ``df=None``) and 1 uniform.

    The proposal must be positive wherever the target is. The kernel never moves a chain
    to where :math:`q` is 0, so, used alone, a proposal that misses part of the target's
    support never reaches that part, and the estimates are biased from any start, without
    a warning. A chain at a point where the target is positive and the proposal is 0
    rejects every proposal.

    Meeting times are geometric only when the weights :math:`\pi/q` are bounded; a
    proposal with lighter tails than the target gives heavy-tailed meeting times. A run
    then usually gives too little mass to the region where :math:`\pi/q` is large, with
    error bars that are too small and often without a warning.

    A tuned proposal covers only the modes that the pilot chains reached. On a target with
    well-separated modes, the chains can settle in one mode, and pairs then meet quickly
    in it: the estimates can be confidently wrong, and the pilot's agreeing chains raise
    no warning. Give ``proposal`` explicitly to cover modes that are known.

    Examples
    --------
    A proposal given explicitly, here a Student-:math:`t` whose scale matrix is twice the
    target's covariance, leaves nothing to tune, so the run has no tuning report:

    >>> from telescopic import examples, kernels, sample
    >>> example = examples.gaussian(2)
    >>> proposal = kernels.MultivariateT(example.mean, 2 * example.cov)
    >>> kernel = kernels.Independence(proposal)
    >>> result = sample(
    ...     example.log_prob, example.initial, kernel=kernel,
    ...     n_replicates=50, vectorize=True, rng=0,
    ... )
    >>> result.tuning is None
    True
    """

    def __init__(self, proposal: ProposalDistribution | None = None) -> None:
        if proposal is not None and not (
            hasattr(proposal, "sample") and hasattr(proposal, "log_density")
        ):
            raise TypeError(
                "proposal must have sample(draws) and log_density(x), as MultivariateT "
                f"does, not {type(proposal).__name__}"
            )
        self._coupling: Coupling = Shared()
        self._proposal: Proposal | None = (
            None if proposal is None else IndependenceProposal(proposal)
        )

    @property
    def proposal(self) -> ProposalDistribution | None:
        """The proposal distribution :math:`q` as it was given, or ``None`` while untuned.

        It is read-only: tuning builds a new kernel.

        Returns
        -------
        object or None
        """
        if self._proposal is None:
            return None
        return cast(IndependenceProposal, self._proposal).distribution

    @property
    def is_tuned(self) -> bool:
        return self.proposal is not None

    @property
    def ndim(self) -> int | None:
        mean = getattr(self.proposal, "mean", None)
        return None if mean is None or callable(mean) else int(np.size(mean))

    def _tuned(
        self, target: Target, initial: Any, pilot: PilotRun, rng: Seed
    ) -> tuple[Kernel, TuningReport]:
        proposal = MultivariateT(pilot.mean, pilot.cov, df=5)  # D-25's t_5, whatever the default
        draws = pilot.samples.reshape(-1, pilot.samples.shape[2])
        log_weights = pilot.log_prob.ravel() - proposal.log_density(draws)
        ratio = float(np.exp(np.max(log_weights) - np.median(log_weights)))
        return Independence(proposal), pilot.report(max_weight_ratio=ratio)

    def _ready(self) -> tuple[Proposal, Coupling]:
        if self._proposal is None:
            raise ValueError("Independence needs a proposal; give one, or tune the kernel")
        return self._proposal, self._coupling

    def __repr__(self) -> str:
        return f"Independence(proposal={self.proposal!r})"
