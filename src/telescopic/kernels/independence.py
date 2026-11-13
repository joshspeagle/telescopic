"""Independence Metropolis-Hastings and its proposal distributions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, Proposal, Seed, integer, log_densities, vector
from telescopic.kernels.base import Kernel, State, acceptance, evaluate, log_ratio, select
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
    cov : array_like, shape (d, d)
        Scale matrix; the covariance is ``cov * df / (df - 2)`` for ``df > 2``.
    df : int or None
    """

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike, df: int | None = 5) -> None:
        self.mean = vector("mean", mean)
        self.cov, self._chol = covariance("cov", cov, len(self.mean))
        expected = "a positive integer, or None for the Gaussian"
        self.df = None if df is None else integer("df", df, expected, minimum=1)
        self._inverse = inverse_lower(self._chol)
        self._log_det = 2.0 * np.log(np.diag(self._chol)).sum()

    def sample(self, draws: Draws) -> FloatArray:
        """One draw per row of ``draws``, shape ``(n, d)``; uses ``d + df`` normals per row."""
        z = draws.normal(len(self.mean))
        if self.df is not None:
            z = z / np.sqrt((draws.normal(self.df) ** 2).sum(axis=1) / self.df)[:, None]
        return self.mean + matvec(self._chol, z)

    def log_density(self, x: FloatArray) -> FloatArray:
        """Normalized log-density at the rows of ``x``, shape ``(n,)``."""
        d = len(self.mean)
        z = matvec(self._inverse, x - self.mean)
        log_density: FloatArray
        if self.df is None:
            log_density = -0.5 * ((z**2).sum(axis=1) + self._log_det + d * math.log(2 * math.pi))
            return log_density
        nu = self.df
        # A squared distance beyond the float64 range is inf, or NaN where whitening
        # overflows to inf - inf, but the log-density is finite there.
        with np.errstate(over="ignore"):
            squared = (z**2).sum(axis=1)
        log_kernel = np.log1p(squared / nu)
        if not np.isfinite(squared).all():
            # log1p(s / nu) = log s - log nu to within nu / s < 1e-307, and log s is
            # computed from the rows scaled by their largest coordinate.
            far = ~np.isfinite(squared)
            centered = x[far] - self.mean
            scale = np.abs(centered).max(axis=1)
            whitened = matvec(self._inverse, centered / scale[:, None])
            log_kernel[far] = 2 * np.log(scale) + np.log((whitened**2).sum(axis=1)) - math.log(nu)
        log_density = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - 0.5 * (d * math.log(nu * math.pi) + self._log_det)
            - 0.5 * (nu + d) * log_kernel
        )
        return log_density

    def __repr__(self) -> str:
        return f"MultivariateT(mean={self.mean.tolist()}, df={self.df})"


class Independence(Kernel):
    r"""Independence Metropolis-Hastings (spec R-KER-IMH).

    A proposal :math:`x' \sim q` is accepted when
    :math:`\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]`. The weight
    :math:`\pi/q` is 0 wherever :math:`\pi` is 0, even where :math:`q` is 0 too, so a chain
    at zero density accepts any proposal of finite density, as random-walk Metropolis
    does. A proposal where :math:`q` is 0 is rejected, even where :math:`\pi` is not:
    ``sample`` drew a point that ``log_density`` gives no mass, such as a draw clipped onto
    the edge of an open support. The coupled step shares the proposal and :math:`u`
    between the chains, which is a maximal kernel coupling (O'Leary, Wang & Jacob 2021):
    a pair meets when both chains accept.

    Parameters
    ----------
    proposal : object, optional
        Any object with ``sample(draws)`` and ``log_density(x)``, the normalized density of
        what ``sample`` draws, positive wherever ``sample`` can land. A NaN or ``+inf``
        log-density raises ``ValueError`` with the point. Its ``mean``, if it has one that
        is not a method, fixes the dimension. ``None`` leaves the kernel untuned:
        :meth:`~telescopic.kernels.Kernel.tune` fits a Student-:math:`t_5` located at the
        mean of adaptive Metropolis pilot chains, with their covariance as its scale
        matrix (spec section 5.6, D-25). Its report gives the largest weight
        :math:`\pi/q` of the pilot's draws relative to their median.

    Notes
    -----
    A step and a coupled step use what the proposal's ``sample`` uses (``d + df``
    normals for :class:`MultivariateT`, ``d`` with ``df=None``) and 1 uniform.

    The proposal must be positive wherever the target is. Chains never move to where
    :math:`q` is 0, so a proposal that misses part of the target's support never reaches
    that part, and the estimates are biased from any start, without a warning: on
    :math:`\mathcal N(0, 1)` with a proposal uniform on :math:`(-2, 2)`, 500 replicates
    estimate :math:`E[x^2] = 1` as 0.778 with a standard error of 0.0093. A chain started
    where the target is positive and the proposal is 0 rejects every proposal.

    Meeting times are geometric only when the weights :math:`\pi/q` are bounded; a
    proposal with lighter tails than the target gives heavy-tailed meeting times.

    A tuned proposal covers only the modes that the pilot chains reached. On a target with
    well-separated modes, the chains can settle in one mode, and pairs then meet quickly
    in it: the estimates can be confidently wrong, and the pilot's agreeing chains raise
    no warning. Give ``proposal`` explicitly to cover modes that are known.
    """

    def __init__(self, proposal: Proposal | None = None) -> None:
        if proposal is not None and not (
            hasattr(proposal, "sample") and hasattr(proposal, "log_density")
        ):
            raise TypeError(
                "proposal must have sample(draws) and log_density(x), as MultivariateT "
                f"does, not {type(proposal).__name__}"
            )
        self.proposal = proposal

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

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        q = self._ready()
        proposal = evaluate(target, q.sample(draws), state.grad is not None)
        new, old = _log_weights(q, proposal, state)
        log_alpha = log_ratio(new, old)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposal, state), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        q = self._ready()
        proposal = evaluate(target, q.sample(draws), state_x.grad is not None)
        log_u = draws.log_uniform()
        new, old_x, old_y = _log_weights(q, proposal, state_x, state_y)
        state_x = select(log_u < log_ratio(new, old_x), proposal, state_x)
        state_y = select(log_u < log_ratio(new, old_y), proposal, state_y)
        return state_x, state_y

    def _ready(self) -> Proposal:
        if self.proposal is None:
            raise ValueError("Independence needs a proposal; give one, or tune the kernel")
        return self.proposal

    def __repr__(self) -> str:
        return f"Independence(proposal={self.proposal!r})"


def _log_weights(q: Proposal, proposal: State, *states: State) -> list[FloatArray]:
    # log pi - log q at the proposal's positions, then at each state's, from one call of
    # q.log_density, whose rows do not depend on each other (spec R-KER-6). A NaN log q
    # would make every move to its point a rejection, and so bias the chain without an
    # error.
    x = np.concatenate([proposal.x, *(state.x for state in states)])
    log_q = np.split(log_densities("proposal log_density", q.log_density(x), x), 1 + len(states))
    # A proposal where log q = -inf has weight -inf, so it is rejected: sample drew a point
    # that log_density gives no mass, such as a draw clipped onto the edge of an open
    # support, and a weight of +inf would accept it and never leave it.
    new = proposal.log_prob - np.where(log_q[0] == -np.inf, np.inf, log_q[0])
    # Where log pi = -inf a state's weight is -inf even where log q = -inf too: -inf - -inf
    # would be NaN, which rejects every proposal, so such a chain would never move.
    return [new] + [
        state.log_prob - np.where(state.log_prob == -np.inf, 0.0, part)
        for state, part in zip(states, log_q[1:], strict=True)
    ]
