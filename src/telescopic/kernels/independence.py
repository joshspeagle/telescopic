"""Independence Metropolis-Hastings and its proposal distributions."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from telescopic._linalg import covariance, inverse_lower, matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray
from telescopic.kernels.base import Kernel, State, evaluate, log_ratio, select
from telescopic.target import Target


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
        self.mean = np.array(mean, dtype=np.float64, ndmin=1)
        self.cov, self._chol = covariance(cov, "cov", len(self.mean))
        if df is not None and (int(df) != df or df < 1):
            raise ValueError(f"df must be a positive integer or None, not {df}")
        self.df = None if df is None else int(df)
        self._inverse = inverse_lower(self._chol)
        self._log_det = 2.0 * np.log(np.diag(self._chol)).sum()

    def sample(self, draws: Draws) -> FloatArray:
        """One draw per row of ``draws``, shape ``(n, d)``; uses ``d + df`` normals per row."""
        z = draws.normal(len(self.mean))
        if self.df is not None:
            z = z / np.sqrt((draws.normal(self.df) ** 2).sum(axis=1) / self.df)[:, None]
        return self.mean + matvec(self._chol, z)

    def log_density(self, x: FloatArray) -> FloatArray:
        """Normalised log-density at the rows of ``x``, shape ``(n,)``."""
        d = len(self.mean)
        z = matvec(self._inverse, x - self.mean)
        squared = (z**2).sum(axis=1)
        log_density: FloatArray
        if self.df is None:
            log_density = -0.5 * (squared + self._log_det + d * math.log(2 * math.pi))
        else:
            nu = self.df
            log_density = (
                gammaln((nu + d) / 2)
                - gammaln(nu / 2)
                - 0.5 * (d * math.log(nu * math.pi) + self._log_det)
                - 0.5 * (nu + d) * np.log1p(squared / nu)
            )
        return log_density

    def __repr__(self) -> str:
        return f"MultivariateT(mean={self.mean.tolist()}, df={self.df})"


class Independence(Kernel):
    r"""Independence Metropolis-Hastings (spec R-KER-IMH).

    A proposal :math:`x' \sim q` is accepted when
    :math:`\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]`. The coupled
    step shares the proposal and :math:`u` between the chains, which is a maximal kernel
    coupling (O'Leary, Wang & Jacob 2021): a pair meets when both chains accept.

    Parameters
    ----------
    proposal : MultivariateT, optional
        Any object with ``sample(draws)`` and ``log_density(x)``. ``None`` leaves the
        kernel untuned: a sampler fits a Student-:math:`t` proposal on a pilot run.

    Notes
    -----
    A step and a coupled step use what the proposal's ``sample`` uses (``d + df``
    normals for :class:`MultivariateT`, ``d`` with ``df=None``) and 1 uniform.

    Meeting times are geometric only when the weights :math:`\pi/q` are bounded; a
    proposal with lighter tails than the target gives heavy-tailed meeting times.
    """

    def __init__(self, proposal: MultivariateT | None = None) -> None:
        self.proposal = proposal

    @property
    def is_tuned(self) -> bool:
        return self.proposal is not None

    @property
    def ndim(self) -> int | None:
        return None if self.proposal is None else len(self.proposal.mean)

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, BoolArray]:
        q = self._ready()
        proposal = evaluate(target, q.sample(draws), state.grad is not None)
        new, old = _log_weights(q, proposal, state)
        accepted = draws.log_uniform() < log_ratio(new, old)
        return select(accepted, proposal, state), accepted

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

    def _ready(self) -> MultivariateT:
        if self.proposal is None:
            raise ValueError("Independence needs a proposal; give one, or tune the kernel")
        return self.proposal

    def __repr__(self) -> str:
        return f"Independence(proposal={self.proposal!r})"


def _log_weights(q: MultivariateT, *states: State) -> list[FloatArray]:
    # log pi - log q at each state's positions, from one call of q.log_density, whose rows
    # do not depend on each other (spec R-KER-6).
    log_q = np.split(q.log_density(np.concatenate([state.x for state in states])), len(states))
    return [state.log_prob - part for state, part in zip(states, log_q, strict=True)]
