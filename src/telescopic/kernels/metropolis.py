"""Metropolis-Hastings kernels built from a proposal and a coupling of that proposal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import FloatArray, ProposalDistribution, instance, log_densities, real_array
from telescopic.couplings import CoupledProposals, Coupling, _reflection, reflection_maximal
from telescopic.kernels.base import (
    Kernel,
    State,
    acceptance,
    evaluate,
    evaluate_pair,
    evaluate_proposal,
    log_ratio,
    select,
)
from telescopic.target import Target


class Proposal(ABC):
    """A Metropolis-Hastings proposal for a batch of chains: its draw, its evaluation and its log acceptance ratio.

    A subclass implements :meth:`sample` and :meth:`log_alpha`, and :meth:`evaluate` when
    its proposals' states need other fields than the current states'. A coupled step
    calls :meth:`log_alpha` once for each chain.
    """

    @property
    def needs_grad(self) -> bool:
        """Whether the proposal needs the gradient of the log-density at the current states.

        Returns
        -------
        bool
        """
        return False

    @property
    def ndim(self) -> int | None:
        """The dimension that the proposal fixes, or ``None``.

        Returns
        -------
        int or None
        """
        return None

    def init_extra(self, state: State) -> dict[str, npt.NDArray[Any]]:
        """The initial values of the proposal's own fields, as for :meth:`telescopic.kernels.Kernel.init_extra`; none by default.

        Parameters
        ----------
        state : State
            The chains, ``n`` of them.

        Returns
        -------
        dict of str to ndarray, each of shape (n, ...)
        """
        return {}

    @abstractmethod
    def sample(self, target: Target, state: State, draws: Draws) -> tuple[FloatArray, Any]:
        r"""Draw a proposal :math:`x' \sim q(\cdot \mid x)` for each chain.

        Parameters
        ----------
        target : Target
            The target that the chains sample.
        state : State
            The ``n`` chains.
        draws : Draws
            Random numbers for the ``n`` chains.

        Returns
        -------
        x : ndarray, shape (n, d)
            The proposals.
        forward
            What :meth:`log_alpha` needs of the draw, or ``None``.
        """

    def evaluate(self, target: Target, x: FloatArray, like: State) -> State:
        """The states at the proposals ``x``, with the fields of ``like``.

        Parameters
        ----------
        target : Target
            The target to evaluate.
        x : ndarray, shape (n, d)
            The proposals.
        like : State
            The current states, whose fields the proposals' states carry: gradients if
            they carry them.

        Returns
        -------
        State
        """
        return evaluate(target, x, like.grad is not None)

    @abstractmethod
    def log_alpha(
        self, target: Target, current: State, proposed: State, forward: Any
    ) -> FloatArray:
        r"""The log acceptance ratios :math:`\log\pi(x') - \log\pi(x) + \log q(x \mid x') - \log q(x' \mid x)`.

        Parameters
        ----------
        target : Target
            The target that the chains sample.
        current, proposed : State
            The current states and those at the proposals.
        forward
            What :meth:`sample` returned with the proposals.

        Returns
        -------
        ndarray, shape (n,)
            NaN, from ``-inf - -inf``, is a rejection.
        """

    def _coupled_log_alpha(
        self,
        target: Target,
        state_x: State,
        state_y: State,
        proposed_x: State,
        proposed_y: State,
        forward_x: Any,
        forward_y: Any,
    ) -> tuple[FloatArray, FloatArray]:
        # The log acceptance ratios of both chains of a coupled step. Internal to the
        # shipped proposals: IndependenceProposal overrides it to share one call of its
        # density between the chains.
        return (
            self.log_alpha(target, state_x, proposed_x, forward_x),
            self.log_alpha(target, state_y, proposed_y, forward_y),
        )


class _GaussianProposal(Proposal):
    # The factor B of the covariance BB^T and its inverse, which the random-walk and the
    # Langevin proposals share.
    def __init__(self, factor: npt.ArrayLike, inverse: npt.ArrayLike) -> None:
        given = factor, inverse
        factor = real_array("factor", factor, "real numbers of shape (d, d)", ndmin=2)
        inverse = real_array("inverse", inverse, "real numbers of shape (d, d)", ndmin=2)
        if factor.ndim != 2 or factor.shape[0] != factor.shape[1] or not len(factor):
            raise ValueError(f"factor must be a square matrix, not shape {factor.shape}")
        if inverse.shape != factor.shape:
            raise ValueError(
                f"inverse must have the shape {factor.shape} of factor, not {inverse.shape}"
            )
        if not (np.all(np.isfinite(factor)) and np.all(np.isfinite(inverse))):
            raise ValueError("factor and inverse must be finite")
        # The residual of a computed inverse is within d u cond(B) of 0 (Higham 2002,
        # §14.1); a wrong inverse would bias every coupled step without an error.
        residual = float(np.abs(inverse @ factor - np.eye(len(factor))).max())
        scale = np.linalg.norm(inverse, np.inf) * np.linalg.norm(factor, np.inf)
        if not residual <= 1e-8 * scale:
            raise ValueError(
                f"inverse must be the inverse of factor, but inverse @ factor differs from "
                f"the identity by {residual:.3g}"
            )
        # Copies in the memory order given (order="K"), on which einsum's order of summation
        # and so the last bits of the products depend; real_array's copies are C-ordered.
        # A copy, so that a caller who changes their array does not change the kernel.
        self.factor, self.inverse = (
            np.array(array, dtype=np.float64, order="K") for array in given
        )

    @property
    def ndim(self) -> int:
        return len(self.factor)

    @classmethod
    def _trusted(cls, factor: FloatArray, inverse: FloatArray) -> Self:
        # For the kernels, whose inverse comes from inverse_lower: no check and no copy,
        # since MALA builds a proposal at each step size of dual averaging.
        proposal = cls.__new__(cls)
        proposal.factor, proposal.inverse = factor, inverse
        return proposal

    def __repr__(self) -> str:
        d = len(self.factor)
        return f"{type(self).__name__}(factor=<{d}x{d}>)"


# Spec R-KER-RW.
class RandomWalkProposal(_GaussianProposal):
    r"""The Gaussian random-walk proposal :math:`x' = x + Bz`, :math:`z \sim \mathcal N(0, I)`.

    Its density is symmetric, so the log acceptance ratio is
    :math:`\log\pi(x') - \log\pi(x)`.

    Parameters
    ----------
    factor : array_like, shape (d, d)
        :math:`B`, a factor of the proposal covariance :math:`BB^\top`, such as its
        Cholesky factor.
    inverse : array_like, shape (d, d)
        :math:`B^{-1}`, checked against ``factor`` to within rounding; a matrix that is
        not its inverse raises ``ValueError``.
    """

    def sample(self, target: Target, state: State, draws: Draws) -> tuple[FloatArray, Any]:
        normals = draws.normal(len(self.factor))
        return state.x + matvec(self.factor, normals, blas=target.vectorize), None

    def log_alpha(
        self, target: Target, current: State, proposed: State, forward: Any
    ) -> FloatArray:
        return log_ratio(proposed.log_prob, current.log_prob)

    def reflect(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> CoupledProposals:
        x, y, identical = reflection_maximal(
            state_x.x, state_y.x, self.factor, self.inverse, draws, blas=target.vectorize
        )
        return CoupledProposals(x, y, identical, None, None)


# Spec R-KER-MALA and section 5.4.
class LangevinProposal(_GaussianProposal):
    r"""The Langevin proposal :math:`x' = x + B(h/2 + z)`, with :math:`h = B^\top\nabla\log\pi(x)`.

    With :math:`z \sim \mathcal N(0, I)`, it is a draw from
    :math:`q(\cdot \mid x) = \mathcal N(x + BB^\top\nabla\log\pi(x)/2, BB^\top)`, and the
    log ratio of the proposal densities is
    :math:`(\lVert z\rVert^2 - \lVert z + (h + h')/2\rVert^2)/2`. A proposal with a
    non-finite position is not evaluated: its log-density is ``-inf``, a rejection.

    Parameters
    ----------
    factor : array_like, shape (d, d)
        :math:`B`, a factor of the proposal covariance :math:`BB^\top`, such as its
        Cholesky factor.
    inverse : array_like, shape (d, d)
        :math:`B^{-1}`, checked against ``factor`` to within rounding; a matrix that is
        not its inverse raises ``ValueError``.
    """

    @property
    def needs_grad(self) -> bool:
        return True

    def sample(self, target: Target, state: State, draws: Draws) -> tuple[FloatArray, Any]:
        blas = target.vectorize
        drift = _drift(state, self.factor, blas)
        z = draws.normal(len(self.factor))
        return state.x + matvec(self.factor, 0.5 * drift + z, blas=blas), (z, drift)

    def evaluate(self, target: Target, x: FloatArray, like: State) -> State:
        return evaluate_proposal(target, x)

    def log_alpha(
        self, target: Target, current: State, proposed: State, forward: Any
    ) -> FloatArray:
        residual, drift = forward
        return _log_alpha(current, proposed, residual, drift, self.factor, target.vectorize)

    def reflect(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> CoupledProposals:
        blas = target.vectorize
        drift_x, drift_y = _drift(state_x, self.factor, blas), _drift(state_y, self.factor, blas)
        # The whitened difference of the proposals' means, B^{-1}(mu_x - mu_y), which is
        # also the direction of the reflection in whitened coordinates. Infinite drifts,
        # from a gradient that overflowed, can give inf - inf = NaN, which fails the
        # meeting test as in reflection_maximal.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = matvec(self.inverse, state_x.x - state_y.x, blas=blas) + 0.5 * (
                drift_x - drift_y
            )
        z, identical, shift = _reflection(delta, draws, delta)
        # y's whitened residual: the reflected z, or z + delta where the pair is identical,
        # since there y' = x' = mu_y + B(z + delta).
        residual_y = np.where(identical[:, None], z + delta, z - shift)
        x = state_x.x + matvec(self.factor, 0.5 * drift_x + z, blas=blas)
        y = np.where(
            identical[:, None],
            x,
            state_y.x + matvec(self.factor, 0.5 * drift_y + residual_y, blas=blas),
        )
        return CoupledProposals(x, y, identical, (z, drift_x), (residual_y, drift_y))


# Spec R-KER-IMH.
class IndependenceProposal(Proposal):
    r"""The independence proposal :math:`x' \sim q`, whatever the current state.

    The log acceptance ratio is
    :math:`[\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]`, with a weight of
    ``-inf`` where :math:`\pi` is 0, and a proposal where :math:`q` is 0 rejected. A
    coupled step evaluates :math:`\log q` in one call, at the proposal and at both
    chains' states.

    Parameters
    ----------
    distribution : object
        An object with ``sample(draws)`` and a normalized ``log_density(x)``, such as
        :class:`~telescopic.kernels.MultivariateT`; one without them raises
        ``TypeError``. Its ``mean``, if it has one that is not a method, fixes the
        dimension.
    """

    def __init__(self, distribution: ProposalDistribution) -> None:
        if not (hasattr(distribution, "sample") and hasattr(distribution, "log_density")):
            raise TypeError(
                "the proposal distribution must have sample(draws) and log_density(x), as "
                f"MultivariateT does, not {type(distribution).__name__}"
            )
        self.distribution = distribution

    @property
    def ndim(self) -> int | None:
        # The size of the distribution's mean, if it has one that is not a method.
        mean = getattr(self.distribution, "mean", None)
        return None if mean is None or callable(mean) else int(np.size(mean))

    def sample(self, target: Target, state: State, draws: Draws) -> tuple[FloatArray, Any]:
        return self.distribution.sample(draws), None

    def log_alpha(
        self, target: Target, current: State, proposed: State, forward: Any
    ) -> FloatArray:
        new, old = _log_weights(self.distribution, proposed, current)
        return log_ratio(new, old)

    def _coupled_log_alpha(
        self,
        target: Target,
        state_x: State,
        state_y: State,
        proposed_x: State,
        proposed_y: State,
        forward_x: Any,
        forward_y: Any,
    ) -> tuple[FloatArray, FloatArray]:
        # One call of the distribution's density, on 3n rows: the proposal, which both
        # chains share, and both chains' states.
        new, old_x, old_y = _log_weights(self.distribution, proposed_x, state_x, state_y)
        return log_ratio(new, old_x), log_ratio(new, old_y)

    def __repr__(self) -> str:
        return f"IndependenceProposal({self.distribution!r})"


# Spec section 5.1, R-KER-1 to R-KER-6.
class MetropolisHastings(Kernel):
    r"""A Metropolis-Hastings kernel from a proposal, coupled by a coupling of that proposal.

    A step draws :math:`x' \sim q(\cdot \mid x)` and accepts it when
    :math:`\log u < \log\pi(x') - \log\pi(x) + \log q(x \mid x') - \log q(x' \mid x)`. A
    coupled step draws both chains' proposals from ``coupling``, evaluates identical
    proposals once, and accepts each with the same uniform. When the coupling returns
    one array for both chains, as :class:`~telescopic.couplings.Shared` does, the step
    evaluates it once and gives both chains its state.

    Parameters
    ----------
    proposal : Proposal or None
        The proposal of every chain. ``None`` leaves the kernel untuned, for a subclass
        that sets its proposal by tuning, as :class:`~telescopic.kernels.RandomWalk` does.
    coupling : Coupling
        A coupling whose ``supports(proposal)`` is true; otherwise ``ValueError``.
    """

    # What _ready raises while the proposal is unset; each tunable subclass words its own.
    _untuned = "MetropolisHastings needs a proposal"

    def __init__(self, proposal: Proposal | None, coupling: Coupling) -> None:
        instance("coupling", coupling, Coupling)
        if proposal is not None:
            instance("proposal", proposal, Proposal)
            if not coupling.supports(proposal):
                raise ValueError(f"{coupling!r} does not couple a {type(proposal).__name__}")
        self._proposal, self._coupling = proposal, coupling

    @property
    def needs_grad(self) -> bool:
        return self._proposal is not None and self._proposal.needs_grad

    @property
    def is_tuned(self) -> bool:
        return self._proposal is not None

    @property
    def ndim(self) -> int | None:
        return None if self._proposal is None else self._proposal.ndim

    def init_extra(self, state: State) -> dict[str, npt.NDArray[Any]]:
        return {} if self._proposal is None else self._proposal.init_extra(state)

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        proposal, _ = self._ready()
        x, forward = proposal.sample(target, state, draws)
        proposed = proposal.evaluate(target, x, state)
        log_alpha = proposal.log_alpha(target, state, proposed, forward)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposed, state), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        proposal, coupling = self._ready()
        pair = coupling.couple(target, proposal, state_x, state_y, draws)
        if pair.y is pair.x:
            # Every pair shares one draw, as under Shared: one state serves both chains,
            # which select never writes to.
            proposed_x = proposed_y = proposal.evaluate(target, pair.x, state_x)
        else:
            proposed_x, proposed_y = evaluate_pair(
                partial(proposal.evaluate, target, like=state_x), pair.x, pair.y, pair.identical
            )
        log_u = draws.log_uniform()
        alpha_x, alpha_y = proposal._coupled_log_alpha(
            target, state_x, state_y, proposed_x, proposed_y, pair.forward_x, pair.forward_y
        )
        return select(log_u < alpha_x, proposed_x, state_x), select(
            log_u < alpha_y, proposed_y, state_y
        )

    def _ready(self) -> tuple[Proposal, Coupling]:
        if self._proposal is None:
            raise ValueError(self._untuned)
        return self._proposal, self._coupling

    def __repr__(self) -> str:
        return f"MetropolisHastings({self._proposal!r}, {self._coupling!r})"


def _drift(state: State, chol: FloatArray, blas: bool) -> FloatArray:
    # The whitened drift h = B^T grad log pi(x), with chol = B = epsilon C, so that the
    # mean of q(. | x) is x + B h / 2.
    drift: FloatArray = matvec(chol.T, cast(FloatArray, state.grad), blas=blas)
    return drift


def _log_alpha(
    current: State,
    proposal: State,
    residual: FloatArray,
    drift: FloatArray,
    chol: FloatArray,
    blas: bool,
) -> FloatArray:
    # log pi(x') - log pi(x) + log q(x | x') - log q(x' | x): x' = x + B(h / 2 + r) gives
    # B^{-1}(x' - mu(x)) = r and B^{-1}(x - mu(x')) = -(r + (h + h') / 2); the Gaussian
    # constants cancel. A rejected proposal (log pi = -inf) carries a zero gradient and may
    # hold non-finite positions: no warnings, and a NaN ratio rejects as -inf does.
    with np.errstate(over="ignore", invalid="ignore"):
        back = residual + 0.5 * (drift + _drift(proposal, chol, blas))
        log_q = 0.5 * (
            np.einsum("ij,ij->i", residual, residual) - np.einsum("ij,ij->i", back, back)
        )
        log_alpha: FloatArray = proposal.log_prob - current.log_prob + log_q
    return log_alpha


def _log_weights(q: ProposalDistribution, proposal: State, *states: State) -> list[FloatArray]:
    # log pi - log q at the proposal's positions, then at each state's, from one call of
    # q.log_density, whose rows do not depend on each other (spec R-KER-6). A NaN log q
    # would make every move to its point a rejection, and so bias the chain without an
    # error.
    x = np.concatenate([proposal.x, *(state.x for state in states)])
    log_q = log_densities("proposal log_density", q.log_density(x), x).reshape(1 + len(states), -1)
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


class SharedCovarianceGaussian(Protocol):
    r"""Gaussian proposals of two chains that share their covariance :math:`BB^\top`."""

    def reflect(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> CoupledProposals:
        """The proposals of both chains from the reflection-maximal coupling of their laws.

        Parameters
        ----------
        target : Target
            The target that the chains sample.
        state_x, state_y : State
            The first and the second chains of the ``n`` pairs.
        draws : Draws
            Random numbers for the ``n`` pairs.

        Returns
        -------
        CoupledProposals
        """
        ...
