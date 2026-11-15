"""Metropolis-Hastings kernels built from a proposal and a coupling of that proposal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, NamedTuple, cast

import numpy as np

from telescopic._linalg import matvec
from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray, instance
from telescopic.couplings import maximal
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


class Proposal(ABC):
    """A Metropolis-Hastings proposal for a batch of chains: its draw, its evaluation and its log acceptance ratio."""

    @property
    def needs_grad(self) -> bool:
        """Whether the proposal needs the gradient of the log-density at the current states.

        Returns
        -------
        bool
        """
        return False

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
    ) -> tuple[FloatArray, State, State]:
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
        log_alpha : ndarray, shape (n,)
            NaN, from ``-inf - -inf``, is a rejection.
        current, proposed : State
            Both states, with any fields that the ratio filled.
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
        identical: BoolArray,
    ) -> tuple[tuple[FloatArray, State, State], tuple[FloatArray, State, State]]:
        # The log acceptance ratios of both chains of a coupled step; a proposal whose
        # ratios share a computation overrides it.
        return (
            self.log_alpha(target, state_x, proposed_x, forward_x),
            self.log_alpha(target, state_y, proposed_y, forward_y),
        )


# Spec R-KER-RW.
class RandomWalkProposal(Proposal):
    r"""The Gaussian random-walk proposal :math:`x' = x + Bz`, :math:`z \sim \mathcal N(0, I)`.

    Its density is symmetric, so the log acceptance ratio is
    :math:`\log\pi(x') - \log\pi(x)`.

    Parameters
    ----------
    factor : ndarray, shape (d, d)
        :math:`B`, a lower-triangular factor of the proposal covariance :math:`BB^\top`.
    inverse : ndarray, shape (d, d)
        :math:`B^{-1}`.
    """

    def __init__(self, factor: FloatArray, inverse: FloatArray) -> None:
        self.factor, self.inverse = factor, inverse

    def sample(self, target: Target, state: State, draws: Draws) -> tuple[FloatArray, Any]:
        normals = draws.normal(len(self.factor))
        return state.x + matvec(self.factor, normals, blas=target.vectorize), None

    def log_alpha(
        self, target: Target, current: State, proposed: State, forward: Any
    ) -> tuple[FloatArray, State, State]:
        return log_ratio(proposed.log_prob, current.log_prob), current, proposed

    def __repr__(self) -> str:
        d = len(self.factor)
        return f"RandomWalkProposal(factor=<{d}x{d}>)"


# Spec section 5.1, R-KER-1 to R-KER-6.
class MetropolisHastings(Kernel):
    r"""A Metropolis-Hastings kernel from a proposal, coupled by a coupling of that proposal.

    A step draws :math:`x' \sim q(\cdot \mid x)` and accepts it when
    :math:`\log u < \log\pi(x') - \log\pi(x) + \log q(x \mid x') - \log q(x' \mid x)`. A
    coupled step draws both chains' proposals from ``coupling``, evaluates identical
    proposals once, and accepts each with the same uniform.

    Parameters
    ----------
    proposal : Proposal
        The proposal of every chain.
    coupling : Coupling
        A coupling whose ``supports(proposal)`` is true.
    """

    def __init__(self, proposal: Proposal, coupling: Coupling) -> None:
        instance("proposal", proposal, Proposal)
        instance("coupling", coupling, Coupling)
        if not coupling.supports(proposal):
            raise ValueError(f"{coupling!r} does not couple a {type(proposal).__name__}")
        self._proposal: Proposal | None = proposal
        self._coupling = coupling

    @property
    def needs_grad(self) -> bool:
        return self._proposal is not None and self._proposal.needs_grad

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        proposal, _ = self._ready()
        x, forward = proposal.sample(target, state, draws)
        proposed = proposal.evaluate(target, x, state)
        log_alpha, current, proposed = proposal.log_alpha(target, state, proposed, forward)
        accepted = draws.log_uniform() < log_alpha
        return select(accepted, proposed, current), acceptance(log_alpha)

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        proposal, coupling = self._ready()
        pair = coupling.couple(target, proposal, state_x, state_y, draws)
        proposed_x, proposed_y = evaluate_pair(
            partial(proposal.evaluate, target, like=state_x), pair.x, pair.y, pair.identical
        )
        log_u = draws.log_uniform()
        (alpha_x, state_x, proposed_x), (alpha_y, state_y, proposed_y) = (
            proposal._coupled_log_alpha(
                target,
                state_x,
                state_y,
                proposed_x,
                proposed_y,
                pair.forward_x,
                pair.forward_y,
                pair.identical,
            )
        )
        return select(log_u < alpha_x, proposed_x, state_x), select(
            log_u < alpha_y, proposed_y, state_y
        )

    def _ready(self) -> tuple[Proposal, Coupling]:
        # The shipped kernels override it to raise while they are untuned.
        return cast(Proposal, self._proposal), self._coupling

    def __repr__(self) -> str:
        return f"MetropolisHastings({self._proposal!r}, {self._coupling!r})"


class CoupledProposals(NamedTuple):
    """The coupled proposals of ``n`` pairs of chains, with what each proposal's ratio needs.

    Attributes
    ----------
    x, y : ndarray, shape (n, d)
        The proposals of the first and the second chains; on identical rows ``y`` is a
        copy of ``x``.
    identical : ndarray of bool, shape (n,)
        Rows where the two proposals are equal.
    forward_x, forward_y : object
        What :meth:`Proposal.log_alpha` needs of each chain's draw, as
        :meth:`Proposal.sample` returns it.
    """

    x: FloatArray
    y: FloatArray
    identical: BoolArray
    forward_x: Any
    forward_y: Any


class Coupling(ABC):
    """A coupling of the proposals of two chains under one proposal.

    Each chain's proposal must have exactly the law of the proposal's
    :meth:`Proposal.sample` from its state, and two chains in equal states must receive
    identical proposals, with ``y`` a copy of ``x``.
    """

    @abstractmethod
    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        """Draw the coupled proposals of ``n`` pairs.

        Parameters
        ----------
        target : Target
            The target that the chains sample.
        proposal : Proposal
            The proposal of both chains, one that :meth:`supports` accepts.
        state_x, state_y : State
            The first and the second chains of the pairs.
        draws : Draws
            Random numbers for the ``n`` pairs.

        Returns
        -------
        CoupledProposals
        """

    @abstractmethod
    def supports(self, proposal: Proposal) -> bool:
        """Whether this coupling couples ``proposal``, matched by its exact type.

        Parameters
        ----------
        proposal : Proposal
            The proposal to check.

        Returns
        -------
        bool
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# Spec R-CPL-2, with eta = 1.
class Maximal(Coupling):
    r"""The rejection maximal coupling of two random-walk proposals.

    Each pair draws from its own generator by the rejection maximal coupling with
    :math:`\eta = 1` of :math:`\mathcal N(x, BB^\top)` and
    :math:`\mathcal N(y, BB^\top)` (Jacob, O'Leary & Atchadé 2020, Algorithm 2). Its
    meeting probability is maximal, but its meeting times grow exponentially with the
    dimension; it is for comparison.
    """

    def couple(
        self,
        target: Target,
        proposal: Proposal,
        state_x: State,
        state_y: State,
        draws: Draws,
    ) -> CoupledProposals:
        walk = cast(RandomWalkProposal, proposal)
        factor, inverse = walk.factor, walk.inverse
        x, y, identical, _ = maximal(
            [_Gaussian(mean, factor, inverse) for mean in state_x.x],
            [_Gaussian(mean, factor, inverse) for mean in state_y.x],
            draws,
        )
        return CoupledProposals(x, y, identical, None, None)

    def supports(self, proposal: Proposal) -> bool:
        return type(proposal) is RandomWalkProposal


class _Gaussian:
    # A Gaussian proposal for the rejection coupling; both members of a pair share chol,
    # so the normalizing constants cancel and are left out.
    def __init__(self, mean: FloatArray, chol: FloatArray, inverse: FloatArray) -> None:
        self.mean, self.chol, self.inverse = mean, chol, inverse

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        return self.mean + self.chol @ random_state.standard_normal(len(self.mean))

    def logpdf(self, x: FloatArray) -> float:
        z = self.inverse @ (x - self.mean)
        return float(-0.5 * z @ z)
