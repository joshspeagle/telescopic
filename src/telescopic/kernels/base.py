"""States, the kernel contract, and helpers shared by Metropolis-type kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray
from telescopic.target import Target


@dataclass
class State:
    """Positions, log-densities and (optionally) gradients of ``n`` chains, batch first.

    A kernel returns states with the fields of its input states, valid at the returned
    positions: a state that carries gradients gets them evaluated at every new position.

    Attributes
    ----------
    x : ndarray, shape (n, d)
    log_prob : ndarray, shape (n,)
    grad : ndarray, shape (n, d), or None
    """

    x: FloatArray
    log_prob: FloatArray
    grad: FloatArray | None = None

    def __len__(self) -> int:
        return len(self.log_prob)

    def take(self, rows: Any) -> State:
        """A copy of the chains in ``rows`` (an index, an index array, a mask or a slice)."""
        rows = np.arange(len(self))[rows]
        return State(
            self.x[rows], self.log_prob[rows], None if self.grad is None else self.grad[rows]
        )

    def put(self, rows: Any, other: State) -> None:
        """Overwrite the chains in ``rows`` with ``other``, in place."""
        self.x[rows] = other.x
        self.log_prob[rows] = other.log_prob
        if self.grad is not None:
            self.grad[rows] = other.grad


class Kernel(ABC):
    """A Markov kernel with its coupling (spec section 5.1, R-KER-1 to R-KER-5).

    Subclasses implement :meth:`step` and :meth:`coupled_step` on batches of chains.
    Each marginal of :meth:`coupled_step` must have exactly the law of :meth:`step`. The
    engine declares a meeting when a pair's returned positions are bitwise equal, so a
    coupling that makes the two proposals identical must give the second chain a copy of
    the first chain's proposal, never a recomputation. A row's arithmetic must not depend
    on the other rows of the batch (spec R-KER-6).
    """

    @property
    def needs_grad(self) -> bool:
        """Whether the kernel evaluates the gradient of the log-density."""
        return False

    @property
    def is_tuned(self) -> bool:
        """Whether every parameter is set, so the kernel can run."""
        return True

    def init_state(self, target: Target, x: FloatArray) -> State:
        """Evaluate the target at the rows of ``x``, with gradients if the kernel needs them.

        A non-finite gradient where the log-density is finite raises ``ValueError`` here;
        gradient kernels reject such a proposal instead (spec R-INPUT-1).
        """
        state = evaluate(target, x, self.needs_grad)
        if state.grad is not None:
            bad = np.isfinite(state.log_prob) & ~np.isfinite(state.grad).all(axis=1)
            if bad.any():
                row = int(np.argmax(bad))
                raise ValueError(
                    f"grad returned {state.grad[row].tolist()} at x = {x[row].tolist()}, "
                    "where log_prob is finite"
                )
        return state

    @abstractmethod
    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, BoolArray]:
        """One transition of each chain.

        Returns
        -------
        state : State
        accepted : ndarray of bool, shape (n,)
            Whether each chain moved to its proposal.
        """

    @abstractmethod
    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        """One coupled transition of each pair.

        Returns
        -------
        state_x, state_y : State
        """


def evaluate(target: Target, x: FloatArray, need_grad: bool) -> State:
    """The state at the rows of ``x``: log-densities, and gradients if needed."""
    if need_grad:
        log_prob, grad = target.log_prob_and_grad(x)
        return State(x, log_prob, grad)
    return State(x, target.log_prob(x))


def evaluate_pair(
    target: Target, x: FloatArray, y: FloatArray, identical: BoolArray, need_grad: bool
) -> tuple[State, State]:
    """States at the coupled proposals ``x`` and ``y`` in one batched call.

    Identical proposals are evaluated once and their state is shared.
    """
    distinct = np.flatnonzero(~identical)
    both = evaluate(target, np.concatenate([x, y[distinct]]), need_grad)
    source = np.arange(len(x))
    source[distinct] = np.arange(len(x), len(both))
    return both.take(slice(0, len(x))), both.take(source)


def log_ratio(new: FloatArray, old: FloatArray) -> FloatArray:
    """``new - old`` for log-densities; ``-inf - -inf`` gives NaN, which every Metropolis
    test rejects, without a warning."""
    with np.errstate(invalid="ignore"):
        return new - old


def select(accepted: BoolArray, proposal: State, current: State) -> State:
    """Row by row, ``proposal`` where ``accepted`` and ``current`` elsewhere."""
    return State(
        np.where(accepted[:, None], proposal.x, current.x),
        np.where(accepted, proposal.log_prob, current.log_prob),
        None if current.grad is None else np.where(accepted[:, None], proposal.grad, current.grad),
    )
