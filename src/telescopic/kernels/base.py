"""States, the kernel contract, and helpers shared by Metropolis-type kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws, FloatArray
from telescopic.target import Target

BoolArray = npt.NDArray[np.bool_]


@dataclass
class State:
    """Positions, log-densities and (optionally) gradients of ``n`` chains, batch first.

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
        """The chains in ``rows``."""
        return State(
            self.x[rows], self.log_prob[rows], None if self.grad is None else self.grad[rows]
        )

    def put(self, rows: Any, other: State) -> None:
        """Overwrite the chains in ``rows`` with ``other``, in place."""
        self.x[rows] = other.x
        self.log_prob[rows] = other.log_prob
        if self.grad is not None and other.grad is not None:
            self.grad[rows] = other.grad


class Kernel(ABC):
    """A Markov kernel with its coupling (spec section 5.1, R-KER-1 to R-KER-5).

    Subclasses implement :meth:`step` and :meth:`coupled_step` on batches of chains.
    Each marginal of :meth:`coupled_step` must have exactly the law of :meth:`step`.
    ``met[i]`` must hold exactly when the returned states of pair ``i`` are bitwise
    equal: on a meeting the kernel copies the first chain's state into the second. From
    equal states, :meth:`coupled_step` must return equal states.
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
        """Evaluate the target at the rows of ``x``."""
        return evaluate(target, x, self.needs_grad)

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
    ) -> tuple[State, State, BoolArray]:
        """One coupled transition of each pair.

        Returns
        -------
        state_x, state_y : State
        met : ndarray of bool, shape (n,)
            Whether each pair's states are now equal.
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
    distinct = ~identical
    both = evaluate(target, np.concatenate([x, y[distinct]]), need_grad)
    state_x = both.take(slice(0, len(x)))
    state_y = State(
        y, state_x.log_prob.copy(), None if both.grad is None else both.grad[: len(x)].copy()
    )
    state_y.put(distinct, both.take(slice(len(x), None)))
    return state_x, state_y


def log_ratio(new: FloatArray, old: FloatArray) -> FloatArray:
    """``new - old`` for log-densities; ``-inf - -inf`` gives NaN, which every Metropolis
    test rejects, without a warning."""
    with np.errstate(invalid="ignore"):
        return new - old


def equal_rows(a: FloatArray, b: FloatArray) -> BoolArray:
    """Whether each row of ``a`` is bitwise equal to the same row of ``b``: a meeting."""
    return np.all(a == b, axis=1)


def select(accepted: BoolArray, proposal: State, current: State) -> State:
    """Row by row, ``proposal`` where ``accepted`` and ``current`` elsewhere."""
    return State(
        np.where(accepted[:, None], proposal.x, current.x),
        np.where(accepted, proposal.log_prob, current.log_prob),
        None
        if current.grad is None or proposal.grad is None
        else np.where(accepted[:, None], proposal.grad, current.grad),
    )
