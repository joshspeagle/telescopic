"""Mixtures of kernels, coupled by a common choice of component."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws
from telescopic.kernels.base import BoolArray, Kernel, State
from telescopic.target import Target


class Mixture(Kernel):
    """A mixture of kernels (spec R-KER-MIX).

    Each step draws one uniform and applies the component it selects. The coupled step
    gives both chains the same component and applies that component's coupling. The
    choice never depends on the states, which keeps each marginal the mixture
    (Atchade & Jacob 2024, section 4.2).

    Parameters
    ----------
    kernels : sequence of Kernel
    weights : array_like
        Non-negative weights, normalised internally.
    """

    def __init__(self, kernels: Sequence[Kernel], weights: npt.ArrayLike) -> None:
        self.kernels = list(kernels)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(self.kernels),) or np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("weights must be one non-negative number per kernel, not all zero")
        self.weights = weights / weights.sum()
        self._cumulative = np.cumsum(self.weights)

    @property
    def needs_grad(self) -> bool:
        return any(kernel.needs_grad for kernel in self.kernels)

    @property
    def is_tuned(self) -> bool:
        return all(kernel.is_tuned for kernel in self.kernels)

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, BoolArray]:
        component = self._choose(draws)
        state = state.take(slice(None))
        accepted = np.zeros(len(state), dtype=bool)
        for index, kernel in enumerate(self.kernels):
            rows = np.flatnonzero(component == index)
            if len(rows):
                new, moved = kernel.step(target, state.take(rows), draws.subset(rows))
                state.put(rows, new)
                accepted[rows] = moved
        return state, accepted

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State, BoolArray]:
        component = self._choose(draws)
        state_x, state_y = state_x.take(slice(None)), state_y.take(slice(None))
        met = np.zeros(len(state_x), dtype=bool)
        for index, kernel in enumerate(self.kernels):
            rows = np.flatnonzero(component == index)
            if len(rows):
                new_x, new_y, met_rows = kernel.coupled_step(
                    target, state_x.take(rows), state_y.take(rows), draws.subset(rows)
                )
                state_x.put(rows, new_x)
                state_y.put(rows, new_y)
                met[rows] = met_rows
        return state_x, state_y, met

    def _choose(self, draws: Draws) -> npt.NDArray[np.intp]:
        index: npt.NDArray[np.intp] = np.searchsorted(
            self._cumulative, draws.uniform(), side="right"
        )
        return np.minimum(index, len(self.kernels) - 1)

    def __repr__(self) -> str:
        return f"Mixture({self.kernels!r}, weights={self.weights.tolist()})"
