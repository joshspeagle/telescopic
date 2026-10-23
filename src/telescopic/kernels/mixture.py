"""Mixtures of kernels, coupled by a common choice of component."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import Draws
from telescopic._types import BoolArray, Distribution, IndexArray
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport, adaptive_metropolis


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
        valid = np.all(np.isfinite(weights) & (weights >= 0)) and weights.sum() > 0
        if weights.shape != (len(self.kernels),) or not valid:
            raise ValueError(
                "weights must be one finite, non-negative number per kernel, not all zero"
            )
        if len({kernel.ndim for kernel in self.kernels} - {None}) > 1:
            raise ValueError("the kernels work in different dimensions")
        self.weights = weights / weights.sum()
        self._cumulative = np.cumsum(self.weights)

    @property
    def needs_grad(self) -> bool:
        return any(kernel.needs_grad for kernel in self.kernels)

    @property
    def is_tuned(self) -> bool:
        return all(kernel.is_tuned for kernel in self.kernels)

    @property
    def ndim(self) -> int | None:
        return next((kernel.ndim for kernel in self.kernels if kernel.ndim is not None), None)

    def tune(
        self,
        target: Target,
        initial: Distribution,
        rng: int | SeedSequence | Generator | None = None,
        *,
        pilot: PilotRun | None = None,
    ) -> tuple[Kernel, TuningReport | None]:
        """Tune the untuned components on one set of pilot chains (see :meth:`Kernel.tune`).

        The report carries the largest weight ratio of any independence component.
        """
        if self.is_tuned:
            return self, None
        pilot = adaptive_metropolis(target, initial, rng) if pilot is None else pilot
        tuned = [kernel.tune(target, initial, rng, pilot=pilot) for kernel in self.kernels]
        ratios = [report.max_weight_ratio for _, report in tuned if report is not None]
        ratio = max((r for r in ratios if r is not None), default=None)
        return Mixture([kernel for kernel, _ in tuned], self.weights), pilot.report(ratio)

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
    ) -> tuple[State, State]:
        component = self._choose(draws)
        state_x, state_y = state_x.take(slice(None)), state_y.take(slice(None))
        for index, kernel in enumerate(self.kernels):
            rows = np.flatnonzero(component == index)
            if len(rows):
                new_x, new_y = kernel.coupled_step(
                    target, state_x.take(rows), state_y.take(rows), draws.subset(rows)
                )
                state_x.put(rows, new_x)
                state_y.put(rows, new_y)
        return state_x, state_y

    def _choose(self, draws: Draws) -> IndexArray:
        index: IndexArray = np.searchsorted(self._cumulative, draws.uniform(), side="right")
        return np.minimum(index, len(self.kernels) - 1)

    def __repr__(self) -> str:
        return f"Mixture({self.kernels!r}, weights={self.weights.tolist()})"
