"""Mixtures of kernels, coupled by a common choice of component."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws
from telescopic._types import FloatArray, IndexArray, Seed, instance, real_array
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target
from telescopic.tuning import PilotRun, TuningReport


# Spec R-KER-MIX.
class Mixture(Kernel):
    """A mixture of kernels.

    Each step draws one uniform and applies the component it selects. The coupled step
    gives both chains the same component and applies that component's coupling. The
    choice never depends on the states, which keeps each marginal the mixture
    (Atchadé & Jacob 2024, section 4.2).

    Parameters
    ----------
    kernels : sequence of Kernel
        Kernel instances. :meth:`~telescopic.kernels.Kernel.tune` tunes the untuned ones on one set of pilot chains; its report gives
        the largest weight ratio of any independence component.
    weights : array_like
        Non-negative weights with a finite sum, normalized internally.
    """

    def __init__(self, kernels: Sequence[Kernel], weights: npt.ArrayLike) -> None:
        self.kernels = list(kernels)
        for i, kernel in enumerate(self.kernels):
            instance(f"kernels[{i}]", kernel, Kernel)
        weights = real_array("weights", weights, "real numbers, one per kernel")
        # Finite weights near the largest float can have a sum that overflows, and inf and
        # -inf one that is NaN; a finite positive sum of non-negative weights makes each
        # finite.
        with np.errstate(over="ignore", invalid="ignore"):
            total = weights.sum()
        valid = np.all(weights >= 0) and 0 < total < np.inf
        if weights.shape != (len(self.kernels),) or not valid:
            raise ValueError(
                "weights must be one finite, non-negative number per kernel, not all zero, "
                "with a finite sum"
            )
        if len({kernel.ndim for kernel in self.kernels} - {None}) > 1:
            raise ValueError("the kernels work in different dimensions")
        self.weights = weights / total
        # Rounding can leave the cumulative weights' last below 1, so _choose clamps the
        # index to the last component of positive weight.
        self._last = int(np.nonzero(self.weights)[0][-1])
        self._cumulative = np.cumsum(self.weights)

    @property
    def needs_grad(self) -> bool:
        return any(kernel.needs_grad for kernel in self.kernels)

    @property
    def needs_log_prob(self) -> bool:
        return any(kernel.needs_log_prob for kernel in self.kernels)

    @property
    def is_tuned(self) -> bool:
        return all(kernel.is_tuned for kernel in self.kernels)

    @property
    def ndim(self) -> int | None:
        return next((kernel.ndim for kernel in self.kernels if kernel.ndim is not None), None)

    def check_ndim(self, ndim: int) -> None:
        # The mixture's dimension, then every component's, which tuning, unlike the
        # constructor, does not compare.
        super().check_ndim(ndim)
        for i, kernel in enumerate(self.kernels):
            try:
                kernel.check_ndim(ndim)
            except ValueError as error:
                raise ValueError(f"component {i} of the mixture: {error}") from None

    def _tuned(
        self, target: Target, initial: Any, pilot: PilotRun, rng: Seed
    ) -> tuple[Kernel, TuningReport]:
        # Each component tuned on the one pilot by its own tune, which a kernel of one's own
        # overrides and may read initial.
        tuned = [kernel.tune(target, initial, rng, pilot=pilot) for kernel in self.kernels]
        ratios = [report.max_weight_ratio for _, report in tuned if report is not None]
        ratio = max((r for r in ratios if r is not None), default=None)
        # A copy keeps the class and settings of a subclass, such as HMC.
        mixture = copy.copy(self)
        mixture.kernels = [kernel for kernel, _ in tuned]
        return mixture, pilot.report(ratio)

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        component = self._choose(draws)
        if (component == component[0]).all():
            # One component for every row, always so at one lane: no copies.
            return self.kernels[int(component[0])].step(target, state, draws)
        state = state.take(slice(None))
        probability = np.zeros(len(state))
        for index, kernel in enumerate(self.kernels):
            rows = (component == index).nonzero()[0]
            if len(rows):
                new, probability[rows] = kernel.step(target, state.take(rows), draws.subset(rows))
                state.put(rows, new)
        return state, probability

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        component = self._choose(draws)
        if (component == component[0]).all():
            return self.kernels[int(component[0])].coupled_step(target, state_x, state_y, draws)
        state_x, state_y = state_x.take(slice(None)), state_y.take(slice(None))
        for index, kernel in enumerate(self.kernels):
            rows = (component == index).nonzero()[0]
            if len(rows):
                new_x, new_y = kernel.coupled_step(
                    target, state_x.take(rows), state_y.take(rows), draws.subset(rows)
                )
                state_x.put(rows, new_x)
                state_y.put(rows, new_y)
        return state_x, state_y

    def _choose(self, draws: Draws) -> IndexArray:
        index: IndexArray = np.searchsorted(self._cumulative, draws.uniform(), side="right")
        return np.minimum(index, self._last)

    def __repr__(self) -> str:
        return f"Mixture({self.kernels!r}, weights={self.weights.tolist()})"
