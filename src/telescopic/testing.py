"""Helpers for testing kernels and couplings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import stats

from telescopic._random import Draws, Streams, root_seed
from telescopic._types import FloatArray, Seed
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target

# Random one-dimensional projections on which check_coupling compares the moved values.
PROJECTIONS = 5

# Coupled steps, and pairs, over which check_coupling expects equal states to stay equal.
FAITHFUL_STEPS = 100
FAITHFUL_PAIRS = 100


@dataclass(frozen=True)
class CouplingReport:
    """The checks of :func:`check_coupling`.

    Attributes
    ----------
    stay : tuple of two floats
        For X and for Y, the p-value of Fisher's exact test that the chain stays put as
        often in a coupled step as in a step of the kernel alone.
    moves : tuple of two floats
        For X and for Y, the Bonferroni-corrected p-value of two-sample
        Kolmogorov-Smirnov tests that compare the moved positions of a coupled step with
        those of a step alone, along random directions and in the distance moved.
    faithful : bool
        Whether 100 pairs of equal states stayed equal over 100 coupled steps.
    near_misses : int
        Pairs whose new positions differ by less than 1e-9 of their size without being
        equal: a coupling that recomputes Y where it should copy X's proposal leaves
        them, and the engine never sees such pairs meet (spec R-KER-3).
    """

    stay: tuple[float, float]
    moves: tuple[float, float]
    faithful: bool
    near_misses: int

    def passed(self, alpha: float = 1e-3) -> bool:
        """Whether every p-value is at least ``alpha``, the coupling is faithful and no pair nearly met.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        bool
        """
        return min(*self.stay, *self.moves) >= alpha and self.faithful and not self.near_misses


def check_coupling(
    kernel: Kernel,
    target: Target,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n: int = 10_000,
    rng: Seed = None,
) -> CouplingReport:
    """Check that each chain of a coupled step has the law of a step alone (spec section 9).

    From ``n`` copies of the pair ``(x, y)``, one coupled step is compared with one step
    of the kernel alone from ``x`` and one from ``y``: how often each chain stays put,
    and where it moves to. Equal states must also stay equal, in 100 pairs over 100
    coupled steps. Custom kernels should pass it before use.

    Parameters
    ----------
    kernel : Kernel
        A tuned kernel.
    target : Target
    x, y : array_like, shape (d,)
        The starting points, close enough for the coupling to meet sometimes.
    n : int
        Copies of the pair; at least 2. More detect smaller differences.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator

    Returns
    -------
    CouplingReport

    Notes
    -----
    Costs one coupled step and two steps of ``n`` chains, then 100 coupled steps of 100
    equal pairs.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be points of one shape (d,), not {x.shape} and {y.shape}")
    if n < 2:
        raise ValueError(f"n must be at least 2, not {n}")
    seeds = root_seed(rng).spawn(5)
    start_x = kernel.init_state(target, np.tile(x, (n, 1)))
    start_y = kernel.init_state(target, np.tile(y, (n, 1)))
    new_x, new_y = kernel.coupled_step(target, start_x, start_y, random_draws(n, seeds[0]))
    alone = [
        kernel.step(target, start, random_draws(n, seed))[0]
        for start, seed in ((start_x, seeds[1]), (start_y, seeds[2]))
    ]
    directions = np.random.default_rng(seeds[3]).standard_normal((PROJECTIONS, len(x)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    stay, moves = [], []
    for point, coupled, single in ((x, new_x.x, alone[0].x), (y, new_y.x, alone[1].x)):
        stayed = [np.all(chain == point, axis=1) for chain in (coupled, single)]
        table = [[int(s.sum()), int(n - s.sum())] for s in stayed]
        stay.append(float(stats.fisher_exact(table).pvalue))
        moved = [chain[~s] for chain, s in zip((coupled, single), stayed, strict=True)]
        moves.append(_compare_moves(point, moved[0], moved[1], directions))
    gap = np.linalg.norm(new_x.x - new_y.x, axis=1)
    size = 1 + np.linalg.norm(new_x.x, axis=1)
    near_misses = int(np.sum((gap > 0) & (gap < 1e-9 * size)))
    return CouplingReport(
        (stay[0], stay[1]),
        (moves[0], moves[1]),
        _faithful(kernel, target, start_x.take(slice(FAITHFUL_PAIRS)), seeds[4]),
        near_misses,
    )


def _compare_moves(
    point: FloatArray, coupled: FloatArray, single: FloatArray, directions: FloatArray
) -> float:
    # Kolmogorov-Smirnov tests along each direction and of the distance moved,
    # Bonferroni-corrected; with fewer than two moves on a side there is nothing to test.
    if min(len(coupled), len(single)) < 2:
        return 1.0
    samples = [(coupled @ direction, single @ direction) for direction in directions] + [
        (np.linalg.norm(coupled - point, axis=1), np.linalg.norm(single - point, axis=1))
    ]
    smallest = min(float(stats.ks_2samp(a, b).pvalue) for a, b in samples)
    return min(1.0, len(samples) * smallest)


def _faithful(kernel: Kernel, target: Target, start: State, seed: np.random.SeedSequence) -> bool:
    state_x, state_y = start, start.take(slice(None))
    draws = random_draws(len(start), seed)
    for _ in range(FAITHFUL_STEPS):
        state_x, state_y = kernel.coupled_step(target, state_x, state_y, draws)
        if not np.array_equal(state_x.x, state_y.x):
            return False
    return True


def fixed_draws(
    normals: npt.ArrayLike | None = None,
    uniforms: npt.ArrayLike | None = None,
    generators: Sequence[Generator] | None = None,
) -> Draws:
    """Draws that replay prescribed random numbers, for tests that inject randomness.

    Parameters
    ----------
    normals, uniforms : array_like, shape (n, k), optional
        Row ``i`` is consumed left to right by row ``i`` of every request.
    generators : sequence of numpy.random.Generator, length n, optional
        Served by :meth:`Draws.generator`.

    Returns
    -------
    Draws
        Draws for ``n`` rows; a request beyond the prescribed numbers raises ``IndexError``.
    """
    arrays = [
        None if a is None else np.array(a, dtype=np.float64, ndmin=2, order="C")
        for a in (normals, uniforms)
    ]
    rows = {len(a) for a in (*arrays, generators) if a is not None}
    if len(rows) != 1:
        raise ValueError("give normals, uniforms or generators, all with the same number of rows")
    n_rows = rows.pop()
    return _Replay(n_rows, arrays, generators).view(np.arange(n_rows))


def random_draws(n_rows: int, rng: Seed = None) -> Draws:
    """Draws whose rows read fresh numbers from one generator, for statistical tests.

    The rows are independent, but a row's numbers depend on the other rows' requests,
    unlike the per-replicate streams of a run.

    Parameters
    ----------
    n_rows : int
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator

    Returns
    -------
    Draws
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be at least 1, not {n_rows}")
    return _Shared(n_rows, np.random.default_rng(rng)).view(np.arange(n_rows))


class _Replay(Streams):
    def __init__(
        self,
        n_rows: int,
        arrays: list[FloatArray | None],
        generators: Sequence[Generator] | None,
    ) -> None:
        # Empty buffers of width 0, so the cursors start at 0 for the given arrays too.
        super().__init__(n_rows, width=0)
        for kind, array in enumerate(arrays):
            if array is not None:
                self._buffers[kind] = array
        if generators is not None:
            self._generators = list(generators)

    def _fill(self, kind: int, lane: int, out: FloatArray) -> None:
        raise IndexError(f"the injected random numbers of row {lane} are exhausted")


class _Shared(Streams):
    def __init__(self, n_rows: int, generator: Generator) -> None:
        super().__init__(n_rows, width=64)
        self._generators = [generator] * n_rows
        self._sources = [[generator, generator]] * n_rows
