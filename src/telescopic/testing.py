"""Helpers for testing kernels and couplings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.random import Generator
from scipy import stats

from telescopic._random import Draws, Streams, root_seed
from telescopic._types import FloatArray, Seed, integer, real
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target

# Random one-dimensional projections on which check_coupling compares the moved values.
PROJECTIONS = 5

# Positions apart by at most this many units in the last place are a near miss.
NEAR_MISS_ULPS = 64


@dataclass(frozen=True)
class CouplingReport:
    """The checks of :func:`check_coupling`.

    Attributes
    ----------
    stay : tuple of two floats
        For the first and the second chain of the pairs, the p-value of Fisher's exact
        test that the chain stays put as often in a coupled step as in a step of the
        kernel alone.
    moves : tuple of two floats
        For the first and the second chain, the Bonferroni-corrected p-value of two-sample
        Kolmogorov-Smirnov tests that compare the moved positions of a coupled step with
        those of a step alone, along random directions and in the distance moved; NaN
        where a position is NaN.
    met : float
        The fraction of pairs that met: a coupling that never meets gives no estimate.
    stale : int
        Chains whose returned log-density or gradient differs from a fresh evaluation at
        their position; the engine reuses them (spec section 5.1).
    near_misses : int
        Pairs whose positions differ by at most 64 units in the last place in every
        coordinate: a coupling that recomputes Y where it should copy X's proposal
        leaves them, and the engine never sees such pairs meet (spec R-KER-3).
    """

    stay: tuple[float, float]
    moves: tuple[float, float]
    met: float
    stale: int
    near_misses: int

    def passed(self, alpha: float = 1e-3) -> bool:
        """Whether every p-value is at least ``alpha`` and pairs met, with no stale state or near miss.

        A p-value of NaN fails.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        bool
        """
        alpha = real("alpha", alpha, 0, 1)
        p_values = (*self.stay, *self.moves)
        return (
            bool(np.min(p_values) >= alpha)
            and self.met > 0
            and not (self.stale or self.near_misses)
        )


def check_coupling(
    kernel: Kernel,
    target: Target,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n: int = 10_000,
    rng: Seed = None,
) -> CouplingReport:
    """Check a kernel's coupling as the engine uses it (spec section 9).

    From ``n`` pairs, ``(x, y)`` or ``(y, x)`` in random rows, one coupled step is compared
    with one step of the kernel alone from the first points and one from the second: how
    often each chain stays put, and where it moves to. Swapped pairs in one batch catch a
    coupling that reads another row's state. It also counts the pairs that meet, the
    states whose cached log-density or gradient is stale, and the pairs that nearly
    meet. Custom kernels should pass it before use.

    Parameters
    ----------
    kernel : Kernel
        A tuned kernel.
    target : Target
    x, y : array_like, shape (d,)
        The starting points, close enough for the coupling to meet them in one step
        sometimes: for HMC, within about its ``meeting_scale``.
    n : int
        Pairs; at least 2. More detect smaller differences.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator

    Returns
    -------
    CouplingReport

    Notes
    -----
    Costs one coupled step and two steps of ``n`` chains, and an evaluation of the target
    at each of their ``4n`` new positions.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"x and y must be points of one shape (d,), not {x.shape} and {y.shape}")
    n = integer("n", n, minimum=2)
    seeds = root_seed(rng).spawn(4)
    generator = np.random.default_rng(seeds[3])
    swapped = (generator.random(n) < 0.5).astype(np.intp)
    starts = kernel.init_state(target, np.stack([x, y]))
    start_x, start_y = starts.take(swapped), starts.take(1 - swapped)
    new_x, new_y = kernel.coupled_step(target, start_x, start_y, random_draws(n, seeds[0]))
    alone = [
        kernel.step(target, start, random_draws(n, seed))[0]
        for start, seed in ((start_x, seeds[1]), (start_y, seeds[2]))
    ]
    # In one dimension every projection gives the same test.
    projections = PROJECTIONS if len(x) > 1 else 1
    directions = generator.standard_normal((projections, len(x)))
    # Each chain's steps are compared among the rows that start it at the same point, and
    # the two points' p-values Bonferroni-corrected.
    rows = (swapped == 0, swapped == 1)
    p_values = []
    for points, coupled, single in (((x, y), new_x.x, alone[0].x), ((y, x), new_y.x, alone[1].x)):
        tests = [
            _compare(point, coupled[group], single[group], directions)
            for point, group in zip(points, rows, strict=True)
        ]
        p_values.append(np.minimum(1.0, 2 * np.min(tests, axis=0)))
    stay, moves = np.transpose(p_values).tolist()
    gap = np.abs(new_x.x - new_y.x)
    ulp = np.spacing(np.maximum(np.abs(new_x.x), np.abs(new_y.x)))
    near = np.any(gap > 0, axis=1) & np.all(gap <= NEAR_MISS_ULPS * ulp, axis=1)
    return CouplingReport(
        (stay[0], stay[1]),
        (moves[0], moves[1]),
        float(np.mean(np.all(gap == 0, axis=1))),
        sum(_stale(kernel, target, state) for state in (new_x, new_y, *alone)),
        int(near.sum()),
    )


def _compare(
    point: FloatArray, coupled: FloatArray, single: FloatArray, directions: FloatArray
) -> tuple[float, float]:
    # For chains that start at point: Fisher's exact test of how often they stay put, and
    # Kolmogorov-Smirnov tests of their moves along each direction and of their lengths,
    # Bonferroni-corrected, which need two moves on a side. A NaN position gives NaN.
    stayed = [np.all(chain == point, axis=1) for chain in (coupled, single)]
    table = [[int(s.sum()), int(len(s) - s.sum())] for s in stayed]
    stay = float(stats.fisher_exact(table).pvalue)
    moved = [(chain - point)[~s] for chain, s in zip((coupled, single), stayed, strict=True)]
    if min(len(m) for m in moved) < 2:
        return stay, 1.0
    samples = [(moved[0] @ direction, moved[1] @ direction) for direction in directions]
    samples.append((np.linalg.norm(moved[0], axis=1), np.linalg.norm(moved[1], axis=1)))
    smallest = np.min([stats.ks_2samp(a, b).pvalue for a, b in samples])
    return stay, float(np.minimum(1.0, len(samples) * smallest))


def _stale(kernel: Kernel, target: Target, state: State) -> int:
    # Chains whose cache disagrees with a fresh evaluation beyond rounding: a vectorised
    # target can change the last bits between batches (spec section 6.1).
    fresh = kernel.init_state(target, state.x)
    close = np.isclose(state.log_prob, fresh.log_prob, rtol=1e-9, atol=0)
    if state.grad is not None and fresh.grad is not None:
        error = np.linalg.norm(state.grad - fresh.grad, axis=1)
        close &= error <= 1e-9 * np.linalg.norm(fresh.grad, axis=1)
    return int(np.sum(~close))


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
        Served by :meth:`telescopic.kernels.Draws.generator`.

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
    n_rows = integer("n_rows", n_rows, minimum=1)
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
