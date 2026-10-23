"""Helpers for testing kernels and couplings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import Draws, FloatArray, Streams


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


def random_draws(n_rows: int, rng: int | SeedSequence | Generator | None = None) -> Draws:
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
