"""Helpers for testing kernels and couplings."""

from __future__ import annotations

import numpy.typing as npt
from numpy.random import Generator

from telescopic._random import Draws, Streams


def fixed_draws(
    normals: npt.ArrayLike | None = None,
    uniforms: npt.ArrayLike | None = None,
    generators: list[Generator] | None = None,
) -> Draws:
    """Draws that replay prescribed random numbers, for tests that inject randomness.

    Parameters
    ----------
    normals, uniforms : array_like, shape (n, k), optional
        Row ``i`` is consumed left to right by row ``i`` of every request.
    generators : list of numpy.random.Generator, optional
        Served by :meth:`Draws.generator`.

    Returns
    -------
    Draws
        Draws for ``n`` rows; a request beyond the prescribed numbers raises ``IndexError``.
    """
    streams = Streams.from_arrays(normals, uniforms, generators)
    return streams.view(range(len(streams._generators)))
