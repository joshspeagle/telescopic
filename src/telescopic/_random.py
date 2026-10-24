"""Random streams: one generator per replicate, buffered for lockstep batches."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import numpy.typing as npt
from numpy.random import PCG64DXSM, Generator, SeedSequence

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]

# Namespaces of replicate streams (spec section 6.1).
MAIN = 0
TUNING = 1
PILOT = 2

_UNIFORMS_PER_STEP = 4


def root_seed(rng: int | SeedSequence | Generator | None) -> SeedSequence:
    """Normalise a SPEC 7 ``rng`` argument to the root seed of a run.

    Parameters
    ----------
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        ``None`` draws fresh entropy from the operating system. A generator gives a
        new child of its seed sequence, so passing the same generator twice gives two
        different runs.

    Returns
    -------
    numpy.random.SeedSequence
        The root seed. Its ``entropy`` and ``spawn_key`` reproduce the run.
    """
    if isinstance(rng, SeedSequence):
        return rng
    if isinstance(rng, Generator):
        seed_seq = rng.bit_generator.seed_seq
        if isinstance(seed_seq, SeedSequence):
            return seed_seq.spawn(1)[0]
        return SeedSequence(rng.integers(2**63, size=4).tolist())
    if rng is None:
        return SeedSequence()
    if isinstance(rng, Integral) and not isinstance(rng, bool):
        return SeedSequence(int(rng))
    raise TypeError(
        "rng must be None, an integer, a numpy.random.SeedSequence or a "
        f"numpy.random.Generator, not {type(rng).__name__}"
    )


def replicate_generator(root: SeedSequence, namespace: int, index: int) -> Generator:
    """Return the generator of replicate ``index`` in ``namespace`` (spec R-EXEC-1)."""
    seed = SeedSequence(
        root.entropy, spawn_key=(*root.spawn_key, namespace, index), pool_size=root.pool_size
    )
    return Generator(PCG64DXSM(seed))


class Streams:
    """Buffered random numbers for the lanes of a lockstep batch.

    Each lane draws only from the generator of the replicate it holds. A request that
    the rest of a lane's buffer cannot satisfy refills the whole buffer and discards
    the rest, so the numbers a replicate receives depend only on its generator and on
    its own sequence of requests, never on the other lanes or on the batch size.

    Parameters
    ----------
    n_lanes : int
        Number of lanes.
    ndim : int
        Dimension of the states; the normal buffer holds ``block * ndim`` values.
    block : int
        Number of steps' worth of random numbers drawn per refill.
    """

    def __init__(self, n_lanes: int, ndim: int, block: int = 64) -> None:
        self._normals: FloatArray = np.empty((n_lanes, block * max(ndim, 1)))
        self._uniforms: FloatArray = np.empty((n_lanes, block * _UNIFORMS_PER_STEP))
        self._normal_cursor = np.full(n_lanes, self._normals.shape[1], dtype=np.intp)
        self._uniform_cursor = np.full(n_lanes, self._uniforms.shape[1], dtype=np.intp)
        self._generators: list[Generator | None] = [None] * n_lanes
        self._fixed = False

    @classmethod
    def from_arrays(
        cls,
        normals: npt.ArrayLike | None = None,
        uniforms: npt.ArrayLike | None = None,
        generators: list[Generator] | None = None,
    ) -> Streams:
        """Streams that replay prescribed numbers and raise once they run out.

        Row ``i`` of ``normals`` and ``uniforms`` is consumed left to right by lane
        ``i``; ``generators[i]`` serves :meth:`Draws.generator` for that lane.
        """
        given = [
            None if a is None else np.atleast_2d(np.array(a, dtype=np.float64))
            for a in (normals, uniforms)
        ]
        sizes = [len(a) for a in (*given, generators) if a is not None]
        if not sizes:
            raise ValueError("give at least one of normals, uniforms and generators")
        streams = cls(sizes[0], 1, block=0)
        if given[0] is not None:
            streams._normals = given[0]
        if given[1] is not None:
            streams._uniforms = given[1]
        streams._normal_cursor[:] = 0
        streams._uniform_cursor[:] = 0
        if generators is not None:
            streams._generators = list(generators)
        streams._fixed = True
        return streams

    def assign(self, lane: int, generator: Generator) -> None:
        """Give ``lane`` to a new replicate drawing from ``generator``."""
        self._generators[lane] = generator
        self._normal_cursor[lane] = self._normals.shape[1]
        self._uniform_cursor[lane] = self._uniforms.shape[1]

    def view(self, lanes: npt.ArrayLike) -> Draws:
        """Random numbers for ``lanes``, one row per lane."""
        return Draws(self, np.asarray(lanes, dtype=np.intp))

    def _take(self, normal: bool, lanes: IndexArray, size: int) -> FloatArray:
        buffer = self._normals if normal else self._uniforms
        cursor = self._normal_cursor if normal else self._uniform_cursor
        width = buffer.shape[1]
        if size > width and not self._fixed:
            # A request beyond one block bypasses the buffer.
            out = np.empty((len(lanes), size))
            for row, lane in enumerate(lanes):
                self._fill(normal, int(lane), out[row])
            return out
        for lane in lanes[cursor[lanes] + size > width]:
            if self._fixed:
                raise IndexError(f"the injected random numbers of lane {lane} are exhausted")
            self._fill(normal, int(lane), buffer[lane])
            cursor[lane] = 0
        taken: FloatArray = buffer[lanes[:, None], cursor[lanes, None] + np.arange(size)]
        cursor[lanes] += size
        return taken

    def _fill(self, normal: bool, lane: int, out: FloatArray) -> None:
        generator = self._generator(lane)
        if normal:
            generator.standard_normal(out=out)
        else:
            generator.random(out=out)

    def _generator(self, lane: int) -> Generator:
        generator = self._generators[lane]
        if generator is None:
            raise RuntimeError(f"lane {lane} has no generator")
        return generator


class Draws:
    """Random numbers for a set of lanes: row ``i`` of every request comes from lane ``i``.

    Kernels receive a :class:`Draws` and call it once per random quantity they need
    per step. Every request returns one row per lane.
    """

    def __init__(self, streams: Streams, lanes: IndexArray) -> None:
        self._streams = streams
        self._lanes = lanes

    def __len__(self) -> int:
        return len(self._lanes)

    def normal(self, size: int) -> FloatArray:
        """Standard normal draws, shape ``(n, size)``."""
        return self._streams._take(True, self._lanes, size)

    def uniform(self) -> FloatArray:
        """Uniform draws on [0, 1), shape ``(n,)``."""
        return self._streams._take(False, self._lanes, 1)[:, 0]

    def log_uniform(self) -> FloatArray:
        """Logarithms of uniform draws on [0, 1), shape ``(n,)``; ``-inf`` for a draw of 0."""
        with np.errstate(divide="ignore"):
            return np.log(self.uniform())

    def generator(self, row: int) -> Generator:
        """The generator of row ``row``, for draws that the buffers do not cover."""
        return self._streams._generator(int(self._lanes[row]))

    def subset(self, rows: npt.ArrayLike) -> Draws:
        """The draws of ``rows`` of this view; they share the lanes' streams."""
        return Draws(self._streams, self._lanes[np.asarray(rows)])
