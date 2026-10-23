"""Random streams: one generator per replicate, buffered for lockstep batches."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import numpy.typing as npt
from numpy.random import PCG64DXSM, Generator, SeedSequence

from telescopic._types import FloatArray, IndexArray, Seed

# Namespaces of replicate streams (spec section 6.1).
MAIN = 0
TUNING = 1
PILOT = 2

# The two kinds of buffered numbers, each read from its own child generator.
NORMAL = 0
UNIFORM = 1

# Replicate streams live under this element of the root's spawn key, far from the small
# integers that SeedSequence.spawn gives the caller's own children ("tel" in ASCII).
_BRANCH = 0x74656C


def root_seed(rng: Seed) -> SeedSequence:
    # The root seed of a run from a SPEC 7 rng (spec section 6.1). A seed sequence is
    # copied, so that a record of it does not change when its owner spawns from it; a
    # generator gives the seed of four integers drawn from it, so the run depends only on
    # its state.
    if isinstance(rng, SeedSequence):
        return SeedSequence(**rng.state)
    if isinstance(rng, Generator):
        return SeedSequence(rng.integers(2**63, size=4).tolist())
    if rng is None:
        return SeedSequence()
    if isinstance(rng, Integral) and not isinstance(rng, bool):
        return SeedSequence(int(rng))
    raise TypeError(
        "rng must be None, an integer, a numpy.random.SeedSequence or a "
        f"numpy.random.Generator, not {type(rng).__name__}"
    )


def namespace_seed(root: SeedSequence, namespace: int) -> SeedSequence:
    # The seed of a pilot's engine runs, whose streams then never meet the main run's.
    return SeedSequence(
        root.entropy, spawn_key=(*root.spawn_key, _BRANCH, namespace), pool_size=root.pool_size
    )


def replicate_generator(root: SeedSequence, namespace: int, index: int) -> Generator:
    seed = SeedSequence(
        root.entropy,
        spawn_key=(*root.spawn_key, _BRANCH, namespace, index),
        pool_size=root.pool_size,
    )
    return Generator(PCG64DXSM(seed))


class Streams:
    # Buffered random numbers for the lanes of a lockstep batch (spec section 6.1). The
    # replicate in a lane reads its normals and its uniforms in order from two children
    # of its generator. A request takes the lane's next unread numbers; when too few
    # remain, the buffer keeps them and is topped up, so a replicate's numbers depend
    # only on its generator and its own requests, never on the other lanes, the batch
    # size or the buffer width (width: values of each kind per lane; a request for more
    # than an eighth of a buffer widens it).

    def __init__(self, n_lanes: int, width: int = 512) -> None:
        self._buffers = [np.empty((n_lanes, width)) for _ in (NORMAL, UNIFORM)]
        self._cursors = [np.full(n_lanes, width, dtype=np.intp) for _ in (NORMAL, UNIFORM)]
        self._generators: list[Generator | None] = [None] * n_lanes
        self._sources: list[list[Generator]] = [[] for _ in range(n_lanes)]

    def assign(self, lane: int, generator: Generator) -> None:
        # A new replicate enters the lane; its buffers read generator.spawn(2).
        self._generators[lane] = generator
        self._sources[lane] = generator.spawn(2)
        for buffer, cursor in zip(self._buffers, self._cursors, strict=True):
            cursor[lane] = buffer.shape[1]

    def view(self, lanes: npt.ArrayLike) -> Draws:
        return Draws(self, np.asarray(lanes, dtype=np.intp))

    def _take(self, kind: int, lanes: IndexArray, size: int) -> FloatArray:
        buffer, cursor = self._buffers[kind], self._cursors[kind]
        if 8 * size > buffer.shape[1]:
            # Widen on the left, so each lane's unread numbers stay at the end.
            pad = 8 * size - buffer.shape[1]
            buffer = self._buffers[kind] = np.pad(buffer, ((0, 0), (pad, 0)))
            cursor += pad
        width = buffer.shape[1]
        start = cursor[lanes]
        short = start > width - size
        if short.any():
            for lane in lanes[short].tolist():
                unread = width - cursor[lane]
                buffer[lane, :unread] = buffer[lane, cursor[lane] :]
                self._fill(kind, lane, buffer[lane, unread:])
            start[short] = 0
        cursor[lanes] = start + size
        # windows[i, j] is buffer[i, j : j + size], so whole windows are gathered without
        # an (n, size) index array.
        windows: FloatArray = np.ndarray(
            (len(buffer), width - size + 1, size),
            buffer.dtype,
            buffer,
            strides=(buffer.strides[0], buffer.strides[1], buffer.strides[1]),
        )
        taken: FloatArray = windows[lanes, start]
        return taken

    def _fill(self, kind: int, lane: int, out: FloatArray) -> None:
        source = self._sources[lane][kind]
        if kind == NORMAL:
            source.standard_normal(out=out)
        else:
            source.random(out=out)

    def _generator(self, lane: int) -> Generator:
        generator = self._generators[lane]
        if generator is None:
            raise RuntimeError(f"lane {lane} has no generator")
        return generator


class Draws:
    """Random numbers for a set of chains: row ``i`` of every request belongs to chain ``i``.

    Kernels receive a :class:`Draws` and call it once per random quantity they need per
    step; every request returns one row per chain. In a run, each row reads the streams
    of its replicate (spec section 6.1). Draws are made by the engine, or by
    :func:`telescopic.testing.fixed_draws` and :func:`telescopic.testing.random_draws`
    in tests.
    """

    def __init__(self, streams: Streams, lanes: IndexArray) -> None:
        self._streams = streams
        self._lanes = lanes

    def __len__(self) -> int:
        return len(self._lanes)

    def normal(self, size: int) -> FloatArray:
        """Standard normal draws, shape ``(n, size)``."""
        return self._streams._take(NORMAL, self._lanes, size)

    def uniform(self) -> FloatArray:
        """Uniform draws on [0, 1), shape ``(n,)``."""
        return self._streams._take(UNIFORM, self._lanes, 1)[:, 0]

    def log_uniform(self) -> FloatArray:
        """Logarithms of uniform draws on [0, 1), shape ``(n,)``; ``-inf`` for a draw of 0."""
        with np.errstate(divide="ignore"):
            return np.log(self.uniform())

    def generator(self, row: int) -> Generator:
        """The generator of row ``row``, for draws that the buffers do not cover."""
        return self._streams._generator(int(self._lanes[row]))

    def subset(self, rows: npt.ArrayLike) -> Draws:
        """The draws of ``rows`` of this set; they share the rows' streams."""
        return Draws(self._streams, self._lanes[np.asarray(rows)])
