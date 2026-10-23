"""The engine: replicates of coupled chains, run in lockstep lanes with refill."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import Streams, replicate_generator
from telescopic._types import BoolArray, Distribution, FloatArray, IntArray
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.initial import draw
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target


@dataclass
class Replicates:
    r"""The outputs of a set of replicates, sorted by replicate index (spec section 6.2).

    Attributes
    ----------
    index : ndarray of int, shape (R,)
        Replicate indices.
    meeting_time : ndarray of float, shape (R,)
        :math:`\tau = \inf\{t \ge L : X_t = Y_{t-L}\}`, or ``inf`` for a censored pair.
    cost : ndarray of int, shape (R,)
        Kernel applications, counting a coupled step before the meeting as two.
    atoms, weights, owner : ndarrays, shapes (N, d), (N,), (N,)
        The signed measures of the replicates that met (spec R-EST-6): the atoms, their
        weights (each replicate's sum to 1), and the index of the replicate each atom
        belongs to. Sorted by owner.
    distance_owner, distance_time, distance : ndarrays, shape (M,)
        :math:`\lVert X_t - Y_{t-L}\rVert` for each pair, at each time :math:`L \le t < \tau`
        (to the cap for a censored pair).
    zero_density : ndarray of int
        The replicates whose :math:`X_0` or :math:`Y_0` has zero target density.
    requested : ndarray of int
        The indices the run was asked for, sorted: ``index`` lacks those of replicates
        that an interrupted run never started (spec section 6.5).
    n_log_prob, n_grad : int
        Evaluations of the log-density and its gradient (spec R-EST-7).
    seconds : float
        Wall time of the run.
    paths : dict of int to (ndarray, ndarray)
        With ``keep_paths``: the paths of :math:`X` and :math:`Y` of each replicate,
        shapes ``(T + 1, d)`` and ``(T_Y + 1, d)``, where :math:`Y` stops at the meeting.
    """

    index: IntArray
    meeting_time: FloatArray
    cost: IntArray
    atoms: FloatArray
    weights: FloatArray
    owner: IntArray
    distance_owner: IntArray
    distance_time: IntArray
    distance: FloatArray
    zero_density: IntArray
    requested: IntArray
    n_log_prob: int = 0
    n_grad: int = 0
    seconds: float = 0.0
    paths: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)


class Status(NamedTuple):
    """The progress of a run, given to its monitor after every iteration.

    Attributes
    ----------
    started, finished : int
        Replicates started and finished so far.
    slowest : int
        Index of the unmet replicate that has run longest, or -1 if every pair has met.
    t : int
        Its time :math:`t`, or 0.
    """

    started: int
    finished: int
    slowest: int
    t: int


def run(
    target: Target,
    kernel: Kernel,
    initial: Distribution,
    indices: Iterable[int],
    *,
    root: SeedSequence,
    namespace: int,
    k: int,
    m: int,
    lag: int,
    thin: int = 1,
    max_iter: int | None = None,
    lanes: int = 64,
    store: bool = True,
    keep_paths: bool = False,
    monitor: Callable[[Status], bool] | None = None,
) -> Replicates:
    r"""Run the replicates ``indices`` of the estimator :math:`H_S` (spec sections 3.1, 6.2).

    Each replicate draws :math:`X_0` and then :math:`Y_0` from ``initial`` with its own
    stream (spec section 6.1), runs :math:`X` alone for ``lag`` steps, then the coupled
    kernel until :math:`X_t = Y_{t-L}` bitwise, then :math:`X` alone until :math:`t = m`.
    A replicate's output depends only on ``root``, ``namespace``, its index and the
    arguments, never on the other replicates or on ``lanes``. The engine emits no
    warning: it records what its caller needs to warn about.

    Parameters
    ----------
    target : Target
    kernel : Kernel
        A tuned kernel.
    initial : object with ``rvs(random_state=generator)``
    indices : iterable of int
        Replicate indices, started in this order; at least one.
    root : numpy.random.SeedSequence
        Root seed of the run.
    namespace : int
        Stream namespace: ``telescopic._random.MAIN``, ``TUNING`` or ``PILOT``.
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        Average every ``thin``-th time from ``k`` (spec R-EST-5).
    max_iter : int, optional
        Censor a pair still unmet when :math:`t` reaches ``max_iter``, which must exceed
        ``max(lag, m)``. ``None`` runs every pair until it meets.
    lanes : int
        Number of replicates advanced together.
    store : bool
        Whether to store the signed measures and distances; meeting-time pilots do not.
    keep_paths : bool
        Whether to keep the paths of both chains, for inspection.
    monitor : callable, optional
        Called with a :class:`Status` after every iteration. Once it returns true, no
        new replicate starts and the ones in progress finish, so the finished indices
        are those started (spec section 6.5).

    Returns
    -------
    Replicates

    Notes
    -----
    Memory: the stored atoms take about :math:`R\,(|S| + 2\,\overline{\max(0, \tau - k - L)})
    (8d + 16)` bytes (spec R-EXEC-7).
    """
    check_settings(k, m, lag, thin, max_iter)
    if lanes < 1:
        raise ValueError(f"lanes must be at least 1, not {lanes}")
    requested = np.fromiter(indices, dtype=np.int64)
    if not len(requested):
        raise ValueError("indices must not be empty")
    queue = iter(requested.tolist())
    first = next(queue)
    evaluations, start = (target.n_log_prob, target.n_grad), time.perf_counter()
    head = _Start(first, initial, root, namespace, None)
    ndim = len(head.x0)
    starts = itertools.chain([head], (_Start(i, initial, root, namespace, ndim) for i in queue))
    batch = _Batch(target, kernel, ndim, lanes, k, m, lag, thin, max_iter, store, keep_paths)
    batch.fill(np.arange(lanes), starts)
    stopping = False
    while batch.active.any():
        batch.advance()
        free = batch.finish()
        stopping = (monitor is not None and monitor(batch.status())) or stopping
        if not stopping:
            batch.fill(free, starts)
    return batch.result(
        np.sort(requested),
        target.n_log_prob - evaluations[0],
        target.n_grad - evaluations[1],
        start,
    )


def check_settings(k: int, m: int, lag: int, thin: int, max_iter: int | None) -> None:
    r"""Raise ``ValueError`` unless the settings of a run are valid (spec section 3.2).

    Parameters
    ----------
    k, m, lag, thin : int
        :math:`0 \le k \le m`, :math:`L \ge 1` and ``thin`` :math:`\ge 1`.
    max_iter : int, optional
        Above :math:`\max(L, m)`.
    """
    if not 0 <= k <= m:
        raise ValueError(f"k and m must satisfy 0 <= k <= m, not k={k}, m={m}")
    if lag < 1 or thin < 1:
        raise ValueError(f"lag and thin must be at least 1, not lag={lag}, thin={thin}")
    if max_iter is not None and max_iter <= max(lag, m):
        raise ValueError(f"max_iter must exceed max(lag, m) = {max(lag, m)}, not {max_iter}")


def merge(parts: Sequence[Replicates]) -> Replicates:
    """Join the outputs of disjoint sets of replicates, sorted by replicate index.

    The requested indices are the union of the parts' requested indices.

    Parameters
    ----------
    parts : sequence of Replicates

    Returns
    -------
    Replicates
    """
    groups = {
        "index": ("index", "meeting_time", "cost"),
        "owner": ("owner", "atoms", "weights"),
        "distance_owner": ("distance_owner", "distance_time", "distance"),
        "zero_density": ("zero_density",),
    }
    index = np.concatenate([part.index for part in parts])
    if len(np.unique(index)) != len(index):
        raise ValueError("the parts share replicate indices; each replicate must run once")
    columns: dict[str, Any] = {}
    for key, names in groups.items():
        order = np.argsort(np.concatenate([getattr(part, key) for part in parts]), kind="stable")
        for name in names:
            columns[name] = np.concatenate([getattr(part, name) for part in parts])[order]
    return Replicates(
        **columns,
        requested=np.unique(np.concatenate([part.requested for part in parts])),
        n_log_prob=sum(part.n_log_prob for part in parts),
        n_grad=sum(part.n_grad for part in parts),
        seconds=sum(part.seconds for part in parts),
        paths={index: path for part in parts for index, path in part.paths.items()},
    )


class _Start:
    # A replicate's generator and initial states, drawn X_0 first (spec section 6.1).
    def __init__(
        self,
        index: int,
        initial: Distribution,
        root: SeedSequence,
        namespace: int,
        ndim: int | None,
    ) -> None:
        self.index = index
        self.generator: Generator = replicate_generator(root, namespace, index)
        self.x0 = draw(initial, self.generator, ndim)
        self.y0 = draw(initial, self.generator, len(self.x0))


class _Batch:
    # The lanes of a lockstep batch, and the outputs of the replicates they have run.
    def __init__(
        self,
        target: Target,
        kernel: Kernel,
        ndim: int,
        lanes: int,
        k: int,
        m: int,
        lag: int,
        thin: int,
        max_iter: int | None,
        store: bool,
        keep_paths: bool,
    ) -> None:
        self.target, self.kernel = target, kernel
        self.k, self.m, self.lag, self.thin, self.max_iter = k, m, lag, thin, max_iter
        self.store, self.keep_paths = store, keep_paths
        self.weight = 1.0 / len(averaging_times(k, m, thin))
        self.streams = Streams(lanes)
        self.replicate = np.full(lanes, -1, dtype=np.int64)
        self.t = np.zeros(lanes, dtype=np.int64)
        self.tau = np.full(lanes, -1, dtype=np.int64)
        self.x = _empty_state(lanes, ndim, kernel.needs_grad)
        self.y = _empty_state(lanes, ndim, kernel.needs_grad)
        integers, reals = np.zeros(0, dtype=np.int64), np.zeros(0)
        self.finished = _Chunks(integers, integers, integers)
        self.atoms = _Chunks(integers, np.zeros((0, ndim)), reals)
        self.distances = _Chunks(integers, integers, reals)
        self.zero_density: list[int] = []
        self.paths: dict[int, tuple[list[FloatArray], list[FloatArray]]] = {}
        self.n_started = self.n_finished = 0

    @property
    def active(self) -> BoolArray:
        active: BoolArray = self.replicate >= 0
        return active

    def fill(self, free: IntArray, starts: Iterator[_Start]) -> None:
        starting = list(itertools.islice(starts, len(free)))
        if not starting:
            return
        lanes = free[: len(starting)]
        states = self.kernel.init_state(
            self.target, np.array([p for s in starting for p in (s.x0, s.y0)])
        )
        self.x.put(lanes, states.take(slice(0, None, 2)))
        self.y.put(lanes, states.take(slice(1, None, 2)))
        outside = np.isneginf(states.log_prob).reshape(-1, 2).any(axis=1)
        self.zero_density += [s.index for s, bad in zip(starting, outside, strict=True) if bad]
        for lane, start in zip(lanes.tolist(), starting, strict=True):
            self.streams.assign(lane, start.generator)
            if self.keep_paths:
                self.paths[start.index] = ([start.x0], [start.y0])
        self.replicate[lanes] = [s.index for s in starting]
        self.t[lanes] = 0
        self.tau[lanes] = -1
        self.n_started += len(starting)
        self._record(lanes, np.zeros(0, dtype=np.int64))

    def advance(self) -> None:
        # One iteration: X alone in the lagging and met lanes, the coupled kernel in the
        # others, then the exact comparison of R-EST-2 for every unmet lane past the lag.
        active, unmet = self.active, self.tau < 0
        solo = (active & ((self.t < self.lag) | ~unmet)).nonzero()[0]
        coupled = (active & unmet & (self.t >= self.lag)).nonzero()[0]
        if len(solo):
            moved, _ = self.kernel.step(self.target, self.x.take(solo), self.streams.view(solo))
            self.x.put(solo, moved)
        if len(coupled):
            new_x, new_y = self.kernel.coupled_step(
                self.target, self.x.take(coupled), self.y.take(coupled), self.streams.view(coupled)
            )
            self.x.put(coupled, new_x)
            self.y.put(coupled, new_y)
        self.t += 1  # idle lanes' clocks are never read
        compared = (active & unmet & (self.t >= self.lag)).nonzero()[0]
        met = np.all(self.x.x[compared] == self.y.x[compared], axis=1)
        self.tau[compared[met]] = self.t[compared[met]]
        stepped = active.nonzero()[0]
        if self.keep_paths:
            for lane in stepped.tolist():
                self.paths[int(self.replicate[lane])][0].append(self.x.x[lane].copy())
            for lane in coupled.tolist():
                self.paths[int(self.replicate[lane])][1].append(self.y.x[lane].copy())
        self._record(stepped, compared[~met])

    def finish(self) -> IntArray:
        # Frees the lanes of the pairs that met and reached m, and of censored pairs.
        done = (self.tau >= 0) & (self.t >= self.m)
        if self.max_iter is not None:
            done |= (self.tau < 0) & (self.t >= self.max_iter)
        lanes = (self.active & done).nonzero()[0]
        if len(lanes):
            t, tau = self.t[lanes], self.tau[lanes]
            # X ran t steps and Y the coupled ones, from L to tau or, if censored, to t.
            cost = t + np.where(tau >= 0, tau, t) - self.lag
            self.finished.add(self.replicate[lanes], tau, cost)
            self.replicate[lanes] = -1
            self.n_finished += len(lanes)
        return lanes

    def status(self) -> Status:
        unmet = self.active & (self.tau < 0)
        if not unmet.any():
            return Status(self.n_started, self.n_finished, -1, 0)
        lane = int(np.argmax(np.where(unmet, self.t, -1)))
        return Status(
            self.n_started, self.n_finished, int(self.replicate[lane]), int(self.t[lane])
        )

    def result(
        self, requested: IntArray, n_log_prob: int, n_grad: int, start: float
    ) -> Replicates:
        # The seconds since start include this join, a large part of a run with many atoms.
        index, tau, cost = self.finished.join()
        owner, atoms, weights = self.atoms.join(keep=index[tau >= 0])
        distance_owner, distance_time, distance = self.distances.join()
        return Replicates(
            index=index,
            meeting_time=np.where(tau >= 0, tau, np.inf),
            cost=cost,
            atoms=atoms,
            weights=weights,
            owner=owner,
            distance_owner=distance_owner,
            distance_time=distance_time,
            distance=distance,
            zero_density=np.sort(np.array(self.zero_density, dtype=np.int64)),
            requested=requested,
            n_log_prob=n_log_prob,
            n_grad=n_grad,
            seconds=time.perf_counter() - start,
            paths={i: (np.array(x), np.array(y)) for i, (x, y) in self.paths.items()},
        )

    def _record(self, stepped: IntArray, unmet: IntArray) -> None:
        # The averaged atoms X_t, t in S, of the lanes that moved, and the bias-correction
        # atoms and distances of the unmet pairs (spec R-EST-6, section 3.6). Kinds without
        # rows are skipped: at one lane most iterations lack one, and its calls add up.
        if not self.store:
            return
        t = self.t[stepped]
        averaged = stepped[(t >= self.k) & (t <= self.m) & ((t - self.k) % self.thin == 0)]
        if len(averaged):
            self.atoms.add(
                self.replicate[averaged], self.x.x[averaged], np.full(len(averaged), self.weight)
            )
        if not len(unmet):
            return
        coefficient = correction_coefficients(self.t[unmet], self.k, self.m, self.lag, self.thin)
        positive = coefficient > 0
        if positive.any():
            corrected = unmet[positive]
            for chain, sign in ((self.x, 1.0), (self.y, -1.0)):
                self.atoms.add(
                    self.replicate[corrected], chain.x[corrected], sign * coefficient[positive]
                )
        self.distances.add(
            self.replicate[unmet],
            self.t[unmet],
            np.linalg.norm(self.x.x[unmet] - self.y.x[unmet], axis=1),
        )


class _Chunks:
    # Columns appended iteration by iteration, the owning replicate first, then joined and
    # sorted by owner once. The empty columns given at construction fix shapes and dtypes.
    # Every 256 chunks merge into one block: at one lane a chunk holds one row, and its
    # arrays' overhead would cost several times the row.
    def __init__(self, *empty: npt.NDArray[Any]) -> None:
        self.blocks: list[tuple[npt.NDArray[Any], ...]] = [empty]
        self.chunks: list[tuple[npt.NDArray[Any], ...]] = []

    def add(self, *columns: npt.NDArray[Any]) -> None:
        self.chunks.append(columns)
        if len(self.chunks) == 256:
            self.blocks.append(_concatenate(self.chunks))
            self.chunks = []

    def join(self, keep: IntArray | None = None) -> list[npt.NDArray[Any]]:
        # The rows whose owners are in keep (all by default), gathered once.
        columns = _concatenate(self.blocks + self.chunks)
        order = np.argsort(columns[0], kind="stable")
        if keep is not None:
            order = order[np.isin(columns[0][order], keep)]
        return [column[order] for column in columns]


def _concatenate(chunks: list[tuple[npt.NDArray[Any], ...]]) -> tuple[npt.NDArray[Any], ...]:
    return tuple(np.concatenate(parts) for parts in zip(*chunks, strict=True))


def _empty_state(lanes: int, ndim: int, with_grad: bool) -> State:
    return State(
        np.zeros((lanes, ndim)), np.zeros(lanes), np.zeros((lanes, ndim)) if with_grad else None
    )
