"""The engine: replicates of coupled chains, run in lockstep lanes with refill."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import Streams, replicate_generator
from telescopic._types import BoolArray, Distribution, FloatArray, IntArray
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.initial import draw
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning, InputWarning


@dataclass
class Replicates:
    r"""The outputs of a set of replicates, sorted by replicate index (spec section 6.2).

    Attributes
    ----------
    index : ndarray of int, shape (R,)
        Replicate indices.
    meeting_time : ndarray of int, shape (R,)
        :math:`\tau = \inf\{t \ge L : X_t = Y_{t-L}\}`, or -1 for a censored pair.
    cost : ndarray of int, shape (R,)
        Kernel applications, counting a coupled step before the meeting as two.
    atoms, weights, owner, correction : ndarrays, shapes (N, d), (N,), (N,), (N,)
        The signed measures of the replicates that met (spec R-EST-6): the atoms, their
        weights (each replicate's sum to 1), the index of the replicate each atom belongs
        to, and whether it is a bias-correction atom. Sorted by owner.
    distance_owner, distance_time, distance : ndarrays, shape (M,)
        :math:`\lVert X_t - Y_{t-L}\rVert` for each pair, at each time :math:`L \le t < \tau`
        (to the cap for a censored pair).
    paths : dict of int to (ndarray, ndarray)
        With ``keep_paths``: the paths of :math:`X` and :math:`Y` of each replicate,
        shapes ``(T + 1, d)`` and ``(T_Y + 1, d)``, where :math:`Y` stops at the meeting.
    """

    index: IntArray
    meeting_time: IntArray
    cost: IntArray
    atoms: FloatArray
    weights: FloatArray
    owner: IntArray
    correction: BoolArray
    distance_owner: IntArray
    distance_time: IntArray
    distance: FloatArray
    paths: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)


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
    slow_after: int | None = None,
    stop: Callable[[], bool] | None = None,
) -> Replicates:
    r"""Run the replicates ``indices`` of the estimator :math:`H_S` (spec sections 3.1, 6.2).

    Each replicate draws :math:`X_0` and then :math:`Y_0` from ``initial`` with its own
    stream (spec section 6.1), runs :math:`X` alone for ``lag`` steps, then the coupled
    kernel until :math:`X_t = Y_{t-L}` bitwise, then :math:`X` alone until :math:`t = m`.
    A replicate's output depends only on ``root``, ``namespace``, its index and the
    arguments, never on the other replicates or on ``lanes``.

    Parameters
    ----------
    target : Target
    kernel : Kernel
        A tuned kernel.
    initial : object with rvs(random_state=generator)
    indices : iterable of int
        Replicate indices, started in this order.
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
    slow_after : int, optional
        Warn once when a pair is still unmet after this time; defaults to
        ``max(10_000, 10 * max(lag, m))`` (spec section 6.3).
    stop : callable, optional
        Polled every iteration; once it returns true, no new replicate starts and the
        ones in progress finish, so the completed indices are those started.

    Returns
    -------
    Replicates

    Notes
    -----
    Memory: the stored atoms take about :math:`R\,(|S| + 2\,\overline{\max(0, \tau - k - L)})
    (8d + 16)` bytes (spec R-EXEC-7).
    """
    if not 0 <= k <= m:
        raise ValueError(f"k and m must satisfy 0 <= k <= m, not k={k}, m={m}")
    if lag < 1 or thin < 1:
        raise ValueError(f"lag and thin must be at least 1, not lag={lag}, thin={thin}")
    if max_iter is not None and max_iter <= max(lag, m):
        raise ValueError(f"max_iter must exceed max(lag, m) = {max(lag, m)}, not {max_iter}")
    if lanes < 1:
        raise ValueError(f"lanes must be at least 1, not {lanes}")
    queue = iter(indices)
    first = next(queue, None)
    if first is None:
        raise ValueError("indices must not be empty")
    head = _Start(first, initial, root, namespace, None)
    ndim = len(head.x0)
    starts = itertools.chain([head], (_Start(i, initial, root, namespace, ndim) for i in queue))
    batch = _Batch(target, kernel, ndim, lanes, k, m, lag, thin, max_iter, store, keep_paths)
    batch.fill(np.arange(lanes), starts)
    slow_after = max(10_000, 10 * max(lag, m)) if slow_after is None else slow_after
    stopping = warned = False
    while batch.active.any():
        stopping = stopping or (stop is not None and stop())
        batch.advance()
        if not warned and (slow := batch.slow(slow_after)) is not None:
            warned = True
            warnings.warn(
                f"replicate {slow} has not met after {slow_after} iterations; "
                "the meeting times may be heavy-tailed",
                ConvergenceWarning,
                stacklevel=2,
            )
        if len(free := batch.finish()):
            batch.fill(free, iter(()) if stopping else starts)
    return batch.result()


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
        self.target, self.kernel, self.ndim = target, kernel, ndim
        self.k, self.m, self.lag, self.thin, self.max_iter = k, m, lag, thin, max_iter
        self.store, self.keep_paths = store, keep_paths
        self.weight = 1.0 / len(averaging_times(k, m, thin))
        self.streams = Streams(lanes)
        self.replicate = np.full(lanes, -1, dtype=np.int64)
        self.t = np.zeros(lanes, dtype=np.int64)
        self.tau = np.full(lanes, -1, dtype=np.int64)
        self.cost = np.zeros(lanes, dtype=np.int64)
        self.x = _empty_state(lanes, ndim, kernel.needs_grad)
        self.y = _empty_state(lanes, ndim, kernel.needs_grad)
        self.finished: list[tuple[int, int, int]] = []
        self.atoms = _Chunks()
        self.distances = _Chunks()
        self.paths: dict[int, tuple[list[FloatArray], list[FloatArray]]] = {}
        self.warned_zero = False

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
        if not self.warned_zero and np.isneginf(states.log_prob).any():
            self.warned_zero = True
            warnings.warn(
                "an initial draw has zero target density; the chains start outside the support",
                InputWarning,
                stacklevel=3,
            )
        self.x.put(lanes, states.take(slice(0, None, 2)))
        self.y.put(lanes, states.take(slice(1, None, 2)))
        for lane, start in zip(lanes.tolist(), starting, strict=True):
            self.streams.assign(lane, start.generator)
            if self.keep_paths:
                self.paths[start.index] = ([start.x0], [start.y0])
        self.replicate[lanes] = [s.index for s in starting]
        self.t[lanes] = 0
        self.tau[lanes] = -1
        self.cost[lanes] = 0
        self._record(lanes, np.zeros(0, dtype=np.intp))

    def advance(self) -> None:
        # One iteration: X alone in the lagging and met lanes, the coupled kernel in the
        # others, then the exact comparison of R-EST-2 for every unmet lane past the lag.
        active, unmet = self.active, self.tau < 0
        solo = np.flatnonzero(active & ((self.t < self.lag) | ~unmet))
        coupled = np.flatnonzero(active & unmet & (self.t >= self.lag))
        if len(solo):
            moved, _ = self.kernel.step(self.target, self.x.take(solo), self.streams.view(solo))
            self.x.put(solo, moved)
        if len(coupled):
            new_x, new_y = self.kernel.coupled_step(
                self.target, self.x.take(coupled), self.y.take(coupled), self.streams.view(coupled)
            )
            self.x.put(coupled, new_x)
            self.y.put(coupled, new_y)
        self.t[active] += 1
        self.cost[solo] += 1
        self.cost[coupled] += 2
        compared = np.flatnonzero(active & unmet & (self.t >= self.lag))
        met = np.all(self.x.x[compared] == self.y.x[compared], axis=1)
        self.tau[compared[met]] = self.t[compared[met]]
        if self.keep_paths:
            for lane in np.flatnonzero(active).tolist():
                self.paths[int(self.replicate[lane])][0].append(self.x.x[lane].copy())
            for lane in coupled.tolist():
                self.paths[int(self.replicate[lane])][1].append(self.y.x[lane].copy())
        self._record(np.flatnonzero(active), compared[~met])

    def slow(self, after: int) -> int | None:
        slow = np.flatnonzero(self.active & (self.tau < 0) & (self.t > after))
        return int(self.replicate[slow[0]]) if len(slow) else None

    def finish(self) -> IntArray:
        # Frees the lanes of the pairs that met and reached m, and of censored pairs.
        done = (self.tau >= 0) & (self.t >= self.m)
        if self.max_iter is not None:
            done |= (self.tau < 0) & (self.t >= self.max_iter)
        lanes = np.flatnonzero(self.active & done)
        self.finished.extend(
            zip(
                self.replicate[lanes].tolist(),
                self.tau[lanes].tolist(),
                self.cost[lanes].tolist(),
                strict=True,
            )
        )
        self.replicate[lanes] = -1
        return lanes

    def result(self) -> Replicates:
        index, meeting_time, cost = (
            np.array(column, dtype=np.int64) for column in zip(*sorted(self.finished), strict=True)
        )
        owner, (atoms, weights, correction) = self.atoms.join(
            [np.zeros((0, self.ndim)), np.zeros(0), np.zeros(0, dtype=bool)]
        )
        met = np.isin(owner, index[meeting_time >= 0])
        distance_owner, (distance_time, distance) = self.distances.join(
            [np.zeros(0, dtype=np.int64), np.zeros(0)]
        )
        return Replicates(
            index=index,
            meeting_time=meeting_time,
            cost=cost,
            atoms=atoms[met],
            weights=weights[met],
            owner=owner[met],
            correction=correction[met],
            distance_owner=distance_owner,
            distance_time=distance_time,
            distance=distance,
            paths={i: (np.array(x), np.array(y)) for i, (x, y) in self.paths.items()},
        )

    def _record(self, stepped: IntArray, unmet: IntArray) -> None:
        # The averaged atoms X_t, t in S, of the lanes that moved, and the bias-correction
        # atoms and distances of the unmet pairs (spec R-EST-6, section 3.6).
        if not self.store:
            return
        t = self.t[stepped]
        averaged = stepped[(t >= self.k) & (t <= self.m) & ((t - self.k) % self.thin == 0)]
        self.atoms.add(
            self.replicate[averaged],
            self.x.x[averaged],
            np.full(len(averaged), self.weight),
            np.zeros(len(averaged), dtype=bool),
        )
        coefficient = correction_coefficients(self.t[unmet], self.k, self.m, self.lag, self.thin)
        corrected = unmet[coefficient > 0]
        for chain, sign in ((self.x, 1.0), (self.y, -1.0)):
            self.atoms.add(
                self.replicate[corrected],
                chain.x[corrected],
                sign * coefficient[coefficient > 0],
                np.ones(len(corrected), dtype=bool),
            )
        self.distances.add(
            self.replicate[unmet],
            self.t[unmet],
            np.linalg.norm(self.x.x[unmet] - self.y.x[unmet], axis=1),
        )


class _Chunks:
    # Arrays appended iteration by iteration with their owners, joined and sorted by
    # owner once at the end.
    def __init__(self) -> None:
        self.owners: list[IntArray] = []
        self.columns: list[list[npt.NDArray[Any]]] = []

    def add(self, owner: IntArray, *columns: npt.NDArray[Any]) -> None:
        if len(owner):
            self.owners.append(owner)
            self.columns.append(list(columns))

    def join(self, empty: list[npt.NDArray[Any]]) -> tuple[IntArray, list[npt.NDArray[Any]]]:
        if not self.owners:
            return np.zeros(0, dtype=np.int64), empty
        owner = np.concatenate(self.owners)
        order = np.argsort(owner, kind="stable")
        columns = [np.concatenate(parts)[order] for parts in zip(*self.columns, strict=True)]
        return owner[order], columns


def _empty_state(lanes: int, ndim: int, with_grad: bool) -> State:
    return State(
        np.zeros((lanes, ndim)), np.zeros(lanes), np.zeros((lanes, ndim)) if with_grad else None
    )
