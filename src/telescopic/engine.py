"""The engine: replicates of coupled chains, run in lockstep lanes with refill."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import MAIN, Streams, replicate_generator, root_seed
from telescopic._types import BoolArray, Distribution, FloatArray, IntArray, Seed, integer
from telescopic.estimators import correction_coefficients
from telescopic.initial import draw
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target

# Replicates advanced together by default.
LANES = 64


class Settings(NamedTuple):
    r"""The settings of a run (spec section 3.2).

    Attributes
    ----------
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        Average every ``thin``-th time from ``k`` (spec R-EST-5).
    max_iter : int or None
        The cap: a pair still unmet when :math:`t` reaches it is censored.
    store : bool
        Whether the run stores the signed measures and distances; meeting-time pilots do
        not.
    """

    k: int
    m: int
    lag: int
    thin: int = 1
    max_iter: int | None = None
    store: bool = True


@dataclass
class Replicates:
    r"""The outputs of a set of replicates, sorted by replicate index, and their run's settings.

    Spec section 6.2.

    Attributes
    ----------
    seed : numpy.random.SeedSequence
        The root seed of the run, from which each replicate's stream derives (spec
        section 6.1).
    settings : Settings
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

    seed: SeedSequence
    settings: Settings
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

    @property
    def stopped(self) -> FloatArray:
        """The time each pair stopped, shape (R,): its meeting time, or the cap if censored."""
        cap = self.settings.max_iter
        stopped: FloatArray = np.minimum(self.meeting_time, np.inf if cap is None else cap)
        return stopped


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
    indices: npt.ArrayLike,
    *,
    k: int,
    m: int,
    lag: int,
    thin: int = 1,
    max_iter: int | None = None,
    rng: Seed = None,
    lanes: int = LANES,
    store: bool = True,
    keep_paths: bool = False,
    monitor: Callable[[Status], bool] | None = None,
) -> Replicates:
    r"""Run the replicates ``indices`` of the estimator :math:`H_S` (spec sections 3.1, 6.2).

    Each replicate draws :math:`X_0` and then :math:`Y_0` from ``initial`` with its own
    stream (spec section 6.1), runs :math:`X` alone for ``lag`` steps, then the coupled
    kernel until :math:`X_t = Y_{t-L}` under ``==``, then :math:`X` alone until :math:`t = m`.
    A replicate's output depends only on the root seed, its index and the arguments,
    never on the other replicates or on ``lanes``. The engine emits no warning: it
    records what its caller needs to warn about.

    Parameters
    ----------
    target : Target
    kernel : Kernel
        A tuned kernel.
    initial : object with ``rvs(random_state=generator)``
    indices : array_like of int, shape (R,)
        Distinct replicate indices, at least one, started in this order.
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        Average every ``thin``-th time from ``k`` (spec R-EST-5).
    max_iter : int, optional
        Censor a pair still unmet when :math:`t` reaches ``max_iter``, which must exceed
        ``max(lag, m)``. ``None`` runs every pair until it meets.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root seed (spec section 6.1), recorded in the output. Runs of other indices
        with the same ``SeedSequence`` or integer can be merged.
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
    (8d + 16)` bytes, and the :math:`\tau - L` distances of each replicate 24 bytes each
    (spec R-EXEC-7).
    """
    settings = check_settings(Settings(k, m, lag, thin, max_iter, store))
    if integer("lanes", lanes) < 1:
        raise ValueError(f"lanes must be at least 1, not {lanes}")
    requested = np.asarray(indices)
    if (
        requested.ndim != 1
        or not len(requested)
        or not np.issubdtype(requested.dtype, np.integer)
        or requested.min() < 0
        or len(np.unique(requested)) != len(requested)
    ):
        raise ValueError("indices must be a non-empty sequence of distinct integers >= 0")
    requested = requested.astype(np.int64)
    seed = root_seed(rng)
    queue = iter(requested.tolist())
    first = next(queue)
    evaluations, start = (target.n_log_prob, target.n_grad), time.perf_counter()
    head = _Start(first, initial, seed, None)
    ndim = len(head.x0)
    starts = itertools.chain([head], (_Start(i, initial, seed, ndim) for i in queue))
    batch = _Batch(target, kernel, ndim, lanes, settings, keep_paths)
    batch.fill(np.arange(lanes), starts)
    stopping = False
    while batch.active.any():
        batch.advance()
        free = batch.finish()
        stopping = (monitor is not None and monitor(batch.status())) or stopping
        if not stopping:
            batch.fill(free, starts)
    return batch.result(
        seed,
        np.sort(requested),
        target.n_log_prob - evaluations[0],
        target.n_grad - evaluations[1],
        start,
    )


def check_settings(settings: Settings) -> Settings:
    r"""Check the settings of a run and return them with Python integers (spec section 3.2).

    They need integers :math:`0 \le k \le m`, :math:`L \ge 1` and ``thin`` :math:`\ge 1`,
    and an integer ``max_iter`` above :math:`\max(L, m)` or ``None``. NumPy integers
    become ``int``.

    Parameters
    ----------
    settings : Settings

    Returns
    -------
    Settings

    Raises
    ------
    TypeError
        If a setting is not an integer.
    ValueError
        If the settings break a bound.
    """
    k, m, lag, thin = (
        integer(name, getattr(settings, name)) for name in ("k", "m", "lag", "thin")
    )
    max_iter = settings.max_iter
    if max_iter is not None:
        max_iter = integer("max_iter", max_iter, "an integer or None")
    if not 0 <= k <= m:
        raise ValueError(f"k and m must satisfy 0 <= k <= m, not k={k}, m={m}")
    if lag < 1 or thin < 1:
        raise ValueError(f"lag and thin must be at least 1, not lag={lag}, thin={thin}")
    if max_iter is not None and max_iter <= max(lag, m):
        raise ValueError(f"max_iter must exceed max(lag, m) = {max(lag, m)}, not {max_iter}")
    return Settings(k, m, lag, thin, max_iter, settings.store)


def merge(parts: Sequence[Replicates]) -> Replicates:
    """Join the outputs of disjoint sets of replicates of one run, sorted by replicate index.

    The parts must share the root seed, the settings and the dimension; the requested
    indices are the union of theirs.

    Parameters
    ----------
    parts : sequence of Replicates

    Returns
    -------
    Replicates
    """
    if not parts:
        raise ValueError("parts must hold at least one Replicates")
    first, identity = parts[0], _seed_key(parts[0].seed)
    for part in parts[1:]:
        if _seed_key(part.seed) != identity:
            raise ValueError(
                "cannot merge runs with different seeds: their indices differ in meaning"
            )
        if part.settings != first.settings:
            raise ValueError(
                f"cannot merge runs with different settings: {first.settings} and {part.settings}"
            )
        if part.atoms.shape[1] != first.atoms.shape[1]:
            raise ValueError(
                f"cannot merge runs of dimensions {first.atoms.shape[1]} and {part.atoms.shape[1]}"
            )
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
        seed=first.seed,
        settings=first.settings,
        **columns,
        requested=np.unique(np.concatenate([part.requested for part in parts])),
        n_log_prob=sum(part.n_log_prob for part in parts),
        n_grad=sum(part.n_grad for part in parts),
        seconds=sum(part.seconds for part in parts),
        paths={index: path for part in parts for index, path in part.paths.items()},
    )


def _seed_key(seed: SeedSequence) -> tuple[Any, ...]:
    # Entropy may be an int, a sequence or an array; lists compare by value.
    return np.atleast_1d(seed.entropy).tolist(), seed.spawn_key, seed.pool_size


class _Start:
    # A replicate's generator and initial states, drawn X_0 first (spec section 6.1).
    def __init__(
        self, index: int, initial: Distribution, seed: SeedSequence, ndim: int | None
    ) -> None:
        self.index = index
        self.generator: Generator = replicate_generator(seed, MAIN, index)
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
        settings: Settings,
        keep_paths: bool,
    ) -> None:
        self.target, self.kernel, self.settings = target, kernel, settings
        self.k, self.m, self.lag, self.thin, self.max_iter, self.store = settings
        self.keep_paths = keep_paths
        self.weight = 1.0 / ((self.m - self.k) // self.thin + 1)  # 1 / |S|, without building S
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
        self, seed: SeedSequence, requested: IntArray, n_log_prob: int, n_grad: int, start: float
    ) -> Replicates:
        # The seconds since start include this join, a large part of a run with many atoms.
        index, tau, cost = self.finished.join()
        owner, atoms, weights = self.atoms.join(keep=index[tau >= 0])
        distance_owner, distance_time, distance = self.distances.join()
        return Replicates(
            seed=seed,
            settings=self.settings,
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
