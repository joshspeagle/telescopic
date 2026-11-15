"""The engine: replicates of coupled chains, run in lockstep lanes with refill."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Self, cast

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

from telescopic._random import MAIN, Streams, replicate_generator, root_seed
from telescopic._types import (
    BoolArray,
    Distribution,
    FloatArray,
    IntArray,
    Seed,
    averaging,
    flag,
    instance,
    integer,
    integral,
    real_array,
)
from telescopic.estimators import correction_coefficients
from telescopic.initial import as_initial, draw
from telescopic.kernels.base import Kernel, State
from telescopic.target import Target

__all__ = ["Replicates", "Settings", "Status", "check_settings", "merge", "run", "stored_bytes"]

# Replicates advanced together by default (DECISIONS #52).
LANES = 256

# The largest time that the int32 times of the atoms hold (DECISIONS #234).
MAX_TIME = 2**31 - 1

# The dimension up to which a run that keeps a subset stores every second moment, and
# above which only the squares: at d = 1,000 the (d, d) sums of 256 lanes would take
# 2.05 GB (DECISIONS #236, #245).
FULL_MOMENTS = 100

# The floats of a block of products that a run with keep adds at a time, 1 MiB, and the
# dimension above which it adds the products of one atom at a time (DECISIONS #245).
BLOCK = 2**17
ROW_PRODUCTS = 24

# The target's counters, which a run's replicates hold under the same names.
COUNTS = ("n_log_prob", "n_grad", "n_grad_zeroed")


# Spec section 3.2 and R-EST-5.
class Settings(NamedTuple):
    r"""The settings of a run.

    Attributes
    ----------
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        Average every ``thin``-th time from ``k``.
    max_iter : int or None
        The cap: a pair still unmet when :math:`t` reaches it is censored.
    store : bool
        Whether the run stores the signed measures and distances; meeting-time pilots do
        not.
    keep : int or None
        With ``store``: the replicates of index below ``keep`` store their signed measures
        and distances, and every replicate stores its moments; ``None`` stores the signed
        measures and distances of every replicate, and no moments.
    """

    k: int
    m: int
    lag: int
    thin: int = 1
    max_iter: int | None = None
    store: bool = True
    keep: int | None = None


# Spec section 6.2; seed, section 6.1; atoms, R-EST-6; requested, section 6.5; evaluations,
# R-EST-7 and R-INPUT-1.
@dataclass
class Replicates:
    r"""The outputs of a set of replicates, sorted by replicate index, and their run's settings.

    Attributes
    ----------
    seed : numpy.random.SeedSequence
        The root seed of the run, from which each replicate's stream derives.
    settings : Settings
    index : ndarray of int, shape (R,)
        Replicate indices.
    meeting_time : ndarray of float, shape (R,)
        :math:`\tau = \inf\{t \ge L : X_t = Y_{t-L}\}`, or ``inf`` for a censored pair.
    atoms, weights, owner : ndarrays, shapes (N, d), (N,), (N,)
        The signed measures of the replicates that met: the atoms, their
        weights (each replicate's sum to 1), and the index of the replicate each atom
        belongs to. Sorted by owner, and within a replicate in record order: by time, and
        at each time the averaged atom first.
    time : ndarray of int32, shape (N,)
        The time :math:`t` of the pair :math:`(X_t, Y_{t-L})` at which each atom was
        recorded; -1 for replicates made without it.
    center : ndarray, shape (R, d), or (R, 0) without ``keep``
        :math:`c_r = X_k`, each replicate's first averaged state; ``nan`` for a censored
        pair.
    moments : ndarray, shape (R, p), or (R, 0) without ``keep``
        :math:`\hat\pi_r(x - c_r)`, then :math:`\hat\pi_r` of the products
        :math:`(x - c_r)_i (x - c_r)_j` for ``i, j`` in ``numpy.triu_indices(d)`` up to
        :math:`d = 100`, or of the squares :math:`(x - c_r)_j^2` above it; ``nan`` for a
        censored pair. The rows of every replicate, whether its atoms are stored or not.
    distance_owner, distance_time, distance : ndarrays, shape (M,)
        :math:`\lVert X_t - Y_{t-L}\rVert` for each pair, at each time :math:`L \le t < \tau`
        (to the cap for a censored pair).
    zero_density : ndarray of int
        The replicates whose :math:`X_0` or :math:`Y_0` has zero target density.
    requested : ndarray of int
        The indices the run was asked for, sorted: ``index`` lacks those of replicates
        that an interrupted run never started, or that an error stopped.
    abandoned : ndarray of int
        The replicates in progress when an error stopped the run, sorted; nothing of them
        is recorded.
    n_log_prob, n_grad : int
        Evaluations of the log-density and its gradient.
    n_grad_zeroed : int
        Gradients taken as 0 because they were not finite where the log-density is.
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
    atoms: FloatArray
    weights: FloatArray
    owner: IntArray
    distance_owner: IntArray
    distance_time: IntArray
    distance: FloatArray
    zero_density: IntArray
    requested: IntArray
    abandoned: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    time: npt.NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    center: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    moments: FloatArray = field(default_factory=lambda: np.zeros((0, 0)))
    n_log_prob: int = 0
    n_grad: int = 0
    n_grad_zeroed: int = 0
    seconds: float = 0.0
    paths: dict[int, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Replicates made without times, as by hand, get -1 for each atom, and without
        # moments, rows with no columns.
        if not len(self.time) and len(self.atoms):
            self.time = np.full(len(self.atoms), -1, dtype=np.int32)
        if not self.center.shape[1]:
            self.center = np.zeros((len(self.index), 0))
        if not self.moments.shape[1]:
            self.moments = np.zeros((len(self.index), 0))

    @property
    def stopped(self) -> FloatArray:
        """The time each pair stopped: its meeting time, or the cap if censored.

        Returns
        -------
        ndarray, shape (R,)
        """
        cap = self.settings.max_iter
        stopped: FloatArray = np.minimum(self.meeting_time, np.inf if cap is None else cap)
        return stopped

    # Spec R-EST-7.
    @property
    def cost(self) -> IntArray:
        r"""Kernel applications, counting a coupled step before the meeting as two.

        X runs to :math:`\max(s, m)` and Y from :math:`L` to :math:`s`, with :math:`s` the
        time the pair stopped.

        Returns
        -------
        ndarray of int, shape (R,)
        """
        stopped = self.stopped
        cost: IntArray = (stopped + np.maximum(stopped, self.settings.m)).astype(np.int64)
        return cost - self.settings.lag


class _Outcome(NamedTuple):
    # What a run returns inside the package (spec section 6.5): the record of its finished
    # replicates, None when none could be made, and the error that stopped it, None when
    # none did; others counts the errors of a pool's other chunks.
    replicates: Replicates | None
    error: Exception | None
    others: int = 0


class Status(NamedTuple):
    """The progress of a run, given to its monitor after every iteration.

    Attributes
    ----------
    finished : int
        Replicates finished so far.
    slowest : int
        Index of the unmet replicate that has run longest, or -1 if every pair has met.
    t : int
        Its time :math:`t`, or 0.
    zero_density : int
        Replicates started so far whose :math:`X_0` or :math:`Y_0` has zero target
        density.
    first_zero_density : int
        Index of the first of them, or -1 if there is none.
    slowest_zero_density : bool
        Whether the replicate ``slowest`` is one of them.
    """

    finished: int
    slowest: int
    t: int
    zero_density: int = 0
    first_zero_density: int = -1
    slowest_zero_density: bool = False


# Spec sections 3.1, 6.1, 6.2 and 6.5, R-EST-5, R-EXEC-2 and R-EXEC-7.
def run(
    target: Target,
    kernel: Kernel,
    initial: Any,
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
    keep: int | None = None,
    keep_paths: bool = False,
    monitor: Callable[[Status], bool] | None = None,
) -> Replicates:
    r"""Run the replicates ``indices`` of the estimator :math:`H_S`.

    Each replicate draws :math:`X_0` and then :math:`Y_0` from ``initial`` with its own
    stream, runs :math:`X` alone for ``lag`` steps, then the coupled
    kernel until :math:`X_t = Y_{t-L}` under ``==``, then :math:`X` alone until :math:`t = m`.
    For a log-density evaluated point by point, a replicate's output depends only on
    the root seed, its index and the arguments, bitwise, never on the other replicates
    or on ``lanes``. For a vectorized one the kernels compute their products with BLAS,
    whose rounding can depend on the replicates beside it and on the number of BLAS
    threads: the output repeats bitwise with the same arguments, ``lanes``, BLAS build
    and number of BLAS threads, and otherwise a replicate can differ by rounding, which
    can move its meeting time and its whole output, while the estimates agree within
    their error bars (Reproducibility, in :doc:`/guide/parallel`). The engine emits no
    warning: it records what its caller needs to warn about.

    Parameters
    ----------
    target : Target
        The target to sample.
    kernel : Kernel
        A tuned kernel.
    initial
        An object with ``rvs(random_state=generator)``, a callable ``initial(rng)`` or an
        array of points of shape ``(n, d)``, as for :func:`telescopic.initial.as_initial`.
    indices : array_like of int, shape (R,)
        Distinct replicate indices, at least one, started in this order.
    k, m : int
        First and last averaged times, :math:`0 \le k \le m`.
    lag : int
        The lag :math:`L \ge 1`.
    thin : int
        Average every ``thin``-th time from ``k``.
    max_iter : int, optional
        Censor a pair still unmet when :math:`t` reaches ``max_iter``, which must exceed
        ``max(lag, m)``. ``None`` runs every pair until it meets.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root seed, recorded in the output. Runs of other indices
        with the same ``SeedSequence`` or integer can be merged.
    lanes : int
        Number of replicates advanced together, at most the number requested.
    store : bool
        Whether to store the signed measures and distances; meeting-time pilots do not.
    keep : int, optional
        With ``store``: store the signed measures, their times and the distances of the
        replicates of index below ``keep`` only, and the moments of every replicate
        (``center`` and ``moments`` of :class:`Replicates`), at a cost of
        :math:`O(p)` per atom with :math:`p = d(d + 3)/2` up to :math:`d = 100` and
        :math:`2d` above it. ``None`` stores the signed measures and distances of every
        replicate, and no moments.
    keep_paths : bool
        Whether to keep the paths of both chains, for inspection.
    monitor : callable, optional
        Called with a :class:`Status` after every iteration. Once it returns true, no
        new replicate starts and the ones in progress finish, so the finished indices
        are those started.

    Returns
    -------
    Replicates

    Raises
    ------
    Exception
        An error raised while the replicates run, such as one of the log-density, is
        raised again as the same object once the replicates finished before it are
        recorded: its attribute ``telescopic_replicates`` holds them, a
        :class:`Replicates` whose ``abandoned`` lists the replicates in progress, or
        ``None`` when no record could be made, and one note says so. An exception object
        raised in several runs carries the attribute and the note of the last of them.

    Notes
    -----
    Memory: :func:`stored_bytes` gives the size of the stored atoms and distances.
    """
    outcome = _run(
        target,
        kernel,
        initial,
        indices,
        k=k,
        m=m,
        lag=lag,
        thin=thin,
        max_iter=max_iter,
        rng=rng,
        lanes=lanes,
        store=store,
        keep=keep,
        keep_paths=keep_paths,
        monitor=monitor,
    )
    replicates, error, _ = outcome
    if error is None:
        return cast(Replicates, replicates)
    if replicates is None:
        _attach(error, "telescopic_replicates", None)  # _run noted why
    else:
        note = (
            f"{_STOPPED} the run with {len(replicates.index)} of its "
            f"{len(replicates.requested)} replicates finished and {len(replicates.abandoned)} "
            "in progress abandoned; error.telescopic_replicates holds the finished ones"
        )
        _attach(error, "telescopic_replicates", replicates, note)
    try:
        raise error
    finally:
        # The traceback keeps this frame, whose names would otherwise keep the error.
        del error, outcome


# The start of the note that a run adds to the error that stopped it, and the notes of a
# run that recorded nothing: none of its replicates had finished, or their record failed.
_STOPPED = "telescopic: this error stopped"
_NONE = f"{_STOPPED} the run before any of its replicates finished"
_LOST = (
    f"{_STOPPED} the run, and no record of the replicates finished before it could be made, "
    "so they were lost"
)


def _attach(error: Exception, name: str, value: Any, note: str | None = None) -> None:
    # Sets a run's attribute on the error that stopped it, and its note.
    setattr(error, name, value)
    _note(error, note)


def _note(error: Exception, note: str | None = None) -> None:
    # The one note of a run on the error that stopped it, which replaces that of an earlier
    # run that raised the same object; without a note, the error keeps the run's note it
    # has, or gets _NONE.
    notes = getattr(error, "__notes__", [])
    kept = [line for line in notes if line.startswith(_STOPPED)]
    if note is None:
        note = kept[0] if kept else _NONE
    notes[:] = [line for line in notes if not line.startswith(_STOPPED)]
    error.add_note(note)


def _run(
    target: Target,
    kernel: Kernel,
    initial: Any,
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
    keep: int | None = None,
    keep_paths: bool = False,
    monitor: Callable[[Status], bool] | None = None,
) -> _Outcome:
    # run, returning an outcome: an error that stops the replicates is returned with the
    # record of those finished before it, never raised, while KeyboardInterrupt and the
    # other BaseExceptions are. Invalid arguments raise before any replicate starts.
    instance("target", target, Target)
    instance("kernel", kernel, Kernel)
    settings = check_settings(Settings(k, m, lag, thin, max_iter, flag("store", store), keep))
    keep_paths = flag("keep_paths", keep_paths)
    lanes = integer("lanes", lanes, minimum=1)
    requested = real_array("indices", indices, "distinct integers >= 0", dtype=None)
    if (
        requested.ndim != 1
        or not len(requested)
        or not np.issubdtype(requested.dtype, np.integer)
        or requested.min() < 0
        or len(np.unique(requested)) != len(requested)
    ):
        raise ValueError("indices must be a non-empty sequence of distinct integers >= 0")
    requested = requested.astype(np.int64)
    lanes = min(lanes, len(requested))  # empty lanes would only cost time
    seed, initial = root_seed(rng), as_initial(initial)
    queue = iter(requested.tolist())
    first = next(queue)
    start = time.perf_counter()
    with _Caught() as caught:
        head = _Start(first, initial, seed, None)
    if caught.error is not None:
        _note(caught.error, _NONE)
        return _Outcome(None, caught.error)
    ndim = len(head.x0)
    starts = itertools.chain([head], (_Start(i, initial, seed, ndim) for i in queue))
    batch = _Batch(target, kernel, ndim, lanes, settings, keep_paths)
    with _Caught() as caught:
        batch.fill(np.arange(lanes), starts)
        stopping = False
        while batch.active.any():
            batch.advance()
            free = batch.finish()
            stopping = (monitor is not None and monitor(batch.status())) or stopping
            if not stopping:
                batch.fill(free, starts)
    error = caught.error
    with _Caught() as recording:
        return _Outcome(batch.result(seed, np.sort(requested), start), error)
    error = error or recording.error
    _note(cast(Exception, error), _LOST)
    return _Outcome(None, error)


class _Caught:
    # A block that stops an Exception raised in it and keeps it in error, while
    # KeyboardInterrupt and the other BaseExceptions pass. The error's traceback keeps the
    # frame of the block, so a frame that must not outlive the error, such as one that
    # holds worker processes, catches in a function of its own.
    def __init__(self) -> None:
        self.error: Exception | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, _: object
    ) -> bool:
        if isinstance(error, Exception):
            self.error = error
            return True
        return False


# Spec section 3.2.
def check_settings(settings: Settings) -> Settings:
    r"""Check the settings of a run and return them with Python integers.

    They need integers :math:`0 \le k \le m`, :math:`L \ge 1` and ``thin`` :math:`\ge 1`,
    and an integer ``max_iter`` above :math:`\max(L, m)` or ``None``, at most
    :math:`2^{31} - 1` when the run stores the times of its atoms, and ``keep`` ``None`` or
    an integer :math:`\ge 0`, only with ``store``. NumPy integers become ``int``.

    Parameters
    ----------
    settings : Settings
        The settings to check.

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
    k, m, thin = averaging(settings.k, settings.m, settings.thin)
    lag = integer("lag", settings.lag, minimum=1)
    max_iter = settings.max_iter
    if max_iter is not None:
        max_iter = integer("max_iter", max_iter, "an integer or None")
    if max_iter is not None and max_iter <= max(lag, m):
        raise ValueError(f"max_iter must exceed max(lag, m) = {max(lag, m)}, not {max_iter}")
    keep = settings.keep
    if keep is not None:
        keep = integer("keep", keep, "None or an integer >= 0", minimum=0)
        if not settings.store:
            raise ValueError("keep needs store=True")
    if max_iter is not None and max_iter > MAX_TIME and settings.store and keep != 0:
        raise ValueError(
            f"max_iter must be at most 2**31 - 1 = {MAX_TIME} where times are stored, not "
            f"{max_iter}"
        )
    return Settings(k, m, lag, thin, max_iter, settings.store, keep)


def merge(parts: Sequence[Replicates]) -> Replicates:
    """Join the outputs of disjoint sets of replicates of one run, sorted by replicate index.

    The parts must share the root seed, the settings and the dimension; the requested
    and the abandoned indices are the unions of theirs. For a target evaluated point by point the result is
    bitwise that of one run of all the replicates; for a vectorized one it can differ
    from it as a run with other ``lanes`` does (:func:`run`).

    Parameters
    ----------
    parts : sequence of Replicates
        The outputs to join, at least one.

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
        "index": ("index", "meeting_time", "center", "moments"),
        "owner": ("owner", "atoms", "weights", "time"),
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
    paths: dict[int, tuple[FloatArray, FloatArray]] = {}
    for part in parts:
        paths.update(part.paths)
    return Replicates(
        seed=first.seed,
        settings=first.settings,
        **columns,
        requested=np.unique(np.concatenate([part.requested for part in parts])),
        abandoned=np.unique(np.concatenate([part.abandoned for part in parts])),
        **{name: sum(getattr(part, name) for part in parts) for name in (*COUNTS, "seconds")},
        paths=paths,
    )


def _seed_key(seed: SeedSequence) -> tuple[Any, ...]:
    # Entropy may be an int, a sequence or an array; lists compare by value.
    return np.atleast_1d(np.asarray(seed.entropy)).tolist(), seed.spawn_key, seed.pool_size


class _Start:
    # A replicate's generator and initial states, drawn X_0 first (spec section 6.1).
    def __init__(
        self, index: int, initial: Distribution, seed: SeedSequence, ndim: int | None
    ) -> None:
        self.index = index
        self.generator: Generator = replicate_generator(seed, MAIN, index)
        self.x0 = draw(initial, self.generator, ndim)
        self.y0 = draw(initial, self.generator, len(self.x0))


def stored_bytes(settings: Settings, ndim: int, coupled: npt.ArrayLike) -> float:
    r"""Bytes of the atoms, distances and moments that replicates store.

    A replicate that met after :math:`c = \tau - L` coupled steps stores the
    :math:`|S|` atoms of the time average and two atoms for each of the
    :math:`\max(0, c - k)` terms of the bias correction, each of :math:`8d + 20` bytes (its
    coordinates in ``atoms``, its weight, its replicate in ``owner`` and its ``time``), and
    :math:`c`
    distances of 24 bytes (``distance_owner``, ``distance_time`` and ``distance``). This
    is exact when ``thin`` is 1 and :math:`m - k + 1 \ge L`, and an upper bound otherwise,
    where some terms of the correction have no atoms. With ``store=False`` nothing is
    stored. With ``keep``, only the first ``keep`` replicates store atoms and distances,
    and every replicate stores its moments, :math:`8(p + d)` bytes (``center`` and
    ``moments`` of :class:`Replicates`), with :math:`p = d(d + 3)/2` up to :math:`d = 100`
    and :math:`2d` above.

    Parameters
    ----------
    settings : Settings
        The settings of the run.
    ndim : int
        The dimension :math:`d`.
    coupled : array_like of int, shape (R,)
        :math:`\tau - L` of each replicate, non-negative, in the order of their indices.

    Returns
    -------
    float
        The total over the replicates.
    """
    instance("settings", settings, Settings)
    k, m, _, thin, _, store, keep = check_settings(settings)
    ndim = integer("ndim", ndim, minimum=1)
    c = real_array("coupled", coupled, "non-negative integers of shape (R,)")
    if c.ndim != 1 or not integral(c, 0):
        raise ValueError("coupled must be non-negative integers of shape (R,)")
    if not store:
        return 0.0
    moments = 0
    if keep is not None:
        moments = len(c) * 8 * (2 * ndim + len(_pairs(ndim)[0]))
        c = c[:keep]
    atoms = len(c) * len(range(k, m + 1, thin)) + 2 * np.maximum(0, c - k).sum()
    return float(atoms * (8 * ndim + 20) + 24 * c.sum() + moments)


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
        self.keep_paths = keep_paths
        self.weight = 1.0 / len(range(settings.k, settings.m + 1, settings.thin))  # 1 / |S|
        self.streams = Streams(lanes)
        self.replicate = np.full(lanes, -1, dtype=np.int64)
        self.t = np.zeros(lanes, dtype=np.int64)
        self.tau = np.full(lanes, -1, dtype=np.int64)
        self.x = _empty_state(lanes, ndim, kernel.needs_grad)
        self.y = _empty_state(lanes, ndim, kernel.needs_grad)
        integers, reals = np.zeros(0, dtype=np.int64), np.zeros(0)
        self.finished = _Chunks(integers, integers)
        # The columns that stored_bytes counts: a change here changes it.
        self.atoms = _Chunks(integers, np.zeros((0, ndim)), reals, np.zeros(0, dtype=np.int32))
        self.distances = _Chunks(integers, integers, reals)
        # The replicates started at zero density, and whether each lane's did.
        self.zero_density: list[int] = []
        self.outside = np.zeros(lanes, dtype=bool)
        self.paths: dict[int, tuple[list[FloatArray], list[FloatArray]]] = {}
        self.n_finished = 0
        self.counts = {name: getattr(target, name) for name in COUNTS}  # before any step
        # With keep: whether each lane's replicate stores its rows, and its center c_r and
        # sums of w h_c(x): of x - c_r, and of the products, all of them up to FULL_MOMENTS
        # dimensions, of which finish keeps those of _pairs, or the squares above it (spec
        # R-EST-9).
        self.reducing = settings.keep is not None
        self.kept = np.ones(lanes, dtype=bool)
        width = ndim if self.reducing else 0
        self.full = self.reducing and ndim <= FULL_MOMENTS
        self.center = np.zeros((lanes, width))
        self.first = np.zeros((lanes, width))
        self.second = np.zeros((lanes, width, width) if self.full else (lanes, width))
        columns = width + len(_pairs(ndim)[0]) if self.reducing else 0
        self.reduced = _Chunks(integers, np.zeros((0, width)), np.zeros((0, columns)))
        # Rows a block of the products holds: its temporaries take at most a few MiB.
        self.block = max(1, BLOCK // max(1, self.second[0].size))

    @property
    def active(self) -> BoolArray:
        active: BoolArray = self.replicate >= 0
        return active

    def fill(self, free: IntArray, starts: Iterator[_Start]) -> None:
        starting = list(itertools.islice(starts, len(free)))
        if not starting:
            return
        lanes = free[: len(starting)]
        states = self.kernel.init_state(self.target, np.vstack([(s.x0, s.y0) for s in starting]))
        self.x.put(lanes, states.take(slice(0, None, 2)))
        self.y.put(lanes, states.take(slice(1, None, 2)))
        outside = np.isneginf(states.log_prob).reshape(-1, 2).any(axis=1)
        self.outside[lanes] = outside
        self.zero_density += [s.index for s, bad in zip(starting, outside, strict=True) if bad]
        for lane, start in zip(lanes.tolist(), starting, strict=True):
            self.streams.assign(lane, start.generator)
            if self.keep_paths:
                self.paths[start.index] = ([start.x0], [start.y0])
        self.replicate[lanes] = [s.index for s in starting]
        if self.reducing:
            self.kept[lanes] = self.replicate[lanes] < cast(int, self.settings.keep)
            self.first[lanes] = 0.0
            self.second[lanes] = 0.0
        self.t[lanes] = 0
        self.tau[lanes] = -1
        self._record(lanes, np.zeros(0, dtype=np.int64))

    def advance(self) -> None:
        # One iteration: X alone in the lagging and met lanes, the coupled kernel in the
        # others, then the exact comparison of R-EST-2 for every unmet lane past the lag.
        active, unmet, lag = self.active, self.tau < 0, self.settings.lag
        solo = (active & ((self.t < lag) | ~unmet)).nonzero()[0]
        coupled = (active & unmet & (self.t >= lag)).nonzero()[0]
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
        compared = (active & unmet & (self.t >= lag)).nonzero()[0]
        if len(compared):
            met = np.all(self.x.x[compared] == self.y.x[compared], axis=1)
            self.tau[compared[met]] = self.t[compared[met]]
            compared = compared[~met]
        stepped = active.nonzero()[0]
        if self.keep_paths:
            for lane in stepped.tolist():
                self.paths[int(self.replicate[lane])][0].append(self.x.x[lane].copy())
            for lane in coupled.tolist():
                self.paths[int(self.replicate[lane])][1].append(self.y.x[lane].copy())
        self._record(stepped, compared)

    def finish(self) -> IntArray:
        # Frees the lanes of the pairs that met and reached m, and of censored pairs.
        done = (self.tau >= 0) & (self.t >= self.settings.m)
        if self.settings.max_iter is not None:
            done |= (self.tau < 0) & (self.t >= self.settings.max_iter)
        lanes = (self.active & done).nonzero()[0]
        if len(lanes):
            self.finished.add(self.replicate[lanes], self.tau[lanes])
            if self.reducing:
                second = self.second[lanes]
                if self.full:
                    second = second[:, *_pairs(self.center.shape[1])]
                center, sums = self.center[lanes], np.hstack([self.first[lanes], second])
                censored = self.tau[lanes] < 0
                center[censored], sums[censored] = np.nan, np.nan
                self.reduced.add(self.replicate[lanes], center, sums)
            self.replicate[lanes] = -1
            self.n_finished += len(lanes)
        return lanes

    def status(self) -> Status:
        unmet = self.active & (self.tau < 0)
        outside = len(self.zero_density), self.zero_density[0] if self.zero_density else -1
        if not unmet.any():
            return Status(self.n_finished, -1, 0, *outside)
        lane = int(np.argmax(np.where(unmet, self.t, -1)))
        return Status(
            self.n_finished,
            int(self.replicate[lane]),
            int(self.t[lane]),
            *outside,
            bool(self.outside[lane]),
        )

    def result(self, seed: SeedSequence, requested: IntArray, start: float) -> Replicates:
        # The finished replicates' outputs. After an error, the lanes still active hold the
        # abandoned replicates, whose rows, starts at zero density and paths are left out
        # (spec section 6.5). The seconds since start include this join, a large part of a
        # run with many atoms.
        index, tau = self.finished.join()
        abandoned = np.sort(self.replicate[self.active])
        owner, atoms, weights, times = self.atoms.join(keep=index[tau >= 0])
        _, center, moments = self.reduced.join()
        distance_owner, distance_time, distance = self.distances.join(
            keep=index if len(abandoned) else None
        )
        zero_density = np.array(self.zero_density, dtype=np.int64)
        finished = set(index.tolist())
        return Replicates(
            seed=seed,
            settings=self.settings,
            index=index,
            meeting_time=np.where(tau >= 0, tau, np.inf),
            atoms=atoms,
            weights=weights,
            owner=owner,
            distance_owner=distance_owner,
            distance_time=distance_time,
            distance=distance,
            zero_density=np.sort(zero_density[np.isin(zero_density, index)]),
            requested=requested,
            abandoned=abandoned,
            time=times,
            center=center,
            moments=moments,
            **{name: getattr(self.target, name) - n for name, n in self.counts.items()},
            seconds=time.perf_counter() - start,
            paths={
                i: (np.array(x), np.array(y)) for i, (x, y) in self.paths.items() if i in finished
            },
        )

    def _record(self, stepped: IntArray, unmet: IntArray) -> None:
        # The averaged atoms X_t, t in S, of the lanes that moved, and the bias-correction
        # atoms and distances of the unmet pairs (spec R-EST-6, section 3.6). Kinds without
        # rows are skipped: at one lane most iterations lack one, and its calls add up.
        k, m, lag, thin, _, store, _ = self.settings
        if not store:
            return
        t = self.t[stepped]
        if t.max(initial=0) > MAX_TIME:
            # A time beyond int32, reachable without a cap, raises naming its replicate,
            # where the replicate stores the times of its atoms (spec R-EST-6).
            late = stepped[(t > MAX_TIME) & self.kept[stepped]]
            if len(late):
                raise OverflowError(
                    f"replicate {self.replicate[late[0]]} reached t = {self.t[late[0]]}, "
                    "beyond the 2**31 - 1 that the times of its atoms hold; set max_iter"
                )
        chosen = (t >= k) & (t <= m) & ((t - k) % thin == 0)
        averaged = stepped[chosen]
        if len(averaged):
            if self.reducing:
                # c_r = X_k, before any atom of the replicate (the corrections start at
                # k + L).
                first = averaged[t[chosen] == k]
                self.center[first] = self.x.x[first]
            self._add(averaged, self.x.x[averaged], np.full(len(averaged), self.weight))
        if not len(unmet):
            return
        coefficient = correction_coefficients(self.t[unmet], k, m, lag, thin)
        positive = coefficient > 0
        if positive.any():
            corrected = unmet[positive]
            for chain, sign in ((self.x, 1.0), (self.y, -1.0)):
                self._add(corrected, chain.x[corrected], sign * coefficient[positive])
        if self.reducing:
            unmet = unmet[self.kept[unmet]]
        self.distances.add(
            self.replicate[unmet],
            self.t[unmet],
            np.linalg.norm(self.x.x[unmet] - self.y.x[unmet], axis=1),
        )

    def _add(self, lanes: IntArray, points: FloatArray, weights: FloatArray) -> None:
        # Atoms of distinct lanes: their weighted moments, with keep, and their rows, for
        # the replicates that store them.
        if self.reducing:
            self._accumulate(lanes, points, weights)
            stored = self.kept[lanes]
            lanes, points, weights = lanes[stored], points[stored], weights[stored]
            if not len(lanes):
                return
        self.atoms.add(self.replicate[lanes], points, weights, self.t[lanes].astype(np.int32))

    def _accumulate(self, lanes: IntArray, points: FloatArray, weights: FloatArray) -> None:
        # Adds w h_c(x) to each lane's sums: w (x - c_r), and w (x - c_r)_i (x - c_r)_j for
        # every i, j, or i = j above FULL_MOMENTS. Elementwise, with no BLAS, so that a
        # row's sums do not depend on its batch (claims K1 and E1). Up to ROW_PRODUCTS
        # dimensions the products of a block of rows are formed at once, the block's
        # temporaries within 1 MiB (DECISIONS #180); above, those of one row at a time,
        # which skips gathering and scattering the lanes' (d, d) sums (DECISIONS #245).
        dx = points - self.center[lanes]
        self.first[lanes] += weights[:, None] * dx
        if self.full and len(dx[0]) > ROW_PRODUCTS:
            for lane, row, weight in zip(lanes.tolist(), dx, weights.tolist(), strict=True):
                products = np.multiply.outer(row, row)
                products *= weight
                self.second[lane] += products
            return
        for start in range(0, len(lanes), self.block):
            rows = slice(start, start + self.block)
            lane, weight, block = lanes[rows], weights[rows], dx[rows]
            if self.full:
                products = block[:, :, None] * block[:, None, :]
                products *= weight[:, None, None]
            else:
                products = block * block
                products *= weight[:, None]
            self.second[lane] += products


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
        self.blocks, self.chunks = [], []  # so that the join holds two copies, not three
        order = np.argsort(columns[0], kind="stable")
        if keep is not None:
            order = order[np.isin(columns[0][order], keep)]
        return [column[order] for column in columns]


def _concatenate(chunks: list[tuple[npt.NDArray[Any], ...]]) -> tuple[npt.NDArray[Any], ...]:
    return tuple(np.concatenate(parts) for parts in zip(*chunks, strict=True))


def _lane_bytes(ndim: int) -> int:
    # The bytes of one lane's center and sums in a run with keep: d, d and d^2 floats up to
    # FULL_MOMENTS dimensions, 3 d above.
    return 8 * (2 * ndim + (ndim * ndim if ndim <= FULL_MOMENTS else ndim))


def _pairs(ndim: int) -> tuple[IntArray, IntArray]:
    # The coordinates (i, j) of the second moments that a run with keep stores: i <= j up
    # to FULL_MOMENTS dimensions, in the order of numpy.triu_indices, and i = j above.
    if ndim <= FULL_MOMENTS:
        return np.triu_indices(ndim)
    diagonal = np.arange(ndim)
    return diagonal, diagonal


def _empty_state(lanes: int, ndim: int, with_grad: bool) -> State:

    return State(
        np.zeros((lanes, ndim)), np.zeros(lanes), np.zeros((lanes, ndim)) if with_grad else None
    )
