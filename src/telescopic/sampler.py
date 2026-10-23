"""The sampler: the automatic pipeline from a log-density to unbiased estimates."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, cast

import numpy as np

from telescopic import diagnostics, engine
from telescopic._parallel import Interrupt, check_pool, execute, opened
from telescopic._progress import Progress
from telescopic._random import MAIN, PILOT, namespace_seed, replicate_generator, root_seed
from telescopic._types import FloatArray, IntArray, Seed, integer
from telescopic.initial import as_initial, draw
from telescopic.kernels import HMC, MALA, Independence, Kernel, RandomWalk
from telescopic.results import Result, parameter_names
from telescopic.target import Target
from telescopic.tuning import TuningReport, choose_k
from telescopic.warnings import ConvergenceWarning, InputWarning, warn

KERNELS: dict[str, Callable[[], Kernel]] = {
    "auto": RandomWalk,
    "rwmh": RandomWalk,
    "mala": MALA,
    "hmc": HMC,
    "independence": Independence,
}
PILOT_REPLICATES = 100
PILOT_MAX_ITER = 100_000
# The meeting-time pilot: lag 1, with nothing stored (spec R-TUNE-2).
PILOT_SETTINGS = engine.Settings(0, 0, 1, max_iter=PILOT_MAX_ITER, store=False)
# The warnings of spec section 7.4 that a run emits as they happen, not at its end, or as
# it ends for slow pairs on a pool.
_LIVE = ("slow", "tuning")


class CoupledSampler:
    r"""Unbiased MCMC with coupled chains: tuning, pilot meeting times and runs (spec R-API-1).

    The pipeline tunes the kernel's unset parameters on adaptive pilot chains, chooses
    ``k``, ``m`` and the lag from the meeting times of 100 lag-1 pilot pairs (spec
    R-TUNE-2), then runs replicates of coupled chains whose signed measures give unbiased
    estimates of any expectation under the target.

    Parameters
    ----------
    log_prob : callable or None
        :math:`\log\pi` up to a constant: ``log_prob(x, *args, **kwargs)`` for ``x`` of
        shape ``(d,)``, or of shape ``(n, d)`` with ``vectorize=True``. ``None`` only with
        a kernel that needs no log-density: a :class:`~telescopic.kernels.Gibbs` kernel or
        a mixture of them.
    init
        The initial distribution :math:`\pi_0`: an object with
        ``rvs(random_state=rng)`` (such as a frozen SciPy distribution or
        :class:`telescopic.initial.Normal`), a callable ``init(rng)``, or an array of
        points of shape ``(n, d)``. Any :math:`\pi_0` keeps the estimates unbiased; one
        wider than the target and covering all its modes keeps them reliable.
    args, kwargs
        Extra arguments of ``log_prob`` and ``grad``.
    grad : callable, optional
        :math:`\nabla\log\pi` with the signature of ``log_prob``.
    vectorize : bool
        Whether ``log_prob`` and ``grad`` take a batch of points.
    kernel : Kernel or str
        A kernel, or one of ``"auto"`` and ``"rwmh"`` (random-walk Metropolis with the
        reflection-maximal coupling), ``"mala"`` and ``"hmc"`` (which need ``grad``) and
        ``"independence"``.
    names : sequence of str, optional
        Parameter names for summaries.
    pool : None, int, executor or pool
        Where the pilot pairs and the replicates run: ``None`` in this process; an
        integer, that many worker processes, started for each call of :meth:`run` or
        :meth:`meeting_times`; or an object with ``submit``, such as a
        :class:`concurrent.futures.Executor`, or with ``map``, such as an emcee or
        schwimmbad pool. A pool runs chunks of consecutive indices, at most four per
        worker; a task in a worker process runs with BLAS pinned to one thread and
        Ctrl-C ignored. With a log-density evaluated point by point, no replicate
        depends on the pool (spec sections 6.1 and 6.4).
    progress : bool
        Show the progress of the pilot and of each run: a tqdm bar when tqdm is
        installed, lines on stderr otherwise.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root of every random stream (spec section 6.1).

    Attributes
    ----------
    kernel : Kernel
        The kernel, replaced by its tuned copy when tuned.
    tuning : TuningReport or None
        The report of the tuning pilot; ``None`` before tuning or with nothing to tune.
    pilot : ndarray or None
        The meeting times of every lag-1 pilot pair run so far; ``None`` before the first.
    seed : numpy.random.SeedSequence
        The root seed.
    ndim : int
        The dimension of the target.
    """

    def __init__(
        self,
        log_prob: Callable[..., Any] | None,
        init: Any,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        grad: Callable[..., Any] | None = None,
        vectorize: bool = False,
        kernel: Kernel | str = "auto",
        names: Sequence[str] | None = None,
        pool: Any = None,
        progress: bool = False,
        rng: Seed = None,
    ) -> None:
        check_pool(pool)
        self.pool, self.progress = pool, progress
        self.target = Target(log_prob, grad, args=args, kwargs=kwargs, vectorize=vectorize)
        self.initial = as_initial(init)
        self.kernel = _kernel(kernel)
        self.kernel.check_target(self.target)
        self.seed = root_seed(rng)
        # One draw from a throwaway generator gives the dimension, so that the kernel's and
        # the names are checked now rather than after tuning.
        self.ndim = len(draw(self.initial, replicate_generator(self.seed, MAIN, 0)))
        if self.kernel.ndim is not None and self.kernel.ndim != self.ndim:
            raise ValueError(
                f"kernel works in {self.kernel.ndim} dimensions, but init draws points in "
                f"{self.ndim}"
            )
        self.names = None if names is None else parameter_names(names, self.ndim)
        self.tuning: TuningReport | None = None
        self.pilot: FloatArray | None = None
        self._replicates: engine.Replicates | None = None

    @property
    def settings(self) -> engine.Settings | None:
        """The settings of the runs, fixed by the first; ``None`` before it."""
        return None if self._replicates is None else self._replicates.settings

    def tune(self) -> TuningReport | None:
        """Tune the kernel's unset parameters on pilot chains, once, and freeze the kernel.

        Warns when the chains disagree, a sign of several modes (spec section 5.3).

        Returns
        -------
        TuningReport or None
            ``None`` when the kernel had nothing to tune.
        """
        if not self.kernel.is_tuned:
            self.kernel, self.tuning = self.kernel.tune(self.target, self.initial, self.seed)
            disagreement = diagnostics.tuning_disagreement(self.tuning)
            if disagreement:
                warn(disagreement, ConvergenceWarning)
        return self.tuning

    def meeting_times(
        self, n: int = PILOT_REPLICATES, *, max_iter: int = PILOT_MAX_ITER
    ) -> FloatArray:
        """Meeting times of ``n`` new pilot pairs with lag 1 (spec R-TUNE-2).

        Each call runs new pilot replicates, continuing their indices, and adds them to
        ``pilot``.

        Parameters
        ----------
        n : int
        max_iter : int
            The cap; a pair still unmet there is censored (spec section 6.3).

        Returns
        -------
        ndarray, shape (n,)
            Meeting times, ``inf`` for a censored pair.
        """
        n = _count("n", n)
        settings = engine.check_settings(PILOT_SETTINGS._replace(max_iter=max_iter))
        with opened(self.pool) as pool:
            return self._meeting_times(n, settings, pool, self.progress)

    def _meeting_times(
        self, n: int, settings: engine.Settings, pool: Any, progress: bool
    ) -> FloatArray:
        self.tune()
        start = 0 if self.pilot is None else len(self.pilot)
        indices = np.arange(start, start + n)
        times = self._run(PILOT, indices, settings, pool, progress).meeting_time
        self.pilot = times if self.pilot is None else np.concatenate([self.pilot, times])
        return times

    def run(
        self,
        n_replicates: int,
        *,
        k: int | str = "auto",
        m: int | str = "auto",
        lag: int | str = "auto",
        thin: int | None = None,
        max_iter: int | None = None,
        progress: bool | None = None,
    ) -> Result:
        """Run ``n_replicates`` more replicates and return the result of all of them so far.

        The first call tunes the kernel, completes the pilot to 100 pairs for an automatic
        ``k``, and fixes the settings of the later calls (spec R-TUNE-2). The result warns
        about the conditions of spec section 7.4.

        The first Ctrl-C during the replicates lets those in progress finish and returns a
        result flagged ``interrupted``; the next call runs the missing replicates first. A
        second Ctrl-C raises ``KeyboardInterrupt`` at once, without a result (spec
        section 6.5).

        Parameters
        ----------
        n_replicates : int
        k, m, lag : int or "auto"
            ``"auto"`` derives a value on the first call: the pilot's quantile for ``k``,
            then ``lag = max(1, k)`` and ``m = 10 k`` (spec R-TUNE-2, D-21). A value given
            explicitly is kept. On later calls ``"auto"`` keeps the first call's value.
        thin : int, optional
            Average every ``thin``-th state from ``k`` (spec R-EST-5): 1 on the first
            call, and the first call's value on later ones.
        max_iter : int, optional
            Censor pairs still unmet at this time. ``None`` runs every pair until it
            meets on the first call, and keeps the first call's cap on later ones.
        progress : bool, optional
            Show the progress of the run and of the pilot it runs; ``None`` takes the
            sampler's setting.

        Returns
        -------
        Result
            Unbiased for an ``n_replicates`` fixed in advance; extending a run after
            looking at it keeps consistency, not exact unbiasedness (spec section 3.3).
        """
        n_replicates = _count("n_replicates", n_replicates)
        given = self._check(k, m, lag, thin, max_iter)
        show = self.progress if progress is None else progress
        # Worker processes start before tuning, which runs here meanwhile, and serve the
        # pilot and the run (DECISIONS #23).
        with opened(self.pool) as pool:
            settings = self.settings or self._configure(given, pool, show)
            done = 0 if self._replicates is None else len(self._replicates.index)
            self._check_memory(settings, done + n_replicates)
            part = self._run(MAIN, self._next_indices(n_replicates), settings, pool, show)
        self._replicates = (
            part if self._replicates is None else engine.merge([self._replicates, part])
        )
        result = Result(
            self._replicates,
            names=self.names,
            kernel=self.kernel,
            initial=self.initial,
            tuning=self.tuning,
            pilot=self.pilot,
        )
        for issue in diagnostics.issues(result):
            if issue.kind not in _LIVE:
                warn(issue.message, ConvergenceWarning)
        return result

    def _next_indices(self, n: int) -> IntArray:
        # The requested replicates that an interrupted run never started come first, then
        # new indices, so that a run completes an interrupted one (spec section 6.5).
        if self._replicates is None:
            return np.arange(n)
        requested = self._replicates.requested
        missing = np.setdiff1d(requested, self._replicates.index, assume_unique=True)
        top = int(requested[-1]) + 1
        return np.concatenate([missing, np.arange(top, top + n)])[:n]

    def _run(
        self,
        namespace: int,
        indices: IntArray,
        settings: engine.Settings,
        pool: Any,
        progress: bool,
    ) -> engine.Replicates:
        # The engine on the given replicates, here or on the opened pool. A main run stores
        # its measures and stops on Ctrl-C (spec section 6.5); a pilot run does neither.
        main = namespace == MAIN
        job = functools.partial(
            engine.run,
            self.target,
            self.kernel,
            self.initial,
            rng=self.seed if main else namespace_seed(self.seed, namespace),
            **settings._asdict(),
        )
        kind = "replicate" if main else "pilot pair"
        slow = _SlowPairs(settings.lag, settings.m, kind)
        bar = Progress(len(indices), f"{kind}s") if progress else None
        interrupt = Interrupt()

        def monitor(status: engine.Status) -> bool:
            if bar is not None:
                bar.update(status.finished)
            slow(status)
            return interrupt.stopping

        with interrupt.catching() if main else nullcontext():
            try:
                replicates = execute(job, indices, pool, monitor)
            finally:
                if bar is not None:
                    bar.close()
        slow.after(replicates)
        if len(replicates.zero_density):
            warn(
                f"{len(replicates.zero_density)} {kind}s start where the target density "
                f"is zero (the first is {replicates.zero_density[0]}); the initial "
                "distribution should lie inside the support of the target",
                InputWarning,
            )
        return replicates

    def _check(
        self, k: int | str, m: int | str, lag: int | str, thin: int | None, max_iter: int | None
    ) -> dict[str, int]:
        # The settings given, those not "auto" or None, as integers, checked before any
        # tuning or worker starts: every bound when k is given, since only k needs the
        # pilot, and otherwise those of lag, thin and max_iter alone. Later runs keep the
        # first run's settings (D-21, DECISIONS #4 and #8).
        given = {}
        for name, value, default in (
            ("k", k, "auto"),
            ("m", m, "auto"),
            ("lag", lag, "auto"),
            ("thin", thin, None),
            ("max_iter", max_iter, None),
        ):
            if value != default:
                given[name] = integer(name, value, f"an integer or {default!r}")
        if self.settings is not None:
            for name, value in given.items():
                kept = getattr(self.settings, name)
                if value != kept:
                    raise ValueError(
                        f"{name} was {kept} in the earlier runs; a sampler keeps its "
                        f"settings, so start a new one for {name}={value}"
                    )
        elif "k" in given:
            _settings(given)
        else:
            for name in ("lag", "thin", "max_iter"):
                if name in given and given[name] < 1:
                    raise ValueError(f"{name} must be at least 1, not {given[name]}")
        return given

    def _configure(self, given: dict[str, int], pool: Any, progress: bool) -> engine.Settings:
        # The first run's settings, which tune the kernel and, for an automatic k, complete
        # the pilot to 100 pairs: meeting_times can leave fewer.
        self.tune()
        if "k" not in given:
            done = 0 if self.pilot is None else len(self.pilot)
            if done < PILOT_REPLICATES:
                self._meeting_times(PILOT_REPLICATES - done, PILOT_SETTINGS, pool, progress)
            given = given | {"k": choose_k(cast(FloatArray, self.pilot))}
        return _settings(given)

    def _check_memory(self, settings: engine.Settings, n_replicates: int) -> None:
        # Spec R-EXEC-7: estimate the atoms and distances to store before running, and warn
        # above 1 GB. The pilot's meeting times stand in for those of the run, their
        # tau - 1 for its tau - L.
        n_averaged = (settings.m - settings.k) // settings.thin + 1
        late = distances = 0.0
        if self.pilot is not None and np.isfinite(self.pilot).any():
            coupled = self.pilot[np.isfinite(self.pilot)] - 1
            late = float(np.mean(np.maximum(0, coupled - settings.k)))
            distances = float(np.mean(coupled))
        size = n_replicates * ((n_averaged + 2 * late) * (8 * self.ndim + 16) + 24 * distances)
        if size > 1e9:
            warn(
                f"the run will store about {size / 1e9:.1f} GB of atoms and distances; pass "
                "thin= to store fewer atoms, or run fewer replicates",
                InputWarning,
            )


def sample(
    log_prob: Callable[..., Any] | None,
    init: Any,
    n_replicates: int = 100,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    grad: Callable[..., Any] | None = None,
    vectorize: bool = False,
    kernel: Kernel | str = "auto",
    k: int | str = "auto",
    m: int | str = "auto",
    lag: int | str = "auto",
    thin: int | None = None,
    max_iter: int | None = None,
    names: Sequence[str] | None = None,
    pool: Any = None,
    progress: bool = False,
    rng: Seed = None,
) -> Result:
    r"""Unbiased estimates of expectations under a target, in one call (spec R-API-1).

    ``sample(log_prob, init, n, ...)`` is ``CoupledSampler(log_prob, init, ...).run(n,
    ...)``: it tunes the kernel's unset parameters, chooses ``k`` from 100 lag-1 pilot
    pairs when it is ``"auto"``, and runs ``n_replicates`` replicates of coupled chains.

    Parameters
    ----------
    log_prob : callable or None
        :math:`\log\pi` up to a constant, ``log_prob(x, *args, **kwargs)``, for ``x`` of
        shape ``(d,)``, or ``(n, d)`` with ``vectorize=True``; ``None`` for a Gibbs kernel.
    init
        The initial distribution :math:`\pi_0`: an object with ``rvs(random_state=rng)``,
        a callable ``init(rng)`` or an array of points of shape ``(n, d)``.
    n_replicates : int
        The number :math:`R` of replicates.
    args, kwargs, grad, vectorize, kernel, names, pool, progress, rng
        As for :class:`CoupledSampler`.
    k, m, lag, thin, max_iter
        As for :meth:`CoupledSampler.run`: ``"auto"`` chooses ``k`` from the pilot, then
        ``lag = max(1, k)`` and ``m = 10 k``; ``thin`` averages every ``thin``-th state;
        ``max_iter`` censors pairs still unmet at that time.

    Returns
    -------
    Result
        Its estimates are unbiased; it warns about the conditions of spec section 7.4.
    """
    sampler = CoupledSampler(
        log_prob,
        init,
        args=args,
        kwargs=kwargs,
        grad=grad,
        vectorize=vectorize,
        kernel=kernel,
        names=names,
        pool=pool,
        progress=progress,
        rng=rng,
    )
    return sampler.run(n_replicates, k=k, m=m, lag=lag, thin=thin, max_iter=max_iter)


class _SlowPairs:
    # Warns once about a pair still unmet at the slow-pair threshold (spec section 6.3):
    # from the engine's status as it happens, or from the meeting times after a run on a
    # pool, whose workers' statuses do not reach the parent. kind names the pairs.
    def __init__(self, lag: int, m: int, kind: str) -> None:
        self.threshold = diagnostics.slow_pair_threshold(lag, m)
        self.kind = kind
        self.warned = False

    def __call__(self, status: engine.Status) -> None:
        if status.t >= self.threshold:
            self._warn(status.slowest)

    def after(self, replicates: engine.Replicates) -> None:
        # A pair is unmet at the threshold when it meets, or is censored, after it.
        slow = replicates.index[replicates.stopped > self.threshold]
        if len(slow):
            self._warn(int(slow[0]))

    def _warn(self, replicate: int) -> None:
        if not self.warned:
            self.warned = True
            warn(
                f"{self.kind} {replicate} has not met after {self.threshold} iterations; "
                "the meeting times may be heavy-tailed",
                ConvergenceWarning,
            )


def _count(name: str, value: Any) -> int:
    # A number of replicates or pilot pairs to run.
    count = integer(name, value)
    if count < 1:
        raise ValueError(f"{name} must be at least 1, not {count}")
    return count


def _settings(given: dict[str, int]) -> engine.Settings:
    # The first run's settings from those given, k among them; lag = max(1, k) and
    # m = 10 k unless given (spec R-TUNE-2, D-21).
    k = given["k"]
    return engine.check_settings(
        engine.Settings(
            k=k,
            m=given.get("m", 10 * k),
            lag=given.get("lag", max(1, k)),
            thin=given.get("thin", 1),
            max_iter=given.get("max_iter"),
        )
    )


def _kernel(kernel: Kernel | str) -> Kernel:
    if isinstance(kernel, Kernel):
        return kernel
    if isinstance(kernel, str) and kernel in KERNELS:
        return KERNELS[kernel]()
    raise ValueError(
        f"kernel must be a Kernel or one of {', '.join(map(repr, KERNELS))}, not {kernel!r}"
    )
