"""The sampler: the automatic pipeline from a log-density to unbiased estimates."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from typing import Any, cast

import numpy as np

from telescopic import diagnostics, engine
from telescopic._gradient_check import check_gradient
from telescopic._parallel import Interrupt, check_pool, check_workers, execute, opened
from telescopic._progress import Progress, tuning
from telescopic._random import MAIN, PILOT, namespace_seed, replicate_generator, root_seed
from telescopic._types import FloatArray, IntArray, Seed, flag, integer, parameter_names
from telescopic.initial import as_initial, draw
from telescopic.kernels import HMC, MALA, Independence, Kernel, RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.tuning import TuningReport, choose_k
from telescopic.warnings import ConvergenceWarning, InputWarning, warn

__all__ = ["CoupledSampler", "sample"]

KERNELS: dict[str, Callable[[], Kernel]] = {
    "auto": RandomWalk,
    "rwmh": RandomWalk,
    "mala": MALA,
    "hmc": HMC,
    "independence": Independence,
}
PILOT_REPLICATES = 100
# The default of check_grad on CoupledSampler and sample (DECISIONS #159).
CHECK_GRAD = True
PILOT_MAX_ITER = 100_000
# The meeting-time pilot: lag 1, with nothing stored (spec R-TUNE-2).
PILOT_SETTINGS = engine.Settings(0, 0, 1, max_iter=PILOT_MAX_ITER, store=False)
# A bound on the peak memory of a run and of its first estimates over the bytes the run
# stores, from d = 2 on for a random walk's rejection rate at thin = 1. In one or two
# dimensions the estimates pass it with thin above 1 or a kernel that rarely rejects,
# since Result.quantile sorts the atoms that are not consecutive repeats
# (notes/experiments/i73_memory_peak.out, DECISIONS #131, #183).
PEAK_PER_STORED = 3
# The warnings of spec section 7.4 that a run emits as they happen, not at its end, or as
# it ends for slow pairs on a pool.
_LIVE = ("slow", "tuning")


# Spec R-API-1 and R-TUNE-2; pool, sections 6.1 and 6.4; rng, section 6.1.
class CoupledSampler:
    r"""Unbiased MCMC with coupled chains: tuning, pilot meeting times and runs.

    The pipeline tunes the kernel's unset parameters on adaptive pilot chains, chooses
    ``k``, ``m`` and the lag from the meeting times of 100 lag-1 pilot pairs, then runs
    replicates of coupled chains whose signed measures give unbiased
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
        points of shape ``(n, d)``. Every :math:`\pi_0` from which the chain converges
        keeps the estimates unbiased (Guarantees and their limits, in :doc:`/theory`);
        one wider than the target and covering all its modes keeps them reliable.
    args : tuple or list
        Extra positional arguments of ``log_prob``, ``grad`` and
        ``log_prob_and_grad``, such as ``(data,)``.
    kwargs : mapping, optional
        Extra keyword arguments of these functions.
    grad : callable, optional
        :math:`\nabla\log\pi` with the signature of ``log_prob``. Only the kernels that
        need it, MALA and HMC, evaluate it; ``"auto"`` is random-walk Metropolis.
    log_prob_and_grad : callable, optional
        In place of ``grad``: the pair :math:`(\log\pi, \nabla\log\pi)` from one call
        with the signature of ``log_prob``, as JAX's ``value_and_grad`` gives, which
        MALA and HMC then call wherever they need a gradient; ``log_prob``, whose values
        it must match, still serves tuning's pilot chains. See :class:`Target`.
    vectorize : bool
        Whether ``log_prob``, ``grad`` and ``log_prob_and_grad`` take a batch of points.
        The kernels then also compute their matrix products for a batch of chains at
        once, with BLAS, so a run repeats bitwise only with the same seed and settings,
        ``pool`` and ``n_workers``, BLAS build and number of BLAS threads. Otherwise a
        replicate can differ by rounding, which can move its meeting time and its whole
        output, and the estimates agree within their error bars (Reproducibility, in
        :doc:`/guide/parallel`).
    kernel : Kernel or str
        A kernel, or one of ``"auto"`` and ``"rwmh"`` (random-walk Metropolis with the
        reflection-maximal coupling), ``"mala"`` and ``"hmc"`` (which need ``grad`` or
        ``log_prob_and_grad``) and ``"independence"``.
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
        depends on the pool. With a vectorized one, a run on a pool can differ by
        rounding from one in this process, through its chunks and through the one BLAS
        thread of its workers (Reproducibility, in :doc:`/guide/parallel`).
    n_workers : int, optional
        The number of workers of an executor or a pool, which sets how many chunks it
        gets: one per worker at least, and four at most unless a chunk would then hold
        more than 1,024 replicates, the most it ever holds; for a pool with only
        ``map``, waves of one per worker. Given, it replaces the size that the pool
        reports; ``None`` reads that size, or else counts the CPUs this process may
        use, which leaves idle the workers of a larger pool that does not report its
        size, such as a cluster's. Only for an executor or a pool: with ``pool`` None or
        an integer it raises ``ValueError``. State the pool's real number of workers: a
        larger value makes smaller chunks, which run fewer replicates together. With
        a log-density evaluated point by point, no replicate depends on it.
    progress : bool
        Show the progress of tuning, of the pilot and of each run: a tqdm bar when
        tqdm is installed, lines on stderr otherwise.
    check_grad : bool
        Compare the gradient with differences of the log-density at 8 draws of the
        initial distribution when the sampler is built, if the kernel evaluates a
        gradient, and warn where they disagree (Checking the gradient, in
        :doc:`/guide/log-density`): ``grad`` with differences of ``log_prob``, or the
        gradient of ``log_prob_and_grad`` with differences of its own values. With
        ``grad`` it costs 40 evaluations of ``log_prob`` and 8 of ``grad`` for float64
        values, 8 more of each for a float32 ``grad`` and again for a float32
        ``log_prob``, and a second sweep for a ``log_prob`` whose values narrow after the
        first draw; with ``log_prob_and_grad``, 40 evaluations of it for float64 values,
        and for narrower ones at most as many as of ``log_prob`` with the same dtypes,
        unless its gradient is narrower only where its value is ``-inf``.
        It draws 48 points of the initial distribution and changes no result.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root of every random stream.

    Attributes
    ----------
    kernel : Kernel
        The kernel, replaced by its tuned copy when tuned.
    gradient_check : telescopic.diagnostics.GradientCheck or None
        What the gradient check compared; ``None`` when it did not run.
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
        log_prob_and_grad: Callable[..., Any] | None = None,
        vectorize: bool = False,
        kernel: Kernel | str = "auto",
        names: Sequence[str] | None = None,
        pool: Any = None,
        n_workers: int | None = None,
        progress: bool = False,
        check_grad: bool = CHECK_GRAD,
        rng: Seed = None,
    ) -> None:
        self.pool, self.progress = check_pool(pool), flag("progress", progress)
        self.n_workers = check_workers(n_workers, self.pool)
        check_grad = flag("check_grad", check_grad)
        self.target = Target(
            log_prob,
            grad,
            log_prob_and_grad=log_prob_and_grad,
            args=args,
            kwargs=kwargs,
            vectorize=vectorize,
        )
        self.initial = as_initial(init)
        if isinstance(kernel, str) and kernel in KERNELS:
            kernel = KERNELS[kernel]()
        if not isinstance(kernel, Kernel):
            error = ValueError if isinstance(kernel, str) else TypeError
            known = ", ".join(map(repr, KERNELS))
            raise error(f"kernel must be a Kernel or one of {known}, not {kernel!r}")
        self.kernel = kernel
        self.kernel.check_target(self.target)
        self.seed = root_seed(rng)
        self._rng_given = rng is not None  # resume refuses an rng that disagrees
        # One draw from a throwaway generator gives the dimension, so that the kernel's and
        # the names are checked now rather than after tuning.
        self.ndim = len(draw(self.initial, replicate_generator(self.seed, MAIN, 0)))
        self.kernel.check_ndim(self.ndim)
        # The dimension comes from a draw, so a length error blames the initial distribution.
        draws = "the initial distribution draws points"
        self.names = None if names is None else parameter_names(names, self.ndim, draws)
        self.tuning: TuningReport | None = None
        self.pilot: FloatArray | None = None
        self._pilot_zero_density = 0  # pilot pairs started where the density is zero
        self._replicates: engine.Replicates | None = None
        # From its own stream, before any tuning, so that it changes no result.
        self.gradient_check: diagnostics.GradientCheck | None = None
        if check_grad and self.kernel.needs_grad:
            self.gradient_check = check_gradient(
                cast(Callable[..., Any], log_prob),
                grad,
                self.initial,
                self.seed,
                args=self.target.args,
                kwargs=self.target.kwargs,
                vectorize=self.target.vectorize,
                log_prob_and_grad=log_prob_and_grad,
            )

    @property
    def settings(self) -> engine.Settings | None:
        """The settings of the runs, fixed by the first; ``None`` before it.

        Returns
        -------
        telescopic.engine.Settings or None
        """
        return None if self._replicates is None else self._replicates.settings

    # Spec section 8.1; seed and indices, section 6.1; interrupted results, section 6.5.
    def resume(self, result: Result) -> None:
        """Continue the run of a result: its replicates, seed, settings, tuning and pilot.

        The next call of :meth:`run` first runs the requested replicates that ``result``
        lacks, then new indices above the highest it requested, with its ``k``, ``m``,
        ``lag``, ``thin`` and ``max_iter``, and returns the result of all of them.

        The sampler must have been built with the target, the initial distribution and
        the kernel of that run. Only the dimension and the kernel's class can be checked;
        a different target or kernel goes unnoticed and mixes replicates of two runs. Pass
        ``result.kernel`` when it is a kernel. A loaded result holds only the kernel's
        description: pass the kernel as it was built, untuned, and ``resume`` tunes it
        again from the result's seed. It warns when the kernel's tuning or description
        differs from the result's; everything is adopted before any warning.

        Parameters
        ----------
        result : Result
            The result to continue: returned by a run, loaded, or merged.
        """
        if not isinstance(result, Result):
            raise TypeError(f"result must be a telescopic.Result, not {type(result).__name__}")
        for used, done in (
            ("run replicates", self._replicates is not None),
            ("run pilot pairs", self.pilot is not None),
            ("tuned its kernel", self.tuning is not None),
        ):
            if done:
                raise ValueError(f"resume needs a new sampler: this one has already {used}")
        if result.ndim != self.ndim:
            raise ValueError(
                f"the result has dimension {result.ndim}, but the initial distribution "
                f"draws points of dimension {self.ndim}"
            )
        # Kernels are compared by their descriptions, which a loaded result holds and which
        # start with the class's name; a description without one (an object's default
        # repr) is not compared.
        ran = None if result.kernel is None else str(result.kernel)
        if ran is not None and "(" in ran:
            ran, mine = ran.split("(", 1)[0], str(self.kernel).split("(", 1)[0]
            if ran != mine:
                raise ValueError(
                    f"the result was run with {ran}, but the sampler's kernel is {mine}"
                )
        if self._rng_given and engine._seed_key(self.seed) != engine._seed_key(result.seed):
            raise ValueError(
                "rng was given, but the result has another seed; omit rng to resume it"
            )
        if self.names is not None and self.names != result.names:
            raise ValueError(f"names {self.names} differ from the result's {result.names}")
        if not self.kernel.is_tuned and result.tuning is None:
            raise ValueError(
                "the result holds no tuning report, so the sampler's kernel must have every "
                "parameter set; pass the kernel the result was run with"
            )
        seed, kernel, report = root_seed(result.seed), self.kernel, result.tuning
        if not kernel.is_tuned:
            # From the tuning namespace of the result's seed, as the run tuned it (spec
            # section 6.1). No tolerance: rounding can move a pilot's accept decision.
            with tuning(self.progress):
                kernel, report = kernel.tune(self.target, self.initial, seed)
        # Everything is adopted at once, before any warning, so that a tuning that raises,
        # or a warning made an error, leaves the sampler either as it was or resumed.
        self.seed, self.kernel, self.tuning = seed, kernel, report
        self.names, self.pilot = result.names, result.pilot
        self._replicates = result._replicates
        differs = False
        if report is not result.tuning:  # tuned again
            new, kept = cast(TuningReport, report), cast(TuningReport, result.tuning)
            disagreement = diagnostics.tuning_disagreement(new)
            if disagreement:
                warn(disagreement, ConvergenceWarning)
            differs = not (
                np.array_equal(new.mean, kept.mean) and np.array_equal(new.cov, kept.cov)
            )
        if differs or (result.kernel is not None and str(kernel) != str(result.kernel)):
            warn(
                "the sampler's kernel differs from the result's: the target, the initial "
                "distribution or the kernel differ from the run's, or the kernel was tuned "
                "again on another machine, library version or number of BLAS threads; the "
                "resumed run keeps the sampler's kernel",
                InputWarning,
            )

    # Spec section 5.3.
    def tune(self) -> TuningReport | None:
        """Tune the kernel's unset parameters on pilot chains, once, and freeze the kernel.

        Warns when the chains disagree, a sign of several modes or of chains that have not
        converged; chains that agree do not show that the target has one mode.

        Returns
        -------
        TuningReport or None
            ``None`` when the kernel had nothing to tune.
        """
        return self._tune(self.progress)

    def _tune(self, progress: bool) -> TuningReport | None:
        if not self.kernel.is_tuned:
            with tuning(progress):
                self.kernel, self.tuning = self.kernel.tune(self.target, self.initial, self.seed)
            disagreement = diagnostics.tuning_disagreement(self.tuning)
            if disagreement:
                warn(disagreement, ConvergenceWarning)
        return self.tuning

    # Spec R-TUNE-2 and section 6.3.
    def meeting_times(
        self, n: int = PILOT_REPLICATES, *, max_iter: int = PILOT_MAX_ITER
    ) -> FloatArray:
        """Meeting times of ``n`` new pilot pairs with lag 1.

        Each call runs new pilot replicates, continuing their indices, and adds them to
        ``pilot``.

        Parameters
        ----------
        n : int
            Pilot pairs to add, at least 1.
        max_iter : int
            The cap, above 1; a pair still unmet there is censored.

        Returns
        -------
        ndarray, shape (n,)
            Meeting times, ``inf`` for a censored pair.
        """
        n = integer("n", n, minimum=1)
        # Before check_settings, which takes None for no cap: an uncapped pilot can run
        # forever where the coupling never meets (D-23).
        max_iter = integer("max_iter", max_iter)
        settings = engine.check_settings(PILOT_SETTINGS._replace(max_iter=max_iter))
        with opened(self.pool) as pool:
            return self._meeting_times(n, settings, pool, self.progress)

    def _meeting_times(
        self, n: int, settings: engine.Settings, pool: Any, progress: bool
    ) -> FloatArray:
        self._tune(progress)
        start = 0 if self.pilot is None else len(self.pilot)
        indices = np.arange(start, start + n)
        # A pilot keeps nothing of a run that an error stopped (spec section 6.5).
        replicates, error, _ = self._run(PILOT, indices, settings, pool, progress)
        if error is not None:
            try:
                raise error
            finally:
                del error  # the traceback keeps this frame
        replicates = cast(engine.Replicates, replicates)
        times = replicates.meeting_time
        self.pilot = times if self.pilot is None else np.concatenate([self.pilot, times])
        self._pilot_zero_density += len(replicates.zero_density)
        return times

    # Spec R-TUNE-2 and D-21, sections 3.3, 6.5 and 7.4, R-EST-5.
    def run(
        self,
        n_replicates: int,
        *,
        k: int | str = "auto",
        m: int | str = "auto",
        lag: int | str = "auto",
        thin: int | None = None,
        max_iter: int | None = None,
        keep: int | None = None,
        progress: bool | None = None,
    ) -> Result:
        """Run ``n_replicates`` more replicates and return the result of all of them so far.

        The first call tunes the kernel, completes the pilot to 100 pairs for an automatic
        ``k``, and fixes the settings of the later calls. The result warns about the
        conditions listed under Warnings, in :doc:`/reference/warnings`.

        The first Ctrl-C during the replicates lets those in progress finish and returns a
        result flagged ``interrupted``; the next call runs the missing replicates first. A
        second Ctrl-C raises ``KeyboardInterrupt`` at once, without a result.

        An error during the replicates, such as one of the log-density, is raised again
        once the sampler keeps the replicates finished before it, with its attribute
        ``telescopic_result`` set to the result of all of them so far (``None`` if none
        finished) and a note; the next call runs the missing replicates first. An error
        in tuning or in the pilot is raised as it is.

        Parameters
        ----------
        n_replicates : int
            The number of replicates to add, at least 1.
        k, m, lag : int or "auto"
            ``"auto"`` derives a value on the first call: the pilot's quantile for ``k``,
            then ``lag = max(1, k)`` and ``m = 10 k``. A value given
            explicitly is kept. On later calls ``"auto"`` keeps the first call's value.
        thin : int, optional
            Average every ``thin``-th state from ``k``: 1 on the first
            call, and the first call's value on later ones.
        max_iter : int, optional
            Censor pairs still unmet at this time. ``None`` runs every pair until it
            meets on the first call, and keeps the first call's cap on later ones.
        keep : int, optional
            Store the signed measures and distances of the replicates of index below
            ``keep`` only, and the moments of every replicate, from which the mean, the
            variances and the covariance (up to 100 dimensions) come; the quantiles,
            histograms and :meth:`~telescopic.Result.estimate` come from the kept
            replicates. ``None`` stores every replicate's signed measure on the first
            call, and keeps the first call's setting on later ones.
        progress : bool, optional
            Show the progress of the run and of the tuning and the pilot it runs;
            ``None`` takes the sampler's setting.

        Returns
        -------
        Result
            Unbiased for an ``n_replicates`` fixed in advance, unless ``max_iter``
            censors a pair, which biases it and warns, or a Ctrl-C interrupts the run,
            which leaves it consistent and flagged ``interrupted``; extending a run after
            looking at it keeps consistency, not exact unbiasedness.

        Examples
        --------
        A second call runs 25 more replicates and returns the result of all 50:

        >>> from telescopic import CoupledSampler, examples
        >>> example = examples.gaussian(2)
        >>> sampler = CoupledSampler(
        ...     example.log_prob, example.initial, vectorize=True, rng=0
        ... )
        >>> sampler.run(25).n_replicates, sampler.run(25).n_replicates
        (25, 50)
        """
        n_replicates = integer("n_replicates", n_replicates, minimum=1)
        given = self._check(k, m, lag, thin, max_iter, keep)
        show = self.progress if progress is None else flag("progress", progress)
        interrupt = Interrupt()
        with ExitStack() as catching:
            # Worker processes start before tuning, which runs here meanwhile, and serve
            # the pilot and the run (DECISIONS #23).
            with opened(self.pool) as pool:
                settings = self.settings or self._configure(given, pool, show)
                done = 0 if self._replicates is None else len(self._replicates.index)
                self._check_memory(settings, done + n_replicates)
                indices = self._next_indices(n_replicates)
                # From the start of the replicates until the result returns, the first
                # Ctrl-C lets the run finish (spec section 6.5): after the replicates,
                # while the workers stop and the result is built, it would lose them.
                catching.enter_context(interrupt.catching())
                part, error, others = self._run(MAIN, indices, settings, pool, show, interrupt)
            # The finished replicates are kept before anything can raise, so that an error,
            # or a warning made one, leaves them in the sampler (spec section 6.5).
            if part is not None and len(part.index):
                self._replicates = (
                    part if self._replicates is None else engine.merge([self._replicates, part])
                )
                if error is not None:
                    note = self._stopped(part, len(indices), others)
                    engine._attach(error, "telescopic_result", self._result(), note)
            elif error is not None:
                note = f"{engine._STOPPED} the run before any of its replicates finished"
                engine._attach(error, "telescopic_result", None, note)
            if error is not None:
                try:
                    raise error
                finally:
                    del error  # the traceback keeps this frame
            result = self._result()
            for issue in diagnostics.issues(result):
                if issue.kind not in _LIVE:
                    warn(issue.message, ConvergenceWarning)
        return result

    def _result(self) -> Result:
        return Result(
            cast(engine.Replicates, self._replicates),
            names=self.names,
            kernel=self.kernel,
            initial=self.initial,
            tuning=self.tuning,
            pilot=self.pilot,
        )

    def _stopped(self, part: engine.Replicates, n: int, others: int) -> str:
        # The note on an error that stopped a run of n replicates, of which part finished,
        # once the sampler keeps them.
        replicates = cast(engine.Replicates, self._replicates)
        missing = len(np.setdiff1d(replicates.requested, replicates.index))
        abandoned = part.abandoned
        lines = [
            f"{engine._STOPPED} the run with {len(part.index):,} of its {n:,} replicates finished"
        ]
        if len(abandoned):
            lines[0] += (
                f"; the {len(abandoned):,} in progress (indices {abandoned[0]:,} to "
                f"{abandoned[-1]:,}) were abandoned"
            )
        lines.append(
            "The sampler keeps the finished replicates, and its next run starts with the "
            f"{missing:,} missing ones; error.telescopic_result holds the result of all its "
            "replicates so far, which CoupledSampler.resume continues in a new sampler. An "
            "error that recurs at a replicate will recur there."
        )
        if len(abandoned):
            lines.append(
                "Leaving abandoned replicates out would bias the estimates, since the "
                "replicates still running when an error comes tend to be those with long "
                "meeting times."
            )
        if others:
            lines.append(f"Errors stopped {others:,} other chunks of the run too.")
        return f"{lines[0]}. {' '.join(lines[1:])}"

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
        interrupt: Interrupt | None = None,
    ) -> engine._Outcome:
        # The engine on the given replicates, here or on the opened pool, and the error
        # that stopped it, if any. A main run stores its measures and stops on Ctrl-C,
        # through interrupt (spec section 6.5); a pilot run does neither.
        main = namespace == MAIN
        job = functools.partial(
            engine._run,
            self.target,
            self.kernel,
            self.initial,
            rng=self.seed if main else namespace_seed(self.seed, namespace),
            **settings._asdict(),
        )
        kind = "replicate" if main else "pilot pair"
        outside = _ZeroDensityStarts(kind)
        slow = _SlowPairs(settings.lag, settings.m, kind, interrupt)
        zeroed = _ZeroedGradients(self.target, kind)
        bar = Progress(len(indices), f"{kind}s") if progress else None

        def monitor(status: engine.Status) -> bool:
            if bar is not None:
                bar.update(status.finished)
            outside(status)
            slow(status)
            zeroed()
            return interrupt is not None and interrupt.stopping

        try:
            replicates, error, others = execute(job, indices, pool, monitor, self.n_workers)
        finally:
            if bar is not None:
                bar.close()
        if replicates is not None:
            error = error or _checked(replicates, outside, slow, zeroed)
        return engine._Outcome(replicates, error, others)

    def _check(
        self,
        k: int | str,
        m: int | str,
        lag: int | str,
        thin: int | None,
        max_iter: int | None,
        keep: int | None,
    ) -> dict[str, int]:
        # The settings given, those not "auto" or None, as integers, checked before any
        # tuning or worker starts: with an automatic k, at k = 0, whose bounds hold for
        # every k. Later runs keep the first run's settings, keep among them (D-21,
        # DECISIONS #4, #8 and #242).
        given = {}
        for name, value, default in (
            ("k", k, "auto"),
            ("m", m, "auto"),
            ("lag", lag, "auto"),
            ("thin", thin, None),
            ("max_iter", max_iter, None),
        ):
            # The type first: == compares an array elementwise.
            if not (isinstance(value, type(default)) and value == default):
                given[name] = integer(name, value, f"an integer or {default!r}")
        if keep is not None:
            given["keep"] = integer("keep", keep, "None or an integer >= 0", minimum=0)
        if self.settings is not None:
            for name, value in given.items():
                kept = getattr(self.settings, name)
                if value != kept:
                    raise ValueError(
                        f"{name} was {kept} in the earlier runs; a sampler keeps its "
                        f"settings, so start a new one for {name}={value}"
                    )
        else:
            _settings({"k": 0} | given)
        return given

    def _configure(self, given: dict[str, int], pool: Any, progress: bool) -> engine.Settings:
        # The first run's settings, which tune the kernel and, for an automatic k, complete
        # the pilot to 100 pairs: meeting_times can leave fewer.
        self._tune(progress)
        if "k" not in given:
            done = 0 if self.pilot is None else len(self.pilot)
            if done < PILOT_REPLICATES:
                self._meeting_times(PILOT_REPLICATES - done, PILOT_SETTINGS, pool, progress)
            given = given | {"k": self._choose_k(cast(FloatArray, self.pilot))}
        return _settings(given)

    def _choose_k(self, pilot: FloatArray) -> int:
        # choose_k's advice of a wider initial distribution makes zero-density starts,
        # from which a pair may never meet, more frequent.
        try:
            return choose_k(pilot)
        except ValueError as error:
            if not self._pilot_zero_density:
                # self.tuning is None when the kernel came with every parameter.
                if self.kernel.needs_grad and self.tuning is not None:
                    error.add_note(
                        "with MALA or HMC, an automatic step size can fail to couple: try a "
                        "step_size checked on pilot pairs, or, for HMC with a given n_steps "
                        "or cov, leaving them unset (When the automatic step fails, in the "
                        "guide 'Check and set a step size')"
                    )
                raise
            raise ValueError(
                "k cannot be chosen: too many pilot pairs never met within their cap, and "
                f"{self._pilot_zero_density} of the {len(pilot)} started where the target "
                "density is zero, from which a pair may never meet; draw the initial "
                "distribution inside the support of the target, or give k, lag and m "
                "explicitly"
            ) from error

    def _check_memory(self, settings: engine.Settings, n_replicates: int) -> None:
        # Spec R-EXEC-7: estimate the atoms and distances to store before running, and warn
        # when a run and its first estimates, which peak at about three times that, would
        # pass 1 GB. The pilot's meeting times stand in for those of the run, their tau - 1
        # for its tau - L; without a pilot, the run's pairs count as meeting at once. With
        # keep, only the kept replicates store atoms, and the peak adds the moments of every
        # replicate, 8 (p + d) bytes each, the lanes' sums in each engine (one per worker of
        # an integer pool) and the block temporary, at most 1 MiB.
        coupled = np.zeros(1)
        if self.pilot is not None and np.isfinite(self.pilot).any():
            coupled = self.pilot[np.isfinite(self.pilot)] - 1
        keep = n_replicates if settings.keep is None else min(settings.keep, n_replicates)
        each = engine.stored_bytes(settings._replace(keep=None), self.ndim, coupled)
        stored = keep * each / len(coupled)
        peak = PEAK_PER_STORED * stored
        stored_what = "atoms and distances"
        if settings.keep is not None:
            engines = self.pool if isinstance(self.pool, int) else 1
            moments = engine.stored_bytes(
                settings._replace(keep=0), self.ndim, np.zeros(n_replicates)
            )
            stored += moments
            peak += moments + engines * engine.LANES * engine._lane_bytes(self.ndim) + 2**20
            stored_what = "atoms, distances and moments"
        if peak > 1e9:
            # In one or two dimensions the peak can reach twice the factor's figure.
            upto = f" to {2 * peak / 1e9:.1f}" if self.ndim <= 2 else ""
            warn(
                f"the run will store about {stored / 1e9:.1f} GB of {stored_what}, and with "
                f"its first estimates take about {peak / 1e9:.1f}{upto} GB; pass thin= or "
                "keep= to store fewer atoms, or run fewer replicates",
                InputWarning,
            )


# Spec R-API-1 and section 7.4.
def sample(
    log_prob: Callable[..., Any] | None,
    init: Any,
    n_replicates: int = 100,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    grad: Callable[..., Any] | None = None,
    log_prob_and_grad: Callable[..., Any] | None = None,
    vectorize: bool = False,
    kernel: Kernel | str = "auto",
    k: int | str = "auto",
    m: int | str = "auto",
    lag: int | str = "auto",
    thin: int | None = None,
    max_iter: int | None = None,
    keep: int | None = None,
    names: Sequence[str] | None = None,
    pool: Any = None,
    n_workers: int | None = None,
    progress: bool = False,
    check_grad: bool = CHECK_GRAD,
    rng: Seed = None,
) -> Result:
    r"""Unbiased estimates of expectations under a target, in one call.

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
        The number :math:`R` of replicates, at least 2.
    args : tuple or list
        Extra positional arguments of ``log_prob``, ``grad`` and
        ``log_prob_and_grad``, such as ``(data,)``.
    kwargs : mapping, optional
        Extra keyword arguments of these functions.
    grad : callable, optional
        :math:`\nabla\log\pi` with the signature of ``log_prob``. Only the kernels that
        need it, MALA and HMC, evaluate it; ``"auto"`` is random-walk Metropolis.
    log_prob_and_grad : callable, optional
        In place of ``grad``: the pair :math:`(\log\pi, \nabla\log\pi)` from one call;
        see :class:`CoupledSampler`.
    vectorize : bool
        Whether ``log_prob``, ``grad`` and ``log_prob_and_grad`` take a batch of points.
        The kernels then also compute their matrix products for a batch of chains at
        once, with BLAS, so a run repeats bitwise only with the same seed and settings,
        ``pool`` and ``n_workers``, BLAS build and number of BLAS threads. Otherwise a
        replicate can differ by rounding, which can move its meeting time and its whole
        output, and the estimates agree within their error bars (Reproducibility, in
        :doc:`/guide/parallel`).
    kernel : Kernel or str
        A kernel, or one of ``"auto"`` and ``"rwmh"`` (random-walk Metropolis with the
        reflection-maximal coupling), ``"mala"`` and ``"hmc"`` (which need ``grad`` or
        ``log_prob_and_grad``) and ``"independence"``.
    k, m, lag : int or "auto"
        ``"auto"`` chooses ``k`` from the pilot's meeting times, then ``lag = max(1, k)``
        and ``m = 10 k``. A value given explicitly is kept.
    thin : int, optional
        Average every ``thin``-th state from ``k``; ``None`` averages every state.
    max_iter : int, optional
        Censor pairs still unmet at this time; ``None`` runs every pair until it meets.
    keep : int, optional
        Store the signed measures of the replicates of index below ``keep`` only, and the
        moments of every replicate; see :meth:`CoupledSampler.run`. ``None`` stores every
        replicate's signed measure.
    names : sequence of str, optional
        Parameter names for summaries.
    pool : None, int, executor or pool
        Where the pilot pairs and the replicates run: ``None`` in this process; an
        integer, that many worker processes; or an object with ``submit``, such as a
        :class:`concurrent.futures.Executor`, or with ``map``, such as an emcee or
        schwimmbad pool. See :class:`CoupledSampler` for how a pool runs them.
    n_workers : int, optional
        The number of workers of an executor or a pool, which sets how many chunks it
        gets; ``None`` reads the pool's size, or else counts the CPUs this process may
        use. See :class:`CoupledSampler`.
    progress : bool
        Show the progress of the pilot and of the run: a tqdm bar when tqdm is installed,
        lines on stderr otherwise.
    check_grad : bool
        Compare the gradient with differences of the log-density before tuning, if the
        kernel evaluates a gradient, and warn where they disagree; see
        :class:`CoupledSampler`.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root of every random stream.

    Returns
    -------
    Result
        Its estimates are unbiased unless ``max_iter`` censors a pair, which biases them
        and warns, or a Ctrl-C interrupts the run, which leaves them consistent and flags
        the result ``interrupted``. It warns about the conditions listed under Warnings,
        in :doc:`/reference/warnings`. An error during the replicates is raised with the
        result of those finished before it as its attribute ``telescopic_result``, which
        :meth:`CoupledSampler.resume` continues.

    Examples
    --------
    One call runs 50 replicates on a correlated Gaussian in two dimensions and estimates
    the mean of each coordinate:

    >>> from telescopic import examples, sample
    >>> example = examples.gaussian(2)
    >>> result = sample(
    ...     example.log_prob, example.initial,
    ...     n_replicates=50, vectorize=True, rng=0,
    ... )
    >>> result.n_replicates, result.mean().value.shape
    (50, (2,))
    """
    # Checked before tuning: the estimates and the summary need 2 replicates that met.
    integer("n_replicates", n_replicates, minimum=2)
    sampler = CoupledSampler(
        log_prob,
        init,
        args=args,
        kwargs=kwargs,
        grad=grad,
        log_prob_and_grad=log_prob_and_grad,
        vectorize=vectorize,
        kernel=kernel,
        names=names,
        pool=pool,
        n_workers=n_workers,
        progress=progress,
        check_grad=check_grad,
        rng=rng,
    )
    return sampler.run(n_replicates, k=k, m=m, lag=lag, thin=thin, max_iter=max_iter, keep=keep)


def _checked(
    replicates: engine.Replicates,
    outside: _ZeroDensityStarts,
    slow: _SlowPairs,
    zeroed: _ZeroedGradients,
) -> Exception | None:
    # The checks after a run, and the exception one raised, as a warning made an error:
    # it stops the run as any error does, with every replicate kept (spec section 6.5).
    with engine._Caught() as caught:
        outside.after(replicates)
        slow.after(replicates)
        zeroed.after(replicates)
    return caught.error


class _ZeroDensityStarts:
    # Warns once about pairs that start where the target density is zero (spec section
    # 4.2): from the engine's status as soon as one starts, or from the replicates after a
    # run on a pool, whose workers' statuses do not reach the parent. A chain there moves
    # only to a proposal of finite density, so a start far outside the support can keep a
    # run waiting forever. kind names the pairs.
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.warned = False

    def __call__(self, status: engine.Status) -> None:
        self._warn(status.zero_density, status.first_zero_density, " so far")

    def after(self, replicates: engine.Replicates) -> None:
        starts = replicates.zero_density
        self._warn(len(starts), int(starts[0]) if len(starts) else -1, "")

    def _warn(self, count: int, first: int, so_far: str) -> None:
        if count and not self.warned:
            self.warned = True
            warn(
                f"{self.kind}s start where the target density is zero ({count}{so_far}, the "
                f"first is {first}): a chain there moves only to a proposal of finite "
                "density, so its pair may never meet; draw the initial distribution inside "
                "the support of the target",
                InputWarning,
            )


class _SlowPairs:
    # Warns once about a pair still unmet at the slow-pair threshold (spec section 6.3):
    # from the engine's status as it happens, or from the meeting times after a run on a
    # pool, whose workers' statuses do not reach the parent. kind names the pairs, and
    # interrupt is the run's Ctrl-C handler, None for a pilot.
    def __init__(self, lag: int, m: int, kind: str, interrupt: Interrupt | None = None) -> None:
        self.threshold = diagnostics.slow_pair_threshold(lag, m)
        self.kind, self.interrupt = kind, interrupt
        self.warned = False

    def __call__(self, status: engine.Status) -> None:
        if status.t >= self.threshold:
            # The first Ctrl-C lets the replicates in progress finish (spec section 6.5),
            # this one too, so it does not end the wait.
            waiting = ", and the run waits for it"
            interrupt = self.interrupt
            if interrupt is not None and interrupt.listening and not interrupt.stopping:
                waiting += (
                    ": a first Ctrl-C starts no new replicate but lets this one finish, "
                    "and a second stops the run without a result"
                )
            self._warn(status.slowest, waiting, status.slowest_zero_density)

    def after(self, replicates: engine.Replicates) -> None:
        # A pair is unmet at the threshold when it meets after it, or is censored at or
        # after it: the engine compared its chains at the cap.
        earliest = replicates.stopped + np.isinf(replicates.meeting_time)
        slow = replicates.index[earliest > self.threshold]
        if len(slow):
            self._warn(int(slow[0]), "", bool(np.isin(slow[0], replicates.zero_density)))

    def _warn(self, replicate: int, waiting: str, outside: bool) -> None:
        if not self.warned:
            self.warned = True
            cause = (
                "it started where the target density is zero, so draw the initial "
                "distribution inside the support of the target"
                if outside
                else "the meeting times may be heavy-tailed, so look at the kernel and the "
                "initial distribution"
            )
            warn(
                f"{self.kind} {replicate} has not met after {self.threshold} "
                f"iterations{waiting}; {cause}",
                ConvergenceWarning,
            )


class _ZeroedGradients:
    # Warns once about gradients taken as 0 where the log-density is finite (spec
    # R-INPUT-1): as soon as this process's target counts one, or from the replicates
    # after a run on a pool, whose workers hold their own targets. kind names the pairs.
    def __init__(self, target: Target, kind: str) -> None:
        self.target, self.kind = target, kind
        self.before = target.n_grad_zeroed, target.n_grad
        self.warned = False

    def __call__(self) -> None:
        zeroed = self.target.n_grad_zeroed - self.before[0]
        if zeroed:
            self._warn(zeroed, self.target.n_grad - self.before[1])

    def after(self, replicates: engine.Replicates) -> None:
        if replicates.n_grad_zeroed:
            self._warn(replicates.n_grad_zeroed, replicates.n_grad)

    def _warn(self, zeroed: int, evaluated: int) -> None:
        if not self.warned:
            self.warned = True
            warn(
                "gradients that were not finite where the log-density is were taken as 0 "
                f"in {zeroed} of {evaluated} evaluations of the {self.kind}s: MALA and HMC "
                "stay exact but lose their drift there, which slows the chains and can keep "
                "HMC pairs from meeting; compute the gradient in log space, as with "
                "scipy.special.logsumexp, or check it",
                InputWarning,
            )


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
            keep=given.get("keep"),
        )
    )
