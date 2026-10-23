"""The sampler: the automatic pipeline from a log-density to unbiased estimates."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.random import Generator, SeedSequence

from telescopic import diagnostics, engine
from telescopic._random import MAIN, PILOT, replicate_generator, root_seed
from telescopic._types import FloatArray
from telescopic.initial import as_initial, draw
from telescopic.kernels import Independence, Kernel, RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.tuning import TuningReport, choose_kml
from telescopic.warnings import ConvergenceWarning, InputWarning

KERNELS = ("auto", "rwmh", "independence")
PILOT_REPLICATES = 100
PILOT_MAX_ITER = 100_000


class CoupledSampler:
    r"""Unbiased MCMC with coupled chains: tuning, pilot meeting times and runs (spec R-API-1).

    The pipeline tunes the kernel's unset parameters on adaptive pilot chains, chooses
    ``k``, ``m`` and the lag from the meeting times of 100 lag-1 pilot pairs (spec
    R-TUNE-2), then runs replicates of coupled chains whose signed measures give unbiased
    estimates of any expectation under the target.

    Parameters
    ----------
    log_prob : callable
        :math:`\log\pi` up to a constant: ``log_prob(x, *args, **kwargs)`` for ``x`` of
        shape ``(d,)``, or of shape ``(n, d)`` with ``vectorize=True``.
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
        reflection-maximal coupling) and ``"independence"``.
    names : sequence of str, optional
        Parameter names for summaries.
    rng : None, int, numpy.random.SeedSequence or numpy.random.Generator
        The root of every random stream (spec section 6.1).
    """

    def __init__(
        self,
        log_prob: Callable[..., Any],
        init: Any,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        grad: Callable[..., Any] | None = None,
        vectorize: bool = False,
        kernel: Kernel | str = "auto",
        names: Sequence[str] | None = None,
        rng: int | SeedSequence | Generator | None = None,
    ) -> None:
        self.target = Target(log_prob, grad, args=args, kwargs=kwargs, vectorize=vectorize)
        self.initial = as_initial(init)
        self.kernel = _kernel(kernel)
        self.names = None if names is None else list(names)
        self.seed = root_seed(rng)
        self.tuning: TuningReport | None = None
        self.pilot: FloatArray | None = None
        self._settings: dict[str, Any] | None = None
        self._parts: list[engine.Replicates] = []
        self._next = {MAIN: 0, PILOT: 0}

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
                warnings.warn(disagreement, ConvergenceWarning, stacklevel=2)
        return self.tuning

    def meeting_times(
        self, n: int = PILOT_REPLICATES, *, max_iter: int = PILOT_MAX_ITER
    ) -> FloatArray:
        """Meeting times of ``n`` new pilot pairs with lag 1 (spec R-TUNE-2).

        Each call runs new pilot replicates, continuing their indices, and adds them to
        :attr:`pilot`.

        Parameters
        ----------
        n : int
        max_iter : int
            The cap; a pair still unmet there is censored.

        Returns
        -------
        ndarray, shape (n,)
            Meeting times, ``inf`` for a censored pair.
        """
        self.tune()
        indices = range(self._next[PILOT], self._next[PILOT] + n)
        replicates = engine.run(
            self.target,
            self.kernel,
            self.initial,
            indices,
            root=self.seed,
            namespace=PILOT,
            k=0,
            m=0,
            lag=1,
            max_iter=max_iter,
            store=False,
            monitor=_slow_pairs(1, 0),
        )
        _warn_about_zero_density(replicates)
        self._next[PILOT] += n
        times = replicates.meeting_time
        self.pilot = times if self.pilot is None else np.concatenate([self.pilot, times])
        return times

    def run(
        self,
        n_replicates: int,
        *,
        k: int | str = "auto",
        m: int | str = "auto",
        lag: int | str = "auto",
        thin: int = 1,
        max_iter: int | None = None,
    ) -> Result:
        """Run ``n_replicates`` more replicates and return the result of all of them so far.

        The first call tunes the kernel and runs the pilot when needed, and fixes ``k``,
        ``m``, the lag, ``thin`` and ``max_iter`` for the later calls (spec R-TUNE-2).

        Parameters
        ----------
        n_replicates : int
        k, m, lag : int or "auto"
            ``"auto"`` derives a value: the pilot's quantile for ``k``, then
            ``lag = k`` and ``m = 10 k`` (spec R-TUNE-2, D-21). A value given explicitly
            is kept.
        thin : int
            Average every ``thin``-th state from ``k`` (spec R-EST-5).
        max_iter : int, optional
            Censor pairs still unmet at this time; ``None`` runs every pair until it
            meets.

        Returns
        -------
        Result
            Unbiased for an ``n_replicates`` fixed in advance; extending a run after
            looking at it keeps consistency, not exact unbiasedness (spec section 3.3).
        """
        if n_replicates < 1:
            raise ValueError(f"n_replicates must be at least 1, not {n_replicates}")
        settings = self._configure(k, m, lag, thin, max_iter)
        self._check_memory(settings, self._next[MAIN] + n_replicates)
        indices = range(self._next[MAIN], self._next[MAIN] + n_replicates)
        part = engine.run(
            self.target,
            self.kernel,
            self.initial,
            indices,
            root=self.seed,
            namespace=MAIN,
            monitor=_slow_pairs(settings["lag"], settings["m"]),
            **settings,
        )
        _warn_about_zero_density(part)
        self._parts.append(part)
        self._next[MAIN] += n_replicates
        return Result(
            engine.merge(self._parts),
            names=self.names,
            kernel=self.kernel,
            initial=self.initial,
            seed=self.seed,
            tuning=self.tuning,
            pilot=self.pilot,
            **settings,
        )

    def _configure(
        self, k: int | str, m: int | str, lag: int | str, thin: int, max_iter: int | None
    ) -> dict[str, Any]:
        # The settings of every run of this sampler: fixed by the first call (D-21).
        if self._settings is not None:
            given = {"k": k, "m": m, "lag": lag, "thin": thin, "max_iter": max_iter}
            for name, value in given.items():
                default = 1 if name == "thin" else None if name == "max_iter" else "auto"
                if value != default and value != self._settings[name]:
                    raise ValueError(
                        f"{name} was {self._settings[name]} in the earlier runs; a sampler "
                        f"keeps its settings, so start a new one for {name}={value}"
                    )
            return self._settings
        self.tune()
        for name, value in (("k", k), ("m", m), ("lag", lag)):
            if value != "auto" and not isinstance(value, int | np.integer):
                raise ValueError(f"{name} must be an integer or 'auto', not {value!r}")
        if k == "auto":
            times = self.pilot if self.pilot is not None else self.meeting_times()
            k = choose_kml(times)[0]
        lag = k if lag == "auto" else lag
        m = 10 * int(k) if m == "auto" else m
        self._settings = {
            "k": int(k),
            "m": int(m),
            "lag": int(lag),
            "thin": thin,
            "max_iter": max_iter,
        }
        return self._settings

    def _check_memory(self, settings: dict[str, Any], n_replicates: int) -> None:
        # Spec R-EXEC-7: estimate the atoms to store before running, and warn above 1 GB.
        # The pilot's meeting times stand in for those of the run.
        d = len(draw(self.initial, replicate_generator(self.seed, MAIN, 0)))
        n_averaged = (settings["m"] - settings["k"]) // settings["thin"] + 1
        late = 0.0
        if self.pilot is not None and np.isfinite(self.pilot).any():
            met = self.pilot[np.isfinite(self.pilot)]
            late = float(np.mean(np.maximum(0, met - 1 - settings["k"])))
        size = n_replicates * (n_averaged + 2 * late) * (8 * d + 16)
        if size > 1e9:
            warnings.warn(
                f"the run will store about {size / 1e9:.1f} GB of atoms; pass thin= to "
                "store fewer, or run fewer replicates",
                InputWarning,
                stacklevel=3,
            )


def sample(
    log_prob: Callable[..., Any],
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
    thin: int = 1,
    max_iter: int | None = None,
    names: Sequence[str] | None = None,
    rng: int | SeedSequence | Generator | None = None,
) -> Result:
    """Unbiased estimates of expectations under a target, in one call (spec R-API-1).

    ``sample(log_prob, init, n)`` is ``CoupledSampler(log_prob, init, ...).run(n, ...)``;
    see :class:`CoupledSampler` for the parameters.

    Returns
    -------
    Result
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
        rng=rng,
    )
    return sampler.run(n_replicates, k=k, m=m, lag=lag, thin=thin, max_iter=max_iter)


def _slow_pairs(lag: int, m: int) -> Callable[[engine.Status], bool]:
    # A monitor that warns once when a pair is still unmet after max(10^4, 10 max(L, m))
    # iterations (spec section 6.3); it never stops the run.
    threshold = max(10_000, 10 * max(lag, m))
    warned = False

    def monitor(status: engine.Status) -> bool:
        nonlocal warned
        if not warned and status.t > threshold:
            warned = True
            warnings.warn(
                f"replicate {status.slowest} has not met after {threshold} iterations; the "
                "meeting times may be heavy-tailed",
                ConvergenceWarning,
                stacklevel=3,
            )
        return False

    return monitor


def _warn_about_zero_density(replicates: engine.Replicates) -> None:
    if len(replicates.zero_density):
        warnings.warn(
            f"{len(replicates.zero_density)} replicates start where the target density is "
            f"zero (the first is {replicates.zero_density[0]}); the initial distribution "
            "should lie inside the support of the target",
            InputWarning,
            stacklevel=3,
        )


def _kernel(kernel: Kernel | str) -> Kernel:
    if isinstance(kernel, Kernel):
        return kernel
    if kernel in ("auto", "rwmh"):
        return RandomWalk()
    if kernel == "independence":
        return Independence()
    raise ValueError(
        f"kernel must be a Kernel or one of {', '.join(map(repr, KERNELS))}, not {kernel!r}"
    )
