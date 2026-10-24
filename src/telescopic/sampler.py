"""The sampler: the automatic pipeline from a log-density to unbiased estimates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from telescopic import diagnostics, engine
from telescopic._random import MAIN, PILOT, replicate_generator, root_seed
from telescopic._types import FloatArray, Seed
from telescopic.estimators import averaging_times
from telescopic.initial import as_initial, draw
from telescopic.kernels import Independence, Kernel, RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.tuning import TuningReport, choose_k
from telescopic.warnings import ConvergenceWarning, InputWarning, warn

KERNELS: dict[str, Callable[[], Kernel]] = {
    "auto": RandomWalk,
    "rwmh": RandomWalk,
    "independence": Independence,
}
PILOT_REPLICATES = 100
PILOT_MAX_ITER = 100_000
# The warnings of spec section 7.4 that a run emits as they happen, not at its end.
_LIVE = ("slow", "tuning")


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

    Attributes
    ----------
    settings : dict or None
        ``k``, ``m``, ``lag``, ``thin`` and ``max_iter``, fixed by the first run.
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
        rng: Seed = None,
    ) -> None:
        self.target = Target(log_prob, grad, args=args, kwargs=kwargs, vectorize=vectorize)
        self.initial = as_initial(init)
        self.kernel = _kernel(kernel)
        self.seed = root_seed(rng)
        # One draw from a throwaway generator gives the dimension, so that names are
        # checked now rather than after the runs.
        self.ndim = len(draw(self.initial, replicate_generator(self.seed, MAIN, 0)))
        self.names = None if names is None else list(names)
        if self.names is not None and len(self.names) != self.ndim:
            raise ValueError(f"names must have {self.ndim} entries, not {len(self.names)}")
        self.tuning: TuningReport | None = None
        self.pilot: FloatArray | None = None
        self.settings: dict[str, Any] | None = None
        self._replicates: engine.Replicates | None = None

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
        :attr:`pilot`.

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
        if n < 1:
            raise ValueError(f"n must be at least 1, not {n}")
        self.tune()
        start = 0 if self.pilot is None else len(self.pilot)
        settings = {"k": 0, "m": 0, "lag": 1, "thin": 1, "max_iter": max_iter}
        times = self._run(PILOT, range(start, start + n), settings, store=False).meeting_time
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
    ) -> Result:
        """Run ``n_replicates`` more replicates and return the result of all of them so far.

        The first call tunes the kernel and runs the pilot when needed, and fixes the
        settings of the later calls (spec R-TUNE-2). The result warns about the
        conditions of spec section 7.4.

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

        Returns
        -------
        Result
            Unbiased for an ``n_replicates`` fixed in advance; extending a run after
            looking at it keeps consistency, not exact unbiasedness (spec section 3.3).
        """
        if n_replicates < 1:
            raise ValueError(f"n_replicates must be at least 1, not {n_replicates}")
        settings = self._configure(k, m, lag, thin, max_iter)
        start = 0 if self._replicates is None else len(self._replicates.index)
        self._check_memory(settings, start + n_replicates)
        part = self._run(MAIN, range(start, start + n_replicates), settings)
        self.settings = settings
        self._replicates = (
            part if self._replicates is None else engine.merge([self._replicates, part])
        )
        result = Result(
            self._replicates,
            names=self.names,
            kernel=self.kernel,
            initial=self.initial,
            seed=self.seed,
            tuning=self.tuning,
            pilot=self.pilot,
            **settings,
        )
        for issue in diagnostics.issues(result):
            if issue.kind not in _LIVE:
                warn(issue.message, ConvergenceWarning)
        return result

    def _run(
        self, namespace: int, indices: range, settings: dict[str, Any], store: bool = True
    ) -> engine.Replicates:
        # The engine on new replicates, warning live about slow pairs and zero densities.
        replicates = engine.run(
            self.target,
            self.kernel,
            self.initial,
            indices,
            root=self.seed,
            namespace=namespace,
            store=store,
            monitor=_slow_pairs(settings["lag"], settings["m"]),
            **settings,
        )
        if len(replicates.zero_density):
            warn(
                f"{len(replicates.zero_density)} replicates start where the target density "
                f"is zero (the first is {replicates.zero_density[0]}); the initial "
                "distribution should lie inside the support of the target",
                InputWarning,
            )
        return replicates

    def _configure(
        self,
        k: int | str,
        m: int | str,
        lag: int | str,
        thin: int | None,
        max_iter: int | None,
    ) -> dict[str, Any]:
        # The settings of every run of this sampler: fixed by the first (D-21, DECISIONS #4).
        if self.settings is not None:
            given = {"k": k, "m": m, "lag": lag, "thin": thin, "max_iter": max_iter}
            for name, value in given.items():
                if value not in ("auto", None) and value != self.settings[name]:
                    raise ValueError(
                        f"{name} was {self.settings[name]} in the earlier runs; a sampler "
                        f"keeps its settings, so start a new one for {name}={value}"
                    )
            return self.settings
        for name, value in (("k", k), ("m", m), ("lag", lag)):
            if value != "auto" and not isinstance(value, int | np.integer):
                raise ValueError(f"{name} must be an integer or 'auto', not {value!r}")
        self.tune()
        if k == "auto":
            k = choose_k(self.pilot if self.pilot is not None else self.meeting_times())
        settings: dict[str, Any] = {
            "k": int(k),
            "m": 10 * int(k) if m == "auto" else int(m),
            "lag": max(1, int(k)) if lag == "auto" else int(lag),
            "thin": 1 if thin is None else thin,
            "max_iter": max_iter,
        }
        engine.check_settings(**settings)
        return settings

    def _check_memory(self, settings: dict[str, Any], n_replicates: int) -> None:
        # Spec R-EXEC-7: estimate the atoms to store before running, and warn above 1 GB.
        # The pilot's meeting times stand in for those of the run.
        d = self.ndim
        n_averaged = len(averaging_times(settings["k"], settings["m"], settings["thin"]))
        late = 0.0
        if self.pilot is not None and np.isfinite(self.pilot).any():
            met = self.pilot[np.isfinite(self.pilot)]
            late = float(np.mean(np.maximum(0, met - 1 - settings["k"])))
        size = n_replicates * (n_averaged + 2 * late) * (8 * d + 16)
        if size > 1e9:
            warn(
                f"the run will store about {size / 1e9:.1f} GB of atoms; pass thin= to store "
                "fewer, or run fewer replicates",
                InputWarning,
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
    thin: int | None = None,
    max_iter: int | None = None,
    names: Sequence[str] | None = None,
    rng: Seed = None,
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
    # A monitor that warns once when a pair passes the slow-pair threshold (spec section
    # 6.3); it never stops the run.
    threshold = diagnostics.slow_pair_threshold(lag, m)
    warned = False

    def monitor(status: engine.Status) -> bool:
        nonlocal warned
        if not warned and status.t >= threshold:
            warned = True
            warn(
                f"replicate {status.slowest} has not met after {threshold} iterations; the "
                "meeting times may be heavy-tailed",
                ConvergenceWarning,
            )
        return False

    return monitor


def _kernel(kernel: Kernel | str) -> Kernel:
    if isinstance(kernel, Kernel):
        return kernel
    if isinstance(kernel, str) and kernel in KERNELS:
        return KERNELS[kernel]()
    raise ValueError(
        f"kernel must be a Kernel or one of {', '.join(map(repr, KERNELS))}, not {kernel!r}"
    )
