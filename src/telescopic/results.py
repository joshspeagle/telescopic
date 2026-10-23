"""Results of a run: estimates, moments, quantiles, diagnostics and the summary."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from telescopic import diagnostics, engine
from telescopic._types import FloatArray
from telescopic.engine import Replicates
from telescopic.estimators import unbiased_covariance, unbiased_variance, weighted_quantile
from telescopic.measure import SignedMeasure
from telescopic.tuning import TuningReport
from telescopic.warnings import ConvergenceWarning, warn


@dataclass(frozen=True, eq=False)
class Estimate:
    r"""An unbiased estimate of :math:`\pi(h)` with its Monte Carlo error (spec R-OUT-2).

    Attributes
    ----------
    value : ndarray, shape () or (p,)
        The mean :math:`\bar H` of the replicates' estimates.
    standard_error : ndarray, shape () or (p,)
        Their sample standard deviation divided by :math:`\sqrt R`.
    interval : tuple of two ndarrays
        :math:`\bar H \mp z\,\hat\sigma/\sqrt R`, with :math:`z` the standard normal
        quantile of ``(1 + level) / 2``; valid as :math:`R` grows when the variance is
        finite.
    level : float
    replicates : ndarray, shape (R,) or (R, p)
        The replicates' estimates :math:`\hat\pi_r(h)`.
    """

    value: FloatArray
    standard_error: FloatArray
    interval: tuple[FloatArray, FloatArray]
    level: float
    replicates: FloatArray

    def __repr__(self) -> str:
        pairs = [
            f"{value:.6g} ± {error:.2g}"
            for value, error in zip(
                np.ravel(self.value), np.ravel(self.standard_error), strict=True
            )
        ]
        body = pairs[0] if self.value.ndim == 0 else "[" + ", ".join(pairs) + "]"
        return f"Estimate({body}, R={len(self.replicates)})"


class Result:
    r"""The output of a run of coupled chains (spec R-OUT-1).

    Censored pairs appear in :attr:`meeting_times` as ``inf`` and are excluded from the
    estimates, from :attr:`n_replicates` and from :attr:`measure`.

    Parameters
    ----------
    replicates : telescopic.engine.Replicates
        The engine's output, with the settings and the root seed of its run.
    names : sequence of str, optional
        Parameter names; ``x0``, ``x1``, ... by default.
    kernel, initial : optional
        The frozen kernel and the initial distribution, for the record.
    tuning : TuningReport, optional
        The report of the tuning pilot.
    pilot : ndarray, shape (R0,), optional
        The pilot's lag-1 meeting times, from which ``k`` was chosen.

    Attributes
    ----------
    k, m, lag, thin : int
        The settings of the run (spec section 3.2).
    max_iter : int or None
        The cap of the run.
    seed : numpy.random.SeedSequence
        The root seed, whose ``entropy`` and ``spawn_key`` reproduce the run.
    index : ndarray of int, shape (R_total,)
        Replicate indices, censored ones included.
    meeting_times : ndarray of float, shape (R_total,)
    costs : ndarray of int, shape (R_total,)
        Kernel applications per replicate.
    measure : SignedMeasure
    n_replicates : int
        :math:`R`, the replicates that met.
    n_censored : int
    ndim : int
    n_log_prob, n_grad : int
        Evaluations of the log-density and its gradient during the runs.
    wall_time : float
        Seconds spent in the runs.
    interrupted : bool
        Whether a requested replicate never ran, so that the estimates are consistent
        but not exactly unbiased until it does (spec section 6.5).
    """

    def __init__(
        self,
        replicates: Replicates,
        *,
        names: Sequence[str] | None = None,
        kernel: Any = None,
        initial: Any = None,
        tuning: TuningReport | None = None,
        pilot: FloatArray | None = None,
    ) -> None:
        met = np.isfinite(replicates.meeting_time)
        self.n_replicates = int(met.sum())
        self.n_censored = len(met) - self.n_replicates
        if self.n_replicates and not len(replicates.atoms):
            raise ValueError(
                "the replicates hold no signed measures: run the engine with store=True"
            )
        self.ndim = replicates.atoms.shape[1]
        self.names = [f"x{i}" for i in range(self.ndim)] if names is None else list(names)
        if len(self.names) != self.ndim:
            raise ValueError(f"names must have {self.ndim} entries, not {len(self.names)}")
        self.index = replicates.index
        self.meeting_times = replicates.meeting_time
        self.costs = replicates.cost
        self.measure = SignedMeasure(
            replicates.atoms,
            replicates.weights,
            np.searchsorted(replicates.index[met], replicates.owner),
            self.n_replicates,
        )
        self.k, self.m, self.lag, self.thin, self.max_iter = replicates.settings
        self.seed, self.kernel, self.initial = replicates.seed, kernel, initial
        self._replicates = replicates
        self.n_log_prob, self.n_grad = replicates.n_log_prob, replicates.n_grad
        self.wall_time = replicates.seconds
        self.tuning, self.pilot = tuning, pilot
        # The engine runs only requested indices, so missing ones make index shorter.
        self.interrupted = len(replicates.index) < len(replicates.requested)
        # The cap for a censored pair makes the bounds of section 3.6 lower bounds of
        # their estimates.
        self._stopped = replicates.stopped.astype(np.int64)

    def estimate(self, h: Callable[[FloatArray], npt.ArrayLike], level: float = 0.95) -> Estimate:
        r"""Estimate :math:`\pi(h)` with a confidence interval (spec section 3.3).

        Parameters
        ----------
        h : callable
            Maps points of shape ``(n, d)`` to values of shape ``(n,)`` or ``(n, p)``.
        level : float
            Confidence level of the interval, in (0, 1).

        Returns
        -------
        Estimate
        """
        if not 0 < level < 1:
            raise ValueError(f"level must be in (0, 1), not {level}")
        self._require(2)
        values = self.measure.per_replicate(h)
        value, error = values.mean(axis=0), np.asarray(stats.sem(values, axis=0))
        z = stats.norm.ppf((1 + level) / 2)
        return Estimate(value, error, (value - z * error, value + z * error), level, values)

    def mean(self) -> Estimate:
        """The estimate of the mean of each coordinate, with shapes ``(d,)``."""
        return self.estimate(lambda x: x)

    def cov(self) -> FloatArray:
        """Unbiased estimate of the covariance matrix of the target (spec section 3.4).

        Returns
        -------
        ndarray, shape (d, d)
        """
        self._require(2)
        first = self.measure.per_replicate(lambda x: x)
        atoms, weights = self.measure.atoms, self.measure.weights
        second = atoms.T @ (weights[:, None] * atoms) / self.n_replicates
        return unbiased_covariance(first, second)

    def var(self) -> FloatArray:
        """Unbiased estimates of the variances, shape ``(d,)``; they can be negative."""
        self._require(2)
        first = self.measure.per_replicate(lambda x: x)
        atoms, weights = self.measure.atoms, self.measure.weights
        second = np.einsum("i,ij,ij->j", weights, atoms, atoms) / self.n_replicates
        return unbiased_variance(first, second)

    def std(self) -> FloatArray:
        r"""Standard deviations :math:`\sqrt{\max(0, \widehat{\mathrm{Var}})}`, shape ``(d,)``.

        Consistent as :math:`R` grows, not unbiased.
        """
        deviations: FloatArray = np.sqrt(np.maximum(0.0, self.var()))
        return deviations

    def quantile(self, q: npt.ArrayLike) -> FloatArray:
        """Quantiles of each coordinate under the pooled signed measure (spec section 3.4).

        Consistent as :math:`R` grows, not unbiased.

        Parameters
        ----------
        q : float or array_like of float in [0, 1]

        Returns
        -------
        ndarray, shape (d,) for a scalar ``q``, else ``(len(q), d)``
        """
        q = np.asarray(q, dtype=np.float64)
        if np.any((q < 0) | (q > 1)):
            raise ValueError(f"q must be in [0, 1], not {q}")
        self._require(1)
        # Rejections repeat atoms; merging consecutive equal rows first shortens every sort.
        atoms = self.measure.atoms
        first = np.concatenate([[True], np.any(atoms[1:] != atoms[:-1], axis=1)]).nonzero()[0]
        weights = np.add.reduceat(self.measure.weights, first) / self.n_replicates
        columns = [weighted_quantile(column, weights, q) for column in atoms[first].T]
        quantiles: FloatArray = np.stack(columns, axis=-1)
        return quantiles

    def interval(self, mass: float = 0.68) -> FloatArray:
        """The central interval holding ``mass`` of each marginal of the target.

        Parameters
        ----------
        mass : float in (0, 1)

        Returns
        -------
        ndarray, shape (2, d)
            Lower and upper ends.
        """
        if not 0 < mass < 1:
            raise ValueError(f"mass must be in (0, 1), not {mass}")
        return self.quantile([(1 - mass) / 2, (1 + mass) / 2])

    def histogram(
        self, dims: int | Sequence[int], bins: int = 20, range: npt.ArrayLike | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray | list[FloatArray]]:
        """Unbiased histogram of one coordinate or several, with standard errors (spec section 3.4).

        The mass of each half-open bin is the average over replicates of its signed mass,
        so a bin can be negative.

        Parameters
        ----------
        dims : int or sequence of int
            The coordinate, or the coordinates of a joint histogram.
        bins : int
            The number of bins along each coordinate.
        range : array_like, shape (2,) or (len(dims), 2), optional
            The lower and upper edges, a pair per coordinate; by default the range of the
            atoms.

        Returns
        -------
        masses, standard_errors : ndarray, shape (bins,) or (bins,) * len(dims)
            Entry ``[i, j]`` of a joint histogram is bin ``i`` of the first coordinate and
            bin ``j`` of the second.
        edges : ndarray, shape (bins + 1,), or a list of them for a sequence of coordinates
        """
        columns = self._columns(dims)
        if bins < 1:
            raise ValueError(f"bins must be at least 1, not {bins}")
        expected = (*np.shape(dims), 2)
        if range is not None and np.shape(range) != expected:
            raise ValueError(f"range must have shape {expected}, not {np.shape(range)}")
        self._require(2)
        limits = [None] * len(columns) if range is None else map(tuple, np.reshape(range, (-1, 2)))
        cells, edges = [], []
        for column, limit in zip(columns, limits, strict=True):
            which, edge = self._bins(column, bins, limit)
            cells.append(which)
            edges.append(edge)
        shape = (bins,) * len(columns)
        which = np.stack(cells)
        inside = np.all((which >= 0) & (which < bins), axis=0)
        cell = np.where(inside, np.ravel_multi_index(which, shape, mode="clip"), -1)
        masses = self.measure.mass(cell, bins ** len(columns))
        mean, error = masses.mean(axis=0), np.asarray(stats.sem(masses, axis=0))
        return mean.reshape(shape), error.reshape(shape), edges if np.ndim(dims) else edges[0]

    def tv_bound(self, t: npt.ArrayLike | None = None) -> FloatArray:
        """Estimated upper bound on the total variation distance from the law of X_t to the target.

        Spec section 3.6; with censored pairs it is a lower bound of that estimate.

        Parameters
        ----------
        t : array_like of int, optional
            Times; by default every time from 0 to the last at which the bound is
            positive, so that entry ``t`` is the bound at ``t``.

        Returns
        -------
        ndarray, shape of ``t``
        """
        return diagnostics.tv_bound(self._stopped, self.lag, self._times() if t is None else t)

    def w1_bound(self, t: npt.ArrayLike | None = None) -> FloatArray:
        """Estimated upper bound on the 1-Wasserstein distance from the law of X_t to the target.

        Spec section 3.6. It depends on the parametrisation of the target.

        Parameters
        ----------
        t : array_like of int, optional
            Times; by default every time from 0 to the last at which the bound is positive.

        Returns
        -------
        ndarray, shape of ``t``
        """
        return diagnostics.w1_bound(
            self._replicates.distance_time,
            self._replicates.distance,
            len(self.index),
            self.lag,
            self._times() if t is None else t,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Diagnostics of the run: meeting times, censoring, bias corrections, cost.

        Returns
        -------
        dict
            ``meeting_time`` quantiles (50, 90, 99) and maximum, ``censored``,
            ``late_fraction``, the fraction of pairs with :math:`\\tau - L > k`, censored
            ones included (the leading term of the TV bound at :math:`k`, spec section
            3.6), with ``late_fraction_upper``, its one-sided 95% Clopper-Pearson upper
            bound, which stays informative when the fraction is 0; ``tv_bound_at_k``;
            ``mean_cost``, ``seconds_per_replicate`` and ``inefficiency`` (mean cost over
            :math:`m - k + 1`; spec R-TUNE-4).
        """
        met = self.meeting_times[np.isfinite(self.meeting_times)]
        quantiles = np.quantile(met, [0.5, 0.9, 0.99]) if len(met) else np.full(3, np.nan)
        n_late = int(np.sum(self.meeting_times - self.lag > self.k))
        n_total = len(self.meeting_times)
        upper = stats.binomtest(n_late, n_total, alternative="less").proportion_ci(0.95).high
        mean_cost = float(self.costs.mean())
        return {
            "meeting_time_50": float(quantiles[0]),
            "meeting_time_90": float(quantiles[1]),
            "meeting_time_99": float(quantiles[2]),
            "meeting_time_max": float(np.max(met)) if len(met) else np.nan,
            "censored": self.n_censored,
            "late_fraction": n_late / n_total,
            "late_fraction_upper": float(upper),
            "tv_bound_at_k": float(self.tv_bound([self.k])[0]),
            "mean_cost": mean_cost,
            "seconds_per_replicate": self.wall_time / len(self.index),
            "inefficiency": mean_cost / (self.m - self.k + 1),
        }

    def summary(self, level: float = 0.95) -> Summary:
        """Estimates per parameter and the diagnostics of the run (spec R-OUT-3).

        Parameters
        ----------
        level : float
            Confidence level of the reported intervals of the means.

        Returns
        -------
        Summary
        """
        mean = self.estimate(lambda x: x, level)
        quantiles = self.quantile([0.16, 0.5, 0.84])
        table = {
            "mean": mean.value,
            "standard_error": mean.standard_error,
            "std": self.std(),
            "q16": quantiles[0],
            "q50": quantiles[1],
            "q84": quantiles[2],
        }
        footer: dict[str, Any] = {
            "n_replicates": self.n_replicates,
            "k": self.k,
            "m": self.m,
            "lag": self.lag,
            "thin": self.thin,
            "kernel": str(self.kernel),
            **self.diagnostics(),
            "interrupted": self.interrupted,
            "warnings": [issue.message for issue in diagnostics.issues(self)],
        }
        return Summary(self.names, table, footer)

    def merge(self, other: Result) -> Result:
        """Join with a run of the same seed and configuration and disjoint replicates.

        Spec R-OUT-1; :func:`telescopic.engine.merge` checks the runs.

        Parameters
        ----------
        other : Result

        Returns
        -------
        Result
            The union of the replicates and of the requested indices; interrupted if a
            requested replicate ran in neither.
        """
        return type(self)(
            engine.merge([self._replicates, other._replicates]),
            names=self.names,
            kernel=self.kernel,
            initial=self.initial,
            tuning=self.tuning,
            pilot=self.pilot,
        )

    def save(self, path: str | os.PathLike[str] | IO[bytes]) -> None:
        """Save as a versioned ``.npz`` file, without pickle (spec R-OUT-1).

        The kernel and the initial distribution are saved as their descriptions,
        ``str(kernel)`` and ``str(initial)``; the engine's paths are not saved.

        Parameters
        ----------
        path : str, path-like or binary file
            ``.npz`` is appended to a name that lacks it.
        """
        # Imported here: telescopic._io imports this module.
        from telescopic import _io

        _io.save(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | IO[bytes]) -> Result:
        """Load a result saved by :meth:`save`, with descriptions of its kernel and initial distribution.

        Parameters
        ----------
        path : str, path-like or binary file
            ``.npz`` is appended to a name that lacks it.

        Returns
        -------
        Result
        """
        from telescopic import _io

        return _io.load(path, cls)

    def plot_meeting_times(self, axes: Sequence[Any] | None = None) -> Any:
        """Plot the meeting times and their survival function (needs matplotlib).

        Parameters
        ----------
        axes : sequence of two matplotlib Axes, optional
            By default, those of a new figure.

        Returns
        -------
        list of two matplotlib Axes
        """
        # Imported here: matplotlib is optional, and telescopic.plotting imports this module.
        from telescopic import plotting

        return plotting.meeting_times(self, axes)

    def plot_tv_bound(self, ax: Any = None) -> Any:
        """Plot the TV bound against t, with k marked (needs matplotlib).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            By default, that of a new figure.

        Returns
        -------
        matplotlib Axes
        """
        from telescopic import plotting

        return plotting.tv_bound(self, ax)

    def plot_marginals(
        self,
        dims: int | Sequence[int] | None = None,
        bins: int = 20,
        axes: Sequence[Any] | None = None,
    ) -> Any:
        """Plot signed histograms of the marginals with their errors (needs matplotlib).

        Parameters
        ----------
        dims : int or sequence of int, optional
            Coordinates; all by default.
        bins : int
        axes : sequence of matplotlib Axes, optional
            One per coordinate; by default, those of a new figure.

        Returns
        -------
        list of matplotlib Axes
        """
        from telescopic import plotting

        return plotting.marginals(self, dims, bins, axes)

    def plot_corner(self, dims: int | Sequence[int] | None = None, bins: int = 20) -> Any:
        """Plot a corner plot of signed histograms (needs matplotlib).

        Parameters
        ----------
        dims : int or sequence of int, optional
            Coordinates; all by default.
        bins : int

        Returns
        -------
        ndarray of matplotlib Axes
        """
        from telescopic import plotting

        return plotting.corner(self, dims, bins)

    def check(self, strict: bool = False) -> None:
        """Warn about the conditions of spec section 7.4 that apply to this run.

        Parameters
        ----------
        strict : bool
            Raise ``RuntimeError`` instead of warning.
        """
        messages = [issue.message for issue in diagnostics.issues(self)]
        if strict and messages:
            raise RuntimeError("; ".join(messages))
        for message in messages:
            warn(message, ConvergenceWarning)

    def _columns(self, dims: int | Sequence[int]) -> npt.NDArray[np.intp]:
        # Coordinates given as one integer or a sequence of them, checked.
        columns = np.atleast_1d(dims)
        if (
            columns.ndim != 1
            or columns.dtype.kind not in "iu"
            or np.any((columns < 0) | (columns >= self.ndim))
        ):
            raise ValueError(
                f"dims must be an integer in [0, {self.ndim}) or a sequence of them, not {dims}"
            )
        return columns.astype(np.intp)

    def _bins(
        self, column: int, bins: int, limit: tuple[float, float] | None = None
    ) -> tuple[npt.NDArray[np.intp], FloatArray]:
        # The bin of each atom along a coordinate, below 0 or from bins outside the edges,
        # and the edges; by default the last is nudged up to keep the largest atom inside.
        values = self.measure.atoms[:, column]
        edges = np.histogram_bin_edges(values, bins, limit)
        if limit is None:
            edges[-1] = np.nextafter(edges[-1], np.inf)
        return np.searchsorted(edges, values, side="right") - 1, edges

    def _require(self, n: int) -> None:
        if self.n_replicates < n:
            raise ValueError(
                f"this needs at least {n} replicates that met, not {self.n_replicates}"
            )

    def _times(self) -> npt.NDArray[np.int64]:
        # Every time from 0 to the last at which the bounds of section 3.6 are positive.
        return np.arange(max(1, int(np.max(self._stopped)) - self.lag))


@dataclass(eq=False)
class Summary:
    """Estimates per parameter and the diagnostics of a run, printable as a table.

    Attributes
    ----------
    names : list of str
    table : dict of str to ndarray
        The columns ``mean``, ``standard_error``, ``std``, ``q16``, ``q50`` and ``q84``,
        each with one entry per parameter.
    footer : dict
        The configuration and diagnostics of the run, and the warnings that apply.
    """

    names: list[str]
    table: dict[str, FloatArray]
    footer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """The summary as nested dictionaries of Python numbers."""
        parameters = {
            name: {column: float(values[i]) for column, values in self.table.items()}
            for i, name in enumerate(self.names)
        }
        return {"parameters": parameters, **self.footer}

    def to_pandas(self) -> Any:
        """The table as a ``pandas.DataFrame`` indexed by parameter name (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(self.table, index=self.names)

    def __repr__(self) -> str:
        width = max(len(name) for name in self.names)
        header = f"{'':{width}}  " + "  ".join(f"{column:>14}" for column in self.table)
        rows = [
            f"{name:{width}}  "
            + "  ".join(f"{float(values[i]):>14.6g}" for values in self.table.values())
            for i, name in enumerate(self.names)
        ]
        footer = self.footer
        lines = [
            (
                f"R = {footer['n_replicates']}, k = {footer['k']}, m = {footer['m']}, "
                f"lag = {footer['lag']}, thin = {footer['thin']}; {footer['kernel']}"
            ),
            (
                f"meeting times: median {footer['meeting_time_50']:.4g}, 90% "
                f"{footer['meeting_time_90']:.4g}, 99% {footer['meeting_time_99']:.4g}, max "
                f"{footer['meeting_time_max']:.4g}; censored {footer['censored']}"
            ),
            (
                f"cost per replicate: {footer['mean_cost']:.4g} kernel applications, "
                f"{footer['seconds_per_replicate']:.3g} s; inefficiency "
                f"{footer['inefficiency']:.3g}; pairs meeting after k + lag "
                f"{footer['late_fraction']:.1%} (at most {footer['late_fraction_upper']:.1%} "
                "at 95%)"
            ),
        ]
        if footer["interrupted"]:
            lines.append(
                "interrupted: the estimates are consistent but not exactly unbiased; run "
                "the remaining replicates to restore unbiasedness"
            )
        lines += [f"warning: {message}" for message in footer["warnings"]]
        return "\n".join([header, *rows, "", *lines])
