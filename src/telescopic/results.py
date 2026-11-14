"""Results of a run: estimates, moments, quantiles, diagnostics and the summary."""

from __future__ import annotations

import functools
import html
import os
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
import numpy.typing as npt

from telescopic import diagnostics, engine
from telescopic._types import (
    FloatArray,
    coordinates,
    flag,
    integer,
    parameter_names,
    real,
    values_of,
)
from telescopic.engine import Replicates
from telescopic.estimators import unbiased_covariance, unbiased_variance, weighted_quantile
from telescopic.measure import SignedMeasure
from telescopic.tuning import TuningReport
from telescopic.warnings import ConvergenceWarning, warn

__all__ = ["Estimate", "Result", "Summary"]

# The summary's footer wraps at the width of a terminal, which a code output of the
# documentation also holds at 1440 px (83 characters).
_WRAP = textwrap.TextWrapper(
    80, subsequent_indent="    ", break_long_words=False, break_on_hyphens=False
)
# A float in a kernel's repr: digits with a point or an exponent, not inside a name or a
# dotted token such as a version, 3.10.12.
_FLOAT = re.compile(r"(?<![\w.])\d+(?:\.\d*(?:[eE][-+]?\d+)?|[eE][-+]?\d+)(?![\w.])")


def _four_figures(match: re.Match[str]) -> str:
    # 4 significant figures, still written as a float (2.0 rather than 2).
    text = f"{float(match[0]):.4g}"
    return text if any(c in text for c in ".en") else text + ".0"


# Spec R-OUT-2.
@dataclass(frozen=True, eq=False)
class Estimate:
    r"""An unbiased estimate of :math:`\pi(h)` with its Monte Carlo error.

    Attributes
    ----------
    value : ndarray, shape () or (p,)
        The mean :math:`\bar H` of the replicates' estimates.
    standard_error : ndarray, shape () or (p,)
        Their sample standard deviation divided by :math:`\sqrt R`.
    interval : ndarray, shape (2,) or (2, p)
        The confidence interval :math:`\bar H \mp t_{R-1}\,\hat\sigma/\sqrt R`, lower end
        first, with :math:`t_{R-1}` the quantile of ``(1 + level) / 2`` of Student's t
        with :math:`R - 1` degrees of freedom; valid as :math:`R` grows when the variance
        is finite, and exact at every :math:`R \ge 2` for normal replicates.
    level : float
        The confidence level of ``interval``, such as 0.95.
    replicates : ndarray, shape (R,) or (R, p)
        The replicates' estimates :math:`\hat\pi_r(h)`.
    """

    value: FloatArray
    standard_error: FloatArray
    interval: FloatArray
    level: float
    replicates: FloatArray

    def __repr__(self) -> str:
        pairs = [
            f"{value:.6g} ± {error:.2g}"
            for value, error in zip(
                np.ravel(self.value), np.ravel(self.standard_error), strict=True
            )
        ]
        body = pairs[0] if self.value.ndim == 0 else f"[{', '.join(pairs)}]"
        return f"Estimate({body}, R={len(self.replicates)})"


# Spec R-OUT-1; settings, section 3.2; n_grad_zeroed, R-INPUT-1; interrupted,
# section 6.5.
class Result:
    r"""The output of a run of coupled chains.

    Censored pairs appear in ``meeting_times`` as ``inf`` and are excluded from the
    estimates, from ``n_replicates`` and from ``measure``.

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
        The sampler's lag-1 pilot meeting times; an automatic ``k`` came from those run
        before the first run.

    Attributes
    ----------
    k, m, lag, thin : int
        The settings of the run.
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
        Evaluations of the log-density and its gradient by the replicates, without those
        of tuning and of the meeting-time pilot.
    n_grad_zeroed : int
        Gradients of the replicates taken as 0 because they were not finite where the
        log-density is.
    wall_time : float
        Seconds spent running the replicates, without tuning and the pilot.
    interrupted : bool
        Whether a requested replicate never ran, so that the estimates are consistent
        but not exactly unbiased until it does.
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
        if not len(replicates.index):
            raise ValueError("replicates must hold at least one replicate, not none")
        met = np.isfinite(replicates.meeting_time)
        self.n_replicates = int(met.sum())
        self.n_censored = len(met) - self.n_replicates
        if not replicates.settings.store:
            raise ValueError(
                "the replicates hold no signed measures: run the engine with store=True"
            )
        if self.n_censored and replicates.settings.max_iter is None:
            raise ValueError("replicates with censored pairs need the max_iter of their run")
        self.ndim = replicates.atoms.shape[1]
        self.names = parameter_names(names, self.ndim, "the atoms are points")
        self.index = replicates.index
        self.meeting_times = replicates.meeting_time
        self.costs = replicates.cost
        self.measure = SignedMeasure(
            replicates.atoms,
            replicates.weights,
            np.searchsorted(replicates.index[met], replicates.owner),
            self.n_replicates,
        )
        self.k, self.m, self.lag, self.thin, self.max_iter, _ = replicates.settings
        self.seed, self.kernel, self.initial = replicates.seed, kernel, initial
        self._replicates = replicates
        self.n_log_prob, self.n_grad = replicates.n_log_prob, replicates.n_grad
        self.n_grad_zeroed = replicates.n_grad_zeroed
        self.wall_time = replicates.seconds
        self.tuning, self.pilot = tuning, pilot
        # The engine runs only requested indices, so missing ones make index shorter.
        self.interrupted = len(replicates.index) < len(replicates.requested)
        # The cap for a censored pair makes the bounds of section 3.6 lower bounds of
        # their estimates. The engine compared its chains at the cap before censoring
        # it, so the TV bound counts it as meeting at cap + 1, the least it can.
        self._stopped = replicates.stopped.astype(np.int64)
        self._earliest_meeting = self._stopped + ~met

    def __repr__(self) -> str:
        # Stored fields only: an echo in a notebook must not compute the diagnostics.
        return (
            f"Result(R={self.n_replicates}, d={self.ndim}, k={self.k}, m={self.m}, "
            f"lag={self.lag}, censored={self.n_censored}, interrupted={self.interrupted})"
        )

    # Spec section 3.3.
    def estimate(self, h: Callable[[FloatArray], npt.ArrayLike], level: float = 0.95) -> Estimate:
        r"""Estimate :math:`\pi(h)` with a confidence interval.

        Parameters
        ----------
        h : callable
            Maps points of shape ``(n, d)``, a read-only array, to values of shape ``(n,)``
            or ``(n, p)``.
        level : float
            Confidence level of the interval, in (0, 1).

        Returns
        -------
        Estimate

        Examples
        --------
        The probability that the first coordinate is positive is the expectation of an
        indicator, one value per point; ``replicates`` holds each replicate's estimate:

        >>> from telescopic import examples, sample
        >>> example = examples.gaussian(2)
        >>> result = sample(
        ...     example.log_prob, example.initial,
        ...     n_replicates=50, vectorize=True, rng=0,
        ... )
        >>> estimate = result.estimate(lambda x: x[:, 0] > 0)
        >>> estimate.value.shape, estimate.replicates.shape
        ((), (50,))
        """
        level = real("level", level, 0, 1)
        self._require(2)
        values = self.measure.per_replicate(h)
        # Imported at first use, which keeps import telescopic, and so a pool worker's
        # start, free of scipy.stats.
        from scipy import stats

        value, error = values.mean(axis=0), np.asarray(stats.sem(values, axis=0))
        # Student's t, not the normal quantile, whose intervals undercover at small R.
        t = stats.t.ppf((1 + level) / 2, self.n_replicates - 1)
        interval = np.stack([value - t * error, value + t * error])
        return Estimate(value, error, interval, level, values)

    def mean(self) -> Estimate:
        """The estimate of the mean of each coordinate.

        Returns
        -------
        Estimate
            With ``value`` and ``standard_error`` of shape ``(d,)``.
        """
        return self.estimate(lambda x: x)

    def cov(self) -> FloatArray:
        r"""Unbiased estimate of the covariance matrix of the target.

        .. math:: \widehat{\mathrm{Cov}}_\pi(x) = R^{-1}\sum_r \hat\pi_r(xx^\top)
                  - \bar a\bar a^\top + S_a/R

        with :math:`a_r = \hat\pi_r(x)`, their mean :math:`\bar a` and sample covariance
        :math:`S_a` over the :math:`R` replicates that met.

        Returns
        -------
        ndarray, shape (d, d)
        """
        self._require(2)
        centered, first = self._centered()
        weights = self.measure.weights
        second = centered.T @ (weights[:, None] * centered) / self.n_replicates
        return unbiased_covariance(first, second)

    def var(self) -> FloatArray:
        r"""Unbiased estimates of the variances; they can be negative.

        .. math:: \widehat{\mathrm{Var}}_\pi(x_j) = R^{-1}\sum_r \hat\pi_r(x_j^2)
                  - \bar a_j^2 + S_{a,jj}/R

        This is the diagonal of :meth:`cov`, whose docstring defines the notation.

        Returns
        -------
        ndarray, shape (d,)
        """
        self._require(2)
        centered, first = self._centered()
        weights = self.measure.weights
        second = np.einsum("i,ij,ij->j", weights, centered, centered) / self.n_replicates
        return unbiased_variance(first, second)

    def std(self) -> FloatArray:
        r"""Standard deviations :math:`\sqrt{\max(0, \widehat{\mathrm{Var}})}`.

        Consistent as :math:`R` grows, not unbiased.

        Returns
        -------
        ndarray, shape (d,)
        """
        deviations: FloatArray = np.sqrt(np.maximum(0.0, self.var()))
        return deviations

    # Spec section 3.4.
    def quantile(
        self, q: npt.ArrayLike, h: Callable[[FloatArray], npt.ArrayLike] | None = None
    ) -> FloatArray:
        """Quantiles of each coordinate, or of ``h``, under the pooled signed measure.

        Spec section 3.4. Consistent as :math:`R` grows, not unbiased.

        Parameters
        ----------
        q : float or array_like of float in [0, 1]
            The levels of the quantiles.
        h : callable, optional
            As in :meth:`estimate`; its quantiles are taken in place of the coordinates'.

        Returns
        -------
        ndarray, shape ``q.shape + (d,)``, or with ``h``, ``q.shape`` or ``q.shape + (p,)``

        Notes
        -----
        Cost: one sort of the atoms per column, after merging consecutive equal atoms, and
        with ``h``, one call of ``h`` on the merged atoms.
        """
        self._require(1)
        # Rejections repeat atoms; merging consecutive equal rows first shortens every sort
        # and the calls of h.
        atoms = self.measure.atoms
        first = np.concatenate([[True], np.any(atoms[1:] != atoms[:-1], axis=1)]).nonzero()[0]
        weights = np.add.reduceat(self.measure.weights, first) / self.n_replicates
        values = atoms[first] if h is None else values_of(h, atoms[first])
        quantiles: FloatArray = np.apply_along_axis(weighted_quantile, 0, values, weights, q)
        return quantiles

    def credible_interval(
        self, mass: float = 0.68, h: Callable[[FloatArray], npt.ArrayLike] | None = None
    ) -> FloatArray:
        r"""The central credible interval holding ``mass`` of each marginal, or of ``h``.

        The quantiles at :math:`(1 - \mathrm{mass})/2` and :math:`(1 + \mathrm{mass})/2`.

        Parameters
        ----------
        mass : float in (0, 1)
            The probability that the interval holds.
        h : callable, optional
            As in :meth:`quantile`.

        Returns
        -------
        ndarray, shape (2, d), or with ``h``, (2,) or (2, p)
            Lower and upper ends.
        """
        mass = real("mass", mass, 0, 1)
        return self.quantile([(1 - mass) / 2, (1 + mass) / 2], h)

    # Spec section 3.4.
    def histogram(
        self, dims: int | Sequence[int], bins: int = 20, range: npt.ArrayLike | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray | list[FloatArray]]:
        """Histogram of one coordinate or several, with standard errors.

        The mass of each bin is the average over replicates of its signed mass, so a bin
        can be negative. The bins are half-open, :math:`[a, b)`, except the last, which is
        closed, as in ``numpy.histogram``, so that an atom on the upper edge of ``range``
        is counted. It is unbiased for a ``range`` fixed before the run; the default
        range, from the atoms, biases every bin, the two outer ones by a term of order
        :math:`1/R`. Memory: a few floats per atom, and per bin.

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
        columns = coordinates(dims, self.ndim)
        bins = integer("bins", bins, minimum=1)
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
        n_cells, n = bins ** len(columns), self.n_replicates
        masses = self.measure.mass(cell, n_cells)
        column, value = masses.indices, masses.data
        mean = np.bincount(column, value, n_cells) / n
        # The squared deviations of the stored masses, and of the other replicates' zeros.
        squares = np.bincount(column, (value - mean[column]) ** 2, n_cells)
        squares += (n - np.bincount(column, minlength=n_cells)) * mean**2
        error = np.sqrt(squares / (n - 1) / n)
        return mean.reshape(shape), error.reshape(shape), edges if np.ndim(dims) else edges[0]

    # Spec section 3.6.
    def tv_bound(self, t: npt.ArrayLike | None = None) -> FloatArray:
        """Estimated upper bound on the total variation distance from the law of X_t to the target.

        The bound derived under Distance to stationarity, in :doc:`/theory`. A censored pair
        counts as meeting just after its cap, the earliest it can, so with censored pairs
        it is a lower bound of the estimate without a cap.

        Parameters
        ----------
        t : array_like of int, optional
            Times; by default every time from 0 to the last at which the bound is
            positive, so that entry ``t`` is the bound at ``t``.

        Returns
        -------
        ndarray, shape of ``t``
        """
        if t is None:
            t = self._times(int(np.max(self._earliest_meeting)))
        return diagnostics.tv_bound(self._earliest_meeting, self.lag, t)

    # Spec section 3.6.
    def w1_bound(self, t: npt.ArrayLike | None = None) -> FloatArray:
        """Estimated upper bound on the 1-Wasserstein distance from the law of X_t to the target.

        The bound derived under Distance to stationarity, in :doc:`/theory`. It depends on
        the parameterization of the target. A censored
        pair's distances stop at its cap, so with censored pairs it is a lower bound of the
        estimate without a cap.

        Parameters
        ----------
        t : array_like of int, optional
            Times; by default every time from 0 to the last at which the bound is positive.

        Returns
        -------
        ndarray, shape of ``t``
        """
        replicates = self._replicates
        if t is None:
            t = self._times(int(np.max(replicates.distance_time, initial=0)) + 1)
        return diagnostics.w1_bound(
            replicates.distance_time, replicates.distance, len(self.index), self.lag, t
        )

    # Spec section 3.6 and R-TUNE-4.
    def diagnostics(self) -> dict[str, Any]:
        """Diagnostics of the run: meeting times, censoring, bias corrections, cost.

        Returns
        -------
        dict
            ``meeting_time`` quantiles (50, 90, 99) and maximum over every pair, ``inf``
            where one falls on a censored pair, which sorts above every meeting time;
            ``tail_index``,
            :func:`~telescopic.diagnostics.hill` of :math:`\\tau - L + 1` with
            ``n_tail`` :math:`= \\lfloor R/10 \\rfloor`, ``nan`` below 500 replicates;
            ``censored``; ``late_fraction``, the fraction of pairs with
            :math:`\\tau - L > k`, censored ones included (the leading term of the TV
            bound at :math:`k`), with ``late_fraction_upper``, its
            one-sided 95% Clopper-Pearson upper bound, which stays informative when the
            fraction is 0; ``tv_bound_at_k``; ``variance_share``, the largest share of one
            replicate in the sample variance of the estimate of a coordinate's mean or
            variance, ``variance_share_of``, that estimate, and
            ``variance_share_replicate``, the index of that replicate, ``nan``, ``None``
            and ``None`` below 2 replicates; ``mean_cost``, ``seconds_per_replicate`` and
            ``relative_cost``, the mean cost over :math:`m - k + 1`.
        """
        return dict(self._diagnostics)

    @functools.cached_property
    def _diagnostics(self) -> dict[str, Any]:
        # Computed once: each run's warnings, the summary and check all read it.
        # Over every pair, censored ones (inf) above the others, as in the pilot of k.
        quantiles = np.quantile(self.meeting_times, [0.5, 0.9, 0.99, 1], method="inverted_cdf")
        tail_index = np.nan
        if self.n_replicates >= 500:
            met = self.meeting_times[np.isfinite(self.meeting_times)]
            tail_index = diagnostics.hill(met - self.lag + 1, self.n_replicates // 10)
        n_late = int(np.sum(self.meeting_times - self.lag > self.k))
        n_total = len(self.meeting_times)
        from scipy import stats

        upper = stats.binomtest(n_late, n_total, alternative="less").proportion_ci(0.95).high
        # The TV bound at k as an average over replicates (section 3.6), in O(R) memory:
        # diagnostics.tv_bound's survival function takes memory up to the largest time.
        excess = np.ceil((self._earliest_meeting - self.lag - self.k) / self.lag)
        tv_at_k = min(1.0, float(np.maximum(0, excess).mean()))
        share, share_of, share_replicate = np.nan, None, None
        if self.n_replicates >= 2:
            # The second moments about the atoms' mean, which the variances use: about 0,
            # far from it, they would vary with the means and hide their own spread.
            centered, first = self._centered()
            # Squared in place: the centered atoms are not read again.
            second = self.measure.per_replicate(lambda _: np.square(centered, out=centered))
            values = np.concatenate([first, second], axis=1)
            shares = diagnostics.variance_share(values)
            worst = int(np.argmax(shares))
            moment = "mean" if worst < self.ndim else "variance"
            share = float(shares[worst])
            share_of = f"the {moment} of {self.names[worst % self.ndim]}"
            # The row of the largest squared deviation, which holds the share, among the
            # replicates that met.
            column = values[:, worst]
            row = int(np.argmax(np.abs(column - column.mean())))
            share_replicate = int(self.index[np.isfinite(self.meeting_times)][row])
        mean_cost = float(self.costs.mean())
        return {
            "meeting_time_50": float(quantiles[0]),
            "meeting_time_90": float(quantiles[1]),
            "meeting_time_99": float(quantiles[2]),
            "meeting_time_max": float(quantiles[3]),
            "tail_index": tail_index,
            "censored": self.n_censored,
            "late_fraction": n_late / n_total,
            "late_fraction_upper": float(upper),
            "tv_bound_at_k": tv_at_k,
            "variance_share": share,
            "variance_share_of": share_of,
            "variance_share_replicate": share_replicate,
            "mean_cost": mean_cost,
            "seconds_per_replicate": self.wall_time / len(self.index),
            "relative_cost": mean_cost / (self.m - self.k + 1),
        }

    # Spec R-OUT-3.
    def summary(self) -> Summary:
        """Estimates per parameter and the diagnostics of the run.

        Returns
        -------
        Summary

        Examples
        --------
        The summary has a row per parameter and a column per estimate; ``print(summary)``
        shows them as a table, with the diagnostics of the run below it:

        >>> from telescopic import examples, sample
        >>> example = examples.gaussian(2)
        >>> result = sample(
        ...     example.log_prob, example.initial,
        ...     n_replicates=50, vectorize=True, rng=0,
        ... )
        >>> summary = result.summary()
        >>> summary.names, list(summary.table)
        (['x0', 'x1'], ['mean', 'standard_error', 'std', 'q16', 'q50', 'q84'])
        """
        mean = self.mean()
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
            "max_iter": self.max_iter,
            "kernel": str(self.kernel),
            **self.diagnostics(),
            "interrupted": self.interrupted,
            "warnings": [issue.message for issue in diagnostics.issues(self)],
        }
        return Summary(self.names, table, footer)

    # Spec R-OUT-1.
    def merge(self, other: Result) -> Result:
        """Join with a run of the same seed and configuration and disjoint replicates.

        :func:`telescopic.engine.merge` checks the seeds, the settings, the
        dimensions and the replicates. That the runs share the target, the kernel and the
        initial distribution is the caller's to ensure: their descriptions omit parameters.
        The merged result takes this result's names, and its kernel, initial distribution,
        tuning report and pilot meeting times, each from ``other`` where this result has
        none.

        Parameters
        ----------
        other : Result
            The result to join with this one.

        Returns
        -------
        Result
            The union of the replicates and of the requested indices; interrupted if a
            requested replicate ran in neither.
        """
        return type(self)(
            engine.merge([self._replicates, other._replicates]),
            names=self.names,
            kernel=self.kernel if self.kernel is not None else other.kernel,
            initial=self.initial if self.initial is not None else other.initial,
            tuning=self.tuning if self.tuning is not None else other.tuning,
            pilot=self.pilot if self.pilot is not None else other.pilot,
        )

    # Spec R-OUT-1.
    def save(self, path: str | os.PathLike[str] | IO[bytes]) -> None:
        """Save as a versioned ``.npz`` file, without pickle.

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
        axes : sequence of two Axes, optional
            By default, those of a new figure.

        Returns
        -------
        list of two Axes
        """
        # Imported here: matplotlib is optional, and telescopic.plotting imports this module.
        from telescopic import plotting

        return plotting.meeting_times(self, axes)

    def plot_tv_bound(self, ax: Any = None) -> Any:
        """Plot the TV bound against t, with k marked (needs matplotlib).

        Parameters
        ----------
        ax : Axes, optional
            By default, that of a new figure.

        Returns
        -------
        Axes
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
            The number of bins along each coordinate.
        axes : sequence of Axes, optional
            One per coordinate; by default, those of a new figure.

        Returns
        -------
        list of Axes
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
            The number of bins along each coordinate.

        Returns
        -------
        ndarray of Axes
        """
        from telescopic import plotting

        return plotting.corner(self, dims, bins)

    # Spec section 7.4.
    def check(self, strict: bool = False) -> None:
        """Warn about the conditions that apply to this run.

        They are listed under Warnings, in :doc:`/guide/diagnostics`.

        Parameters
        ----------
        strict : bool
            Raise ``RuntimeError`` instead of warning.
        """
        strict = flag("strict", strict)
        messages = [issue.message for issue in diagnostics.issues(self)]
        if strict and messages:
            raise RuntimeError("; ".join(messages))
        for message in messages:
            warn(message, ConvergenceWarning)

    def _bins(
        self, column: int, bins: int, limit: tuple[float, float] | None = None
    ) -> tuple[npt.NDArray[np.intp], FloatArray]:
        # The bin of each atom along a coordinate, below 0 or from bins outside the edges,
        # and the edges; the last bin is closed, as in numpy.histogram.
        values = self.measure.atoms[:, column]
        edges = np.histogram_bin_edges(values, bins, limit)
        which = np.searchsorted(edges, values, side="right") - 1
        which[values == edges[-1]] = bins - 1
        return which, edges

    def _centered(self) -> tuple[FloatArray, FloatArray]:
        # The atoms about their mean, and each replicate's mean of them. Each replicate's
        # weights sum to 1, so the covariance estimator is the same about any point; about
        # 0, a mean far from 0 next to the spread would cancel catastrophically.
        atoms = self.measure.atoms
        centered = atoms - np.ones(len(atoms)) @ atoms / len(atoms)  # faster than mean(axis=0)
        return centered, self.measure.per_replicate(lambda _: centered)

    def _require(self, n: int) -> None:
        if self.n_replicates < n:
            raise ValueError(
                f"this needs at least {n} replicates that met, not {self.n_replicates}"
            )

    def _times(self, end: int) -> npt.NDArray[np.int64]:
        # The times t from 0 at which a bound of section 3.6 can be positive when no time
        # from end on enters it: t + L < end.
        return np.arange(max(1, end - self.lag))


@dataclass(eq=False)
class Summary:
    """Estimates per parameter and the diagnostics of a run, printable as a table.

    ``repr`` gives the table, then the footer as lines of at most 80 characters (a word
    longer than that keeps its line), with the kernel on a line of its own and its
    floats to 4 significant figures. A notebook shows the summary as an HTML table and
    the footer as a list, one warning per item.

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
        """The summary as nested dictionaries of Python numbers.

        Returns
        -------
        dict
            ``"parameters"``, a dictionary of each parameter's columns, and the entries of
            the footer.
        """
        parameters = {
            name: {column: float(values[i]) for column, values in self.table.items()}
            for i, name in enumerate(self.names)
        }
        return {"parameters": parameters, **self.footer}

    def to_pandas(self) -> Any:
        """The table as a data frame indexed by parameter name (needs pandas).

        Returns
        -------
        pandas.DataFrame
        """
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError(
                "to_pandas needs pandas: pip install 'telescopic[pandas]'"
            ) from error

        return pd.DataFrame(self.table, index=self.names)

    def __repr__(self) -> str:
        rows = self._rows()
        widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
        table = [
            "  ".join([row[0].ljust(widths[0]), *map(str.rjust, row[1:], widths[1:])])
            for row in rows
        ]
        footer = [wrapped for line in self._footer_lines() for wrapped in _WRAP.wrap(line)]
        return "\n".join([*table, "", *footer])

    def _repr_html_(self) -> str:
        (_, *columns), *rows = [[html.escape(cell) for cell in row] for row in self._rows()]
        head = "".join(f"<th>{column}</th>" for column in columns)
        body = "".join(
            f"<tr><th>{name}</th>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
            for name, *cells in rows
        )
        items = "".join(f"<li>{html.escape(line)}</li>" for line in self._footer_lines())
        return (
            f'<div class="telescopic-summary"><table><thead><tr><th></th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table><ul>{items}</ul></div>"
        )

    def _rows(self) -> list[list[str]]:
        # The table's cells as text: the header, then a row per parameter.
        return [["", *self.table]] + [
            [name, *(f"{float(values[i]):.6g}" for values in self.table.values())]
            for i, name in enumerate(self.names)
        ]

    def _footer_lines(self) -> list[str]:
        footer = self.footer
        # A quantile that falls on a censored pair is inf: beyond the cap.
        times = [
            f"> {footer['max_iter']}" if value == np.inf else f"{value:.4g}"
            for value in (footer[f"meeting_time_{q}"] for q in ("50", "90", "99", "max"))
        ]
        # The kernel's own repr keeps every digit; the summary rounds its floats.
        kernel = _FLOAT.sub(_four_figures, footer["kernel"])
        lines = [
            (
                f"R = {footer['n_replicates']}, k = {footer['k']}, m = {footer['m']}, "
                f"lag = {footer['lag']}, thin = {footer['thin']}"
            ),
            f"kernel: {kernel}",
            (
                f"meeting times: median {times[0]}, 90% {times[1]}, 99% {times[2]}, max "
                f"{times[3]}; censored {footer['censored']}"
            ),
        ]
        # The tail index needs 500 replicates.
        if not np.isnan(footer["tail_index"]):
            lines.append(f"tail index of the meeting times: {footer['tail_index']:.3g}")
        lines += [
            (
                f"cost per replicate: {footer['mean_cost']:.4g} kernel applications, "
                f"{footer['seconds_per_replicate']:.3g} s; relative cost "
                f"{footer['relative_cost']:.3g}"
            ),
            (
                f"pairs meeting after k + lag: {footer['late_fraction']:.1%} (at most "
                f"{footer['late_fraction_upper']:.1%} at 95%)"
            ),
            (
                "largest share of one replicate in a variance: "
                f"{footer['variance_share']:.0%}, for {footer['variance_share_of']}"
            ),
        ]
        if footer["interrupted"]:
            lines.append(
                "interrupted: the estimates are consistent but not exactly unbiased; run "
                "the remaining replicates to restore unbiasedness"
            )
        return lines + [f"warning: {message}" for message in footer["warnings"]]
