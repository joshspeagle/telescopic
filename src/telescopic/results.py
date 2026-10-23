"""Results of a run: estimates, moments, quantiles, diagnostics and the summary."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence
from scipy import stats

from telescopic import diagnostics
from telescopic._types import FloatArray
from telescopic.engine import Replicates
from telescopic.estimators import unbiased_covariance, weighted_quantile
from telescopic.measure import SignedMeasure
from telescopic.warnings import ConvergenceWarning


@dataclass(frozen=True)
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
        The engine's output.
    k, m, lag, thin : int
        The configuration of the run (spec section 3.2).
    names : sequence of str, optional
        Parameter names; ``x0``, ``x1``, ... by default.
    max_iter : int, optional
        The cap of the run; required when a pair was censored.
    kernel, initial : optional
        The frozen kernel and the initial distribution, for the record.
    seed : numpy.random.SeedSequence, optional
        The root seed, whose ``entropy`` and ``spawn_key`` reproduce the run.
    n_log_prob, n_grad : int
        Evaluations of the log-density and its gradient during the run.
    wall_time : float
        Duration of the run in seconds.
    tuning, pilot : optional
        Reports of the pilot runs.
    interrupted : bool
        Whether the run was stopped early (spec section 6.5).

    Attributes
    ----------
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
    """

    def __init__(
        self,
        replicates: Replicates,
        *,
        k: int,
        m: int,
        lag: int,
        thin: int = 1,
        names: Sequence[str] | None = None,
        max_iter: int | None = None,
        kernel: Any = None,
        initial: Any = None,
        seed: SeedSequence | None = None,
        n_log_prob: int = 0,
        n_grad: int = 0,
        wall_time: float = 0.0,
        tuning: Any = None,
        pilot: Any = None,
        interrupted: bool = False,
    ) -> None:
        met = replicates.meeting_time >= 0
        self.n_replicates = int(met.sum())
        self.n_censored = len(met) - self.n_replicates
        if self.n_censored and max_iter is None:
            raise ValueError("a result with censored pairs needs the max_iter of the run")
        self.ndim = replicates.atoms.shape[1]
        self.names = [f"x{i}" for i in range(self.ndim)] if names is None else list(names)
        if len(self.names) != self.ndim:
            raise ValueError(f"names must have {self.ndim} entries, not {len(self.names)}")
        self.index = replicates.index
        self.meeting_times = np.where(met, replicates.meeting_time, np.inf)
        self.costs = replicates.cost
        self.measure = SignedMeasure(
            replicates.atoms,
            replicates.weights,
            np.searchsorted(replicates.index[met], replicates.owner),
            self.n_replicates,
        )
        self.k, self.m, self.lag, self.thin, self.max_iter = k, m, lag, thin, max_iter
        self.kernel, self.initial, self.seed = kernel, initial, seed
        self.n_log_prob, self.n_grad, self.wall_time = n_log_prob, n_grad, wall_time
        self.tuning, self.pilot, self.interrupted = tuning, pilot, interrupted
        # The time each pair stopped: its meeting time, or the cap for a censored pair,
        # which makes the bounds of section 3.6 lower bounds of their estimates.
        stopped = (
            self.meeting_times if max_iter is None else np.minimum(self.meeting_times, max_iter)
        )
        self._stopped = stopped.astype(np.int64)
        self._distance_time, self._distance = replicates.distance_time, replicates.distance

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
        if self.n_replicates < 2:
            raise ValueError(f"an error bar needs 2 replicates that met, not {self.n_replicates}")
        values = self.measure.per_replicate(h)
        value = values.mean(axis=0)
        error = values.std(axis=0, ddof=1) / np.sqrt(self.n_replicates)
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
        d = self.ndim
        first = self.measure.per_replicate(lambda x: x)
        second = self.measure.per_replicate(
            lambda x: (x[:, :, None] * x[:, None, :]).reshape(len(x), d * d)
        )
        return unbiased_covariance(first, second.reshape(-1, d, d))

    def var(self) -> FloatArray:
        """Unbiased estimates of the variances, shape ``(d,)``; they can be negative."""
        variances: FloatArray = np.diag(self.cov()).copy()
        return variances

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
        weights = self.measure.weights / self.n_replicates
        columns = [weighted_quantile(column, weights, q) for column in self.measure.atoms.T]
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
        self, dim: int, bins: int = 20, range: tuple[float, float] | None = None
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Unbiased histogram of one coordinate, with standard errors (spec section 3.4).

        The mass of each half-open bin is the average over replicates of its signed mass,
        so a bin can be negative.

        Parameters
        ----------
        dim : int
            The coordinate.
        bins : int
            The number of bins.
        range : (float, float), optional
            The lower and upper edges; by default the range of the atoms.

        Returns
        -------
        masses, standard_errors : ndarray, shape (bins,)
        edges : ndarray, shape (bins + 1,)
        """
        values = self.measure.atoms[:, dim]
        edges = np.histogram_bin_edges(values, bins=bins, range=range)
        if range is None:
            edges[-1] = np.nextafter(edges[-1], np.inf)  # keep the largest atom inside
        which = np.searchsorted(edges, values, side="right") - 1
        inside = (which >= 0) & (which < bins)
        masses = np.bincount(
            self.measure.replicate[inside] * bins + which[inside],
            weights=self.measure.weights[inside],
            minlength=self.n_replicates * bins,
        ).reshape(self.n_replicates, bins)
        errors = masses.std(axis=0, ddof=1) / np.sqrt(self.n_replicates)
        return masses.mean(axis=0), errors, edges

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
        if t is None:
            t = np.arange(max(1, int(np.max(self._stopped)) - self.lag + 1))
        return diagnostics.tv_bound(self._stopped, self.lag, t)

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
        if t is None:
            t = np.arange(max(1, int(np.max(self._stopped)) - self.lag + 1))
        return diagnostics.w1_bound(
            self._distance_time, self._distance, len(self.index), self.lag, t
        )

    def diagnostics(self) -> dict[str, Any]:
        """Diagnostics of the run: meeting times, censoring, bias corrections, cost.

        Returns
        -------
        dict
            ``meeting_time`` quantiles (50, 90, 99) and maximum, ``censored``,
            ``late_fraction`` (pairs with :math:`\\tau - L > k`, an estimate of the TV
            bound at :math:`k`), ``mean_cost``, ``seconds_per_replicate`` and
            ``inefficiency`` (mean cost over :math:`m - k + 1`; spec R-TUNE-4).
        """
        met = self.meeting_times[np.isfinite(self.meeting_times)]
        quantiles = np.quantile(met, [0.5, 0.9, 0.99]) if len(met) else np.full(3, np.nan)
        mean_cost = float(self.costs.mean())
        return {
            "meeting_time_50": float(quantiles[0]),
            "meeting_time_90": float(quantiles[1]),
            "meeting_time_99": float(quantiles[2]),
            "meeting_time_max": float(met.max()) if len(met) else np.nan,
            "censored": self.n_censored,
            "late_fraction": float(np.mean(met - self.lag > self.k)) if len(met) else np.nan,
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
            "kernel": repr(self.kernel),
            **self.diagnostics(),
            "interrupted": self.interrupted,
            "warnings": diagnostics.issues(self),
        }
        return Summary(self.names, table, footer)

    def check(self, strict: bool = False) -> None:
        """Warn about the conditions of spec section 7.4 that apply to this run.

        Parameters
        ----------
        strict : bool
            Raise ``RuntimeError`` instead of warning.
        """
        messages = diagnostics.issues(self)
        if strict and messages:
            raise RuntimeError("; ".join(messages))
        for message in messages:
            warnings.warn(message, ConvergenceWarning, stacklevel=2)


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

    def __init__(
        self, names: list[str], table: dict[str, FloatArray], footer: dict[str, Any]
    ) -> None:
        self.names, self.table, self.footer = names, table, footer

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
                f"{footer['late_fraction']:.1%}"
            ),
        ]
        if footer["interrupted"]:
            lines.append(
                "interrupted: the estimates are consistent but not exactly unbiased; run "
                "the remaining replicates to restore unbiasedness"
            )
        lines += [f"warning: {message}" for message in footer["warnings"]]
        return "\n".join([header, *rows, "", *lines])
