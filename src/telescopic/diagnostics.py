"""Diagnostics: bounds on the distance to stationarity, tail indices, warnings, gradient checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from telescopic._types import FloatArray, IntArray, integer, integral, real_array

if TYPE_CHECKING:
    from telescopic.results import Result
    from telescopic.tuning import TuningReport

__all__ = [
    "GradientCheck",
    "Issue",
    "hill",
    "issues",
    "slow_pair_threshold",
    "tuning_disagreement",
    "tv_bound",
    "variance_share",
    "w1_bound",
]


# Spec section 3.6.
def tv_bound(meeting_times: npt.ArrayLike, lag: int, t: npt.ArrayLike) -> FloatArray:
    r"""Estimated upper bound on the total variation distance to the target.

    .. math:: d_{TV}(\pi_t, \pi) \le \mathbb E\big[\max(0, \lceil (\tau - L - t)/L \rceil)\big]

    (Biswas, Jacob & Vanetti 2019, Theorem 2.5), estimated by the average over replicates
    and reported as :math:`\min(1, \cdot)`.

    Parameters
    ----------
    meeting_times : array_like of int, shape (R,)
        The meeting times :math:`\tau`. A censored pair, given as its cap plus 1, the
        least meeting time it can have, makes the estimate a lower bound of the true
        average.
    lag : int
        The lag :math:`L` of the runs that produced the meeting times.
    t : array_like of int
        Times at which to bound the distance.

    Returns
    -------
    ndarray, shape of ``t``

    Notes
    -----
    Computed as :math:`\sum_{j \ge 1} \hat S(t + jL)` with the empirical survival
    function :math:`\hat S(u) = R^{-1}\#\{\tau > u\}`, in :math:`O(R + \max\tau)`.
    """
    tau = real_array("meeting_times", meeting_times, "integers of shape (R,)")
    if tau.ndim != 1 or len(tau) == 0 or not integral(tau, 0):
        raise ValueError(
            "meeting_times must be a non-empty one-dimensional array of non-negative "
            "integers: give a censored pair as its cap plus 1"
        )
    survival = 1.0 - np.cumsum(np.bincount(tau.astype(np.int64))) / len(tau)
    bound: FloatArray = np.minimum(1.0, _lagged_tail(survival, lag, t))
    return bound


# Spec section 3.6.
def w1_bound(
    distance_time: npt.ArrayLike,
    distance: npt.ArrayLike,
    n_replicates: int,
    lag: int,
    t: npt.ArrayLike,
) -> FloatArray:
    r"""Estimated upper bound on the 1-Wasserstein distance to the target.

    .. math:: d_{W_1}(\pi_t, \pi) \le \mathbb E\Big[\sum_{j=1}^{\lceil(\tau - L - t)/L\rceil}
              \lVert X_{t+jL} - Y_{t+(j-1)L}\rVert\Big]

    (Biswas, Jacob & Vanetti 2019), estimated by the average over replicates. It depends
    on the parameterization of the target.

    Parameters
    ----------
    distance_time, distance : array_like, shape (M,)
        The times :math:`s` and distances :math:`\lVert X_s - Y_{s-L}\rVert` recorded for
        :math:`L \le s < \tau`, pooled over replicates.
    n_replicates : int
        The number of replicates :math:`R`, including those with no recorded distance.
    lag : int
        The lag :math:`L` of the runs that recorded the distances.
    t : array_like of int
        Times at which to bound the distance.

    Returns
    -------
    ndarray, shape of ``t``
    """
    n_replicates = integer("n_replicates", n_replicates, minimum=1)
    times = real_array("distance_time", distance_time, "integers of shape (M,)", dtype=None)
    distances = real_array("distance", distance, "real numbers of shape (M,)")
    if times.ndim != 1 or times.shape != distances.shape:
        raise ValueError(
            f"distance_time and distance must have one shape (M,), not {times.shape} and "
            f"{distances.shape}"
        )
    if not integral(times, 0):
        raise ValueError("distance_time must be non-negative integers")
    if not np.all(np.isfinite(distances) & (distances >= 0)):
        raise ValueError("distance must be finite and non-negative")
    total = np.bincount(times.astype(np.int64), weights=distances)
    bound: FloatArray = _lagged_tail(total, lag, t) / n_replicates
    return bound


def _lagged_tail(values: FloatArray, lag: int, t: npt.ArrayLike) -> FloatArray:
    # sum_{j >= 1} values[t + jL], with values zero past the end. tail[i, r] sums values
    # over the times i'L + r with i' >= i; the padding adds a last row of zeros.
    lag = integer("lag", lag, minimum=1)
    times = real_array("t", t, "integers", dtype=None)
    if not integral(times, 0):
        raise ValueError(f"t must be non-negative integers, not {times}")
    # Past the end every sum is 0, so a larger t needs no larger integer.
    clamped = np.minimum(times, len(values)).astype(np.int64)
    rows = -(-len(values) // lag) + 1
    padded = np.zeros(rows * lag)
    padded[: len(values)] = values
    tail = np.cumsum(padded.reshape(rows, lag)[::-1], axis=0)[::-1]
    sums: FloatArray = tail[np.minimum(clamped // lag + 1, rows - 1), clamped % lag]
    return sums


def hill(values: npt.ArrayLike, n_tail: int) -> float:
    r"""Hill's estimate of the tail index :math:`\alpha` of positive values.

    .. math:: \hat\alpha = \Big[\frac{1}{k}\sum_{i=1}^{k}
              \log\frac{x_{(n-i+1)}}{x_{(n-k)}}\Big]^{-1}

    from the :math:`k` = ``n_tail`` largest of :math:`n` values. A tail
    :math:`P(X > x) \propto x^{-\alpha}` has finite moments below :math:`\alpha` only.
    Where ties make the smallest value the threshold :math:`x_{(n-k)}`, as when most
    meeting times take their smallest possible value, the estimate is undefined: above such
    ties it reads a light tail as a heavy one. With ``n_tail`` :math:`= \lfloor n/10
    \rfloor`, as in the ``tail`` warning of :meth:`telescopic.Result.check`, this happens
    when nine values in ten or more share the smallest.

    Parameters
    ----------
    values : array_like, shape (n,)
        Positive, finite values.
    n_tail : int
        The number :math:`k` of upper order statistics, :math:`1 \le k < n`.

    Returns
    -------
    float
        ``nan`` where ties make the smallest value the threshold, and otherwise ``inf``
        when the :math:`k + 1` largest values are equal.
    """
    x = real_array("values", values, "real numbers of shape (n,)")
    if x.ndim != 1:
        raise ValueError(f"values must have shape (n,), not {x.shape}")
    x = np.sort(x)
    n_tail = integer("n_tail", n_tail)
    if not 1 <= n_tail < len(x):
        raise ValueError(
            f"n_tail must be at least 1 and below the number of values, {len(x)}, not {n_tail}"
        )
    if not (x[0] > 0 and x[-1] < np.inf):
        raise ValueError(f"values must be positive and finite, not in [{x[0]}, {x[-1]}]")
    if x[1] == x[-n_tail - 1] == x[0]:
        return np.nan
    mean_log = float(np.mean(np.log(x[-n_tail:] / x[-n_tail - 1])))
    return np.inf if mean_log == 0 else 1.0 / mean_log


def variance_share(values: npt.ArrayLike) -> FloatArray:
    """The largest share of one replicate in the sample variance of each column.

    Parameters
    ----------
    values : array_like, shape (R,) or (R, p)
        Replicate values.

    Returns
    -------
    ndarray, shape () or (p,)
        0 for a column without variance, or whose values differ only by rounding.
    """
    values = real_array("values", values, "real numbers of shape (R,) or (R, p)")
    if values.ndim not in (1, 2) or len(values) == 0:
        raise ValueError(f"values must have shape (R,) or (R, p) with R >= 1, not {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    squares = (values - values.mean(axis=0)) ** 2
    total = squares.sum(axis=0)
    # Spreads of a few units in the last place of the values are rounding, not variance.
    rounding = len(values) * (8 * np.finfo(np.float64).eps * np.abs(values).max(axis=0)) ** 2
    share: FloatArray = np.divide(
        squares.max(axis=0), total, out=np.zeros_like(total), where=total > rounding
    )
    return share


# Spec section 4.4.
@dataclass(frozen=True, eq=False)
class GradientCheck:
    r"""What the gradient check of a sampler built with ``grad`` compared.

    At each of 8 draws :math:`x` of :math:`\pi_0` the check compares, along the
    direction :math:`u` between two further draws, the change of :math:`\log\pi` that
    ``grad`` predicts with central differences of ``log_prob`` at steps
    :math:`h u`, :math:`h = \epsilon^{1/3}` for the epsilon :math:`\epsilon` of the
    dtype that ``log_prob`` returns:

    .. math:: \Delta = \big|\nabla\log\pi(x)\cdot u - D_u\big|,
       \qquad D_u = \frac{\log\pi(x + hu) - \log\pi(x - hu)}{2h},

    and a line disagrees when :math:`\Delta` exceeds its tolerance, 10 times the
    estimated error of the comparison: the truncation of the differences, from a second
    difference at :math:`2hu`, and the rounding of ``log_prob`` and of ``grad``. All the
    figures are in nats along :math:`u`, across the spread of :math:`\pi_0`.

    Attributes
    ----------
    draw : ndarray of int, shape (n,)
        Which of the 8 draws, numbered from 1, each compared line starts from.
    gap : ndarray, shape (n,)
        :math:`\Delta` of each line.
    tolerance : ndarray, shape (n,)
        The tolerance of each line.
    predicted : ndarray, shape (n,)
        The change :math:`\nabla\log\pi(x)\cdot u` that ``grad`` predicts on each line.
    n_skipped : int
        Draws skipped because the density is zero or the gradient is not finite there.
    n_unusable : int
        Lines not compared because they reach zero density, or because their steps round
        away in the dtype of ``log_prob``, as with a float32 ``log_prob`` and a start whose
        spread is tiny beside its distance from 0.
    """

    draw: IntArray
    gap: FloatArray
    tolerance: FloatArray
    predicted: FloatArray
    n_skipped: int
    n_unusable: int

    @property
    def ratio(self) -> FloatArray:
        """The gap of each line over its tolerance, which sizes each line's disagreement.

        ``inf`` for a gap above a tolerance of 0, and 0 for a gap of 0. The check decides
        on ``gap > tolerance`` itself, and uses the ratio only to choose the line that its
        warning describes.

        Returns
        -------
        ndarray, shape (n,)
        """
        ratio: FloatArray = np.divide(
            self.gap,
            self.tolerance,
            out=np.where(self.gap > 0, np.inf, 0.0),
            where=self.tolerance > 0,
        )
        return ratio


# Spec section 7.4.
class Issue(NamedTuple):
    """A condition that applies to a result and makes it warn.

    Attributes
    ----------
    kind : str
        ``"censored"``, ``"late"``, ``"variance"``, ``"tail"``, ``"tuning"`` or ``"slow"``.
    message : str
    """

    kind: str
    message: str


# Spec section 7.4.
def issues(result: Result) -> list[Issue]:
    """The warnings that apply to ``result`` (Warnings, in :doc:`/guide/diagnostics`).

    Parameters
    ----------
    result : Result
        The result to check.

    Returns
    -------
    list of Issue
        Empty for a run that raises no warning.
    """
    found = []
    values = result.diagnostics()
    if result.n_censored:
        # A pair that starts where the density is zero may never meet, so no cap would
        # leave the run waiting forever.
        censored = result.index[np.isinf(result.meeting_times)]
        outside = int(np.isin(censored, result._replicates.zero_density).sum())
        remedy = (
            f"{outside} of them started where the target density is zero, from which a "
            "pair may never meet: draw the initial distribution inside the support of the "
            "target, and run a new sampler with a larger max_iter"
            if outside
            else "run a new sampler with a larger max_iter, or none"
        )
        found.append(
            Issue(
                "censored",
                f"{result.n_censored} of {len(result.meeting_times)} pairs reached max_iter = "
                f"{result.max_iter} without meeting and are excluded, which biases the "
                f"estimates; {remedy}",
            )
        )
    if values["late_fraction"] > 0.25:
        found.append(
            Issue(
                "late",
                f"{values['late_fraction']:.0%} of the pairs met after k + lag, so k may be "
                "too small: the bias correction is large and the variance high; run a new "
                "sampler with a larger k",
            )
        )
    if result.n_replicates >= 100 and values["variance_share"] > 0.5:
        found.append(
            Issue(
                "variance",
                f"replicate {values['variance_share_replicate']} holds "
                f"{values['variance_share']:.0%} of the variance of "
                f"{values['variance_share_of']}: heavy tails or rare excursions make the "
                "error bars unreliable; check the initial distribution, and look for a mode "
                "that the other replicates missed",
            )
        )
    # The tail index is nan, which compares false, below 500 replicates.
    if values["tail_index"] < 2:
        found.append(
            Issue(
                "tail",
                f"the meeting times look heavy-tailed (tail index {values['tail_index']:.2f} "
                "< 2), so the variance of the estimates may be infinite; look at the "
                "survival function of plot_meeting_times, and at the kernel",
            )
        )
    disagreement = tuning_disagreement(result.tuning)
    if disagreement:
        found.append(Issue("tuning", disagreement))
    threshold = slow_pair_threshold(result.lag, result.m)
    if np.max(result._earliest_meeting) > threshold:
        found.append(
            Issue(
                "slow",
                f"a pair was still unmet after {threshold} iterations; the meeting times may "
                "be heavy-tailed; look at the kernel and the initial distribution",
            )
        )
    return found


# Spec section 6.3.
def slow_pair_threshold(lag: int, m: int) -> int:
    r"""The time :math:`\max(10^4, 10\max(L, m))` past which a pair is slow.

    Parameters
    ----------
    lag : int
        At least 1.
    m : int
        At least 0.

    Returns
    -------
    int
    """
    return max(10_000, 10 * max(integer("lag", lag, minimum=1), integer("m", m, minimum=0)))


# Spec section 5.3.
def tuning_disagreement(tuning: TuningReport | None) -> str | None:
    r"""The warning about tuning chains that disagree, or ``None``.

    The chains disagree when, in some coordinate, their means spread more widely than the
    chains themselves: :math:`\hat R > \sqrt 2`, a sign of several modes or of chains
    that have not converged. Chains that agree do not show that the target has one mode.
    The warning also names the coordinates whose :math:`\hat R` is NaN, where all four
    chains sat at one value, so the pilot never moved in them.

    Parameters
    ----------
    tuning : TuningReport or None
        The report of the tuning pilot; ``None`` when nothing was tuned.

    Returns
    -------
    str or None
    """
    if tuning is None or np.all(tuning.rhat <= np.sqrt(2)):
        return None
    nan = np.isnan(tuning.rhat)
    largest = np.max(tuning.rhat, where=~nan, initial=0.0)
    parts = []
    if largest > np.sqrt(2):
        parts.append(
            f"the tuning chains disagree (R-hat up to {largest:.2f}): the target may have "
            "several modes, or the chains may not have converged; several modes make "
            "meeting times heavy-tailed, and pairs that never leave one mode miss the "
            "others; unless the initial distribution already covers every mode, use one "
            "that does"
        )
    if nan.any():
        parts.append(
            f"the tuning chains never moved in coordinates {np.flatnonzero(nan).tolist()} "
            "(R-hat NaN), so the tuned kernel barely moves in them either: check that "
            "float64 resolves the target's scale at their values, or give the kernel's "
            "parameters"
        )
    return "; and ".join(parts)
