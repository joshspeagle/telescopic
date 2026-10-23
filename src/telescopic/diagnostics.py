"""Coupling diagnostics: bounds on the distance to stationarity, tail indices and warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from telescopic._types import FloatArray

if TYPE_CHECKING:
    from telescopic.results import Result
    from telescopic.tuning import TuningReport


def tv_bound(meeting_times: npt.ArrayLike, lag: int, t: npt.ArrayLike) -> FloatArray:
    r"""Estimated upper bound on the total variation distance to the target (spec section 3.6).

    .. math:: d_{TV}(\pi_t, \pi) \le \mathbb E\big[\max(0, \lceil (\tau - L - t)/L \rceil)\big]

    (Biswas, Jacob & Vanetti 2019, Theorem 2.5), estimated by the average over replicates
    and reported as :math:`\min(1, \cdot)`.

    Parameters
    ----------
    meeting_times : array_like of int, shape (R,)
        The meeting times :math:`\tau`. A censored pair, given as its cap, makes the
        estimate a lower bound of the true average.
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
    tau = np.asarray(meeting_times, dtype=np.float64)
    if not np.all(np.isfinite(tau) & (tau >= 0)):
        raise ValueError(
            "meeting_times must be finite and non-negative: give a censored pair as its cap"
        )
    survival = 1.0 - np.cumsum(np.bincount(tau.astype(np.int64))) / len(tau)
    bound: FloatArray = np.minimum(1.0, _lagged_tail(survival, lag, t))
    return bound


def w1_bound(
    distance_time: npt.ArrayLike,
    distance: npt.ArrayLike,
    n_replicates: int,
    lag: int,
    t: npt.ArrayLike,
) -> FloatArray:
    r"""Estimated upper bound on the 1-Wasserstein distance to the target (spec section 3.6).

    .. math:: d_{W_1}(\pi_t, \pi) \le \mathbb E\Big[\sum_{j=1}^{\lceil(\tau - L - t)/L\rceil}
              \lVert X_{t+jL} - Y_{t+(j-1)L}\rVert\Big]

    (Biswas, Jacob & Vanetti 2019), estimated by the average over replicates. It depends
    on the parametrisation of the target.

    Parameters
    ----------
    distance_time, distance : array_like, shape (M,)
        The times :math:`s` and distances :math:`\lVert X_s - Y_{s-L}\rVert` recorded for
        :math:`L \le s < \tau`, pooled over replicates.
    n_replicates : int
        The number of replicates :math:`R`, including those with no recorded distance.
    lag : int
    t : array_like of int

    Returns
    -------
    ndarray, shape of ``t``
    """
    total = np.bincount(
        np.asarray(distance_time, dtype=np.int64), weights=np.asarray(distance, dtype=np.float64)
    )
    bound: FloatArray = _lagged_tail(total, lag, t) / n_replicates
    return bound


def _lagged_tail(values: FloatArray, lag: int, t: npt.ArrayLike) -> FloatArray:
    # sum_{j >= 1} values[t + jL], with values zero past the end. tail[i, r] sums values
    # over the times i'L + r with i' >= i; the padding adds a last row of zeros.
    t = np.asarray(t, dtype=np.int64)
    if np.any(t < 0):
        raise ValueError(f"t must be non-negative, not {t}")
    rows = -(-len(values) // lag) + 1
    padded = np.zeros(rows * lag)
    padded[: len(values)] = values
    tail = np.cumsum(padded.reshape(rows, lag)[::-1], axis=0)[::-1]
    sums: FloatArray = tail[np.minimum(t // lag + 1, rows - 1), t % lag]
    return sums


def hill(values: npt.ArrayLike, n_tail: int) -> float:
    r"""Hill's estimate of the tail index :math:`\alpha` of positive values.

    .. math:: \hat\alpha = \Big[\frac{1}{k}\sum_{i=1}^{k}
              \log\frac{x_{(n-i+1)}}{x_{(n-k)}}\Big]^{-1}

    from the :math:`k` = ``n_tail`` largest of :math:`n` values. A tail
    :math:`P(X > x) \propto x^{-\alpha}` has finite moments below :math:`\alpha` only.

    Parameters
    ----------
    values : array_like, shape (n,)
        Positive values.
    n_tail : int
        The number :math:`k` of upper order statistics, :math:`1 \le k < n`.

    Returns
    -------
    float
        ``inf`` when the largest values are all equal.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    if not x[0] > 0:
        raise ValueError(f"values must be positive, not as small as {x[0]}")
    if not 1 <= n_tail < len(x):
        raise ValueError(f"n_tail must be in [1, {len(x) - 1}], not {n_tail}")
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
        0 for a column without variance.
    """
    values = np.asarray(values, dtype=np.float64)
    squares = (values - values.mean(axis=0)) ** 2
    total = squares.sum(axis=0)
    share: FloatArray = np.divide(
        squares.max(axis=0), total, out=np.zeros_like(total), where=total > 0
    )
    return share


class Issue(NamedTuple):
    """A condition of spec section 7.4 that applies to a result.

    Attributes
    ----------
    kind : str
        ``"censored"``, ``"late"``, ``"variance"``, ``"tail"``, ``"tuning"`` or ``"slow"``.
    message : str
    """

    kind: str
    message: str


def issues(result: Result) -> list[Issue]:
    """The warnings of spec section 7.4 that apply to ``result``.

    Parameters
    ----------
    result : Result

    Returns
    -------
    list of Issue
        Empty for a run that raises no warning.
    """
    found = []
    tau = result.meeting_times
    met = tau[np.isfinite(tau)]
    if result.n_censored:
        found.append(
            Issue(
                "censored",
                f"{result.n_censored} of {len(tau)} pairs reached max_iter = {result.max_iter} "
                "without meeting and are excluded, which biases the estimates; run them "
                "without a cap or with a larger one",
            )
        )
    late = result.diagnostics()["late_fraction"]
    if late > 0.25:
        found.append(
            Issue(
                "late",
                f"{late:.0%} of the pairs met after k + lag, so k may be too small: the bias "
                "correction is large and the variance high",
            )
        )
    if result.n_replicates >= 100:
        moments = [result.measure.per_replicate(h) for h in (lambda x: x, np.square)]
        shares = variance_share(np.concatenate(moments, axis=1))
        worst = int(np.argmax(shares))
        if shares[worst] > 0.5:
            name = result.names[worst % result.ndim]
            moment = "mean" if worst < result.ndim else "second moment"
            found.append(
                Issue(
                    "variance",
                    f"one replicate holds {shares[worst]:.0%} of the variance of the {moment} "
                    f"of {name}: heavy tails or rare excursions make the error bars unreliable",
                )
            )
    if result.n_replicates >= 500:
        alpha = hill(met - result.lag + 1, max(10, result.n_replicates // 10))
        if alpha < 2:
            found.append(
                Issue(
                    "tail",
                    f"the meeting times look heavy-tailed (tail index {alpha:.2f} < 2), so the "
                    "variance of the estimates may be infinite",
                )
            )
    disagreement = tuning_disagreement(result.tuning)
    if disagreement:
        found.append(Issue("tuning", disagreement))
    threshold = slow_pair_threshold(result.lag, result.m)
    if np.any(met > threshold):
        found.append(
            Issue(
                "slow",
                f"a pair was still unmet after {threshold} iterations; the meeting times may "
                "be heavy-tailed",
            )
        )
    return found


def slow_pair_threshold(lag: int, m: int) -> int:
    r"""The time :math:`\max(10^4, 10\max(L, m))` past which a pair is slow (spec section 6.3).

    Parameters
    ----------
    lag, m : int

    Returns
    -------
    int
    """
    return max(10_000, 10 * max(lag, m))


def tuning_disagreement(tuning: TuningReport | None) -> str | None:
    r"""The warning about tuning chains that disagree (spec section 5.3), or ``None``.

    The chains disagree when, in some coordinate, their means spread more widely than the
    chains themselves: :math:`\hat R > \sqrt 2`, a sign of several modes.

    Parameters
    ----------
    tuning : TuningReport or None
        The report of the tuning pilot; ``None`` when nothing was tuned.

    Returns
    -------
    str or None
    """
    if tuning is None or not np.max(tuning.rhat) > np.sqrt(2):
        return None
    return (
        f"the tuning chains disagree (R-hat up to {np.max(tuning.rhat):.2f}): the target "
        "may have several modes, which make meeting times heavy-tailed, and pairs that "
        "never leave one mode miss the others; use an initial distribution that covers "
        "every mode"
    )
