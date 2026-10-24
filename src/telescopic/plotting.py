"""Plots of results: meeting times, the TV bound and signed histograms (needs matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from telescopic.engine import Replicates
from telescopic.results import Result

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as error:
    raise ImportError(
        "telescopic's plots need matplotlib: pip install 'telescopic[plot]'"
    ) from error


def meeting_times(result: Result, axes: Sequence[Axes] | None = None) -> list[Axes]:
    r"""The meeting times: a histogram of :math:`\tau - L`, and its survival function.

    The survival function :math:`P(\tau - L > t)` is on a log scale, where a straight
    tail means geometric meeting times (spec section 7.5). A censored pair counts in it
    up to the cap, and not in the histogram.

    Parameters
    ----------
    result : Result
    axes : sequence of two Axes, optional
        By default, those of a new figure.

    Returns
    -------
    list of two Axes
    """
    histogram, survival = (
        plt.subplots(1, 2, figsize=(9, 3.5))[1] if axes is None else _axes(axes, 2)
    )
    shifted = result.meeting_times - result.lag
    histogram.hist(shifted[np.isfinite(shifted)], bins="auto")
    histogram.set(xlabel=r"$\tau - L$", ylabel="pairs")
    stopped = np.sort(result._stopped - result.lag)
    t = np.union1d(0, stopped)  # the function steps only there
    survival.step(t, 1 - np.searchsorted(stopped, t, side="right") / len(stopped), where="post")
    survival.set(yscale="log", xlabel="$t$", ylabel=r"$P(\tau - L > t)$")
    return [histogram, survival]


def tv_bound(result: Result, ax: Axes | None = None) -> Axes:
    """The estimated bound on the total variation distance to the target against t.

    The bound is on a log scale, with ``k`` marked (spec section 3.6).

    Parameters
    ----------
    result : Result
    ax : Axes, optional
        By default, that of a new figure.

    Returns
    -------
    Axes
    """
    ax = plt.subplots(figsize=(5, 3.5))[1] if ax is None else ax
    # At most 4,000 times draw the step function, however long the tail of meeting times.
    times = result._times()
    t = times[:: max(1, len(times) // 2000)]
    bound = result.tv_bound(t)
    positive = bound > 0
    ax.plot(t[positive], bound[positive])
    ax.axvline(result.k, color="grey", linestyle="--", label="$k$")
    ax.set(yscale="log", xlabel="$t$", ylabel=r"TV bound on $X_t$")
    ax.legend(loc="upper right")
    return ax


def marginals(
    result: Result,
    dims: int | Sequence[int] | None = None,
    bins: int = 20,
    axes: Sequence[Axes] | None = None,
) -> list[Axes]:
    """Signed histograms of the marginals, with boxes of plus or minus 2 standard errors.

    A bin's density is the average over replicates of its signed mass, divided by its
    width, so a bin can be negative: the bias corrections carry negative weights, and
    where few atoms fall they can outweigh the others (spec section 7.5). The boxes show
    the Monte Carlo error.

    Parameters
    ----------
    result : Result
    dims : int or sequence of int, optional
        Coordinates; all by default.
    bins : int
    axes : sequence of Axes, optional
        One per coordinate; by default, those of a new figure.

    Returns
    -------
    list of Axes
    """
    columns = range(result.ndim) if dims is None else result._columns(dims)
    n = len(columns)
    if axes is None:
        axes = plt.subplots(1, n, figsize=(3.5 * n, 3), squeeze=False)[1][0]
    axes = _axes(axes, n)
    for ax, dim in zip(axes, columns, strict=True):
        masses, errors, edges = result.histogram(dim, bins)
        widths = np.diff(edges)
        density, error = masses / widths, errors / widths
        ax.bar(edges[:-1], density, widths, align="edge", alpha=0.6)
        ax.bar(edges[:-1], 4 * error, widths, density - 2 * error, align="edge", fill=False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(xlabel=result.names[dim], ylabel="density")
    return axes


def corner(result: Result, dims: int | Sequence[int] | None = None, bins: int = 20) -> Any:
    """A corner plot of signed histograms: marginals on the diagonal, pairs below it.

    The pairs are coloured by signed density with a diverging map centred on 0
    (spec section 7.5). Telescopic never passes signed weights to ``corner`` or ArviZ,
    whose weighted quantiles assume non-negative weights.

    Parameters
    ----------
    result : Result
    dims : int or sequence of int, optional
        Coordinates; all by default.
    bins : int

    Returns
    -------
    ndarray of Axes, shape (len(dims), len(dims))
    """
    columns = range(result.ndim) if dims is None else result._columns(dims)
    n = len(columns)
    grid = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), squeeze=False)[1]
    # Each coordinate binned once, as Result.histogram bins it; a pair's masses are then one
    # bincount, without the per-replicate masses that its errors would need.
    cells = [result._bins(dim, bins) for dim in columns]
    weights = result.measure.weights / result.n_replicates
    for row in range(n):
        marginals(result, columns[row], bins, [grid[row, row]])
        for column in range(n):
            ax = grid[row, column]
            if column > row:
                ax.axis("off")
            elif column < row:
                (which_x, edges_x), (which_y, edges_y) = cells[column], cells[row]
                masses = np.bincount(which_x * bins + which_y, weights, bins * bins)
                area = np.outer(np.diff(edges_x), np.diff(edges_y))
                density = masses.reshape(bins, bins) / area
                limit = np.max(np.abs(density))
                ax.pcolormesh(edges_x, edges_y, density.T, cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set(xlabel=result.names[columns[column]], ylabel=result.names[columns[row]])
    return grid


def paths(
    replicates: Replicates,
    dims: Sequence[int] = (0, 1),
    n: int = 3,
    ax: Axes | None = None,
) -> Axes:
    """The coupled paths of a few replicates, for teaching: X solid, Y dashed.

    They need the paths that :func:`telescopic.engine.run` keeps with
    ``keep_paths=True``. In one dimension they run against time, with Y shifted by the
    lag so that each pair meets at its meeting time; otherwise they are drawn in the
    plane of ``dims`` (spec section 7.5).

    Parameters
    ----------
    replicates : telescopic.engine.Replicates
    dims : two ints
        The plane, for more than one dimension.
    n : int
        Replicates drawn, the first ones.
    ax : Axes, optional
        By default, that of a new figure.

    Returns
    -------
    Axes
    """
    if not replicates.paths:
        raise ValueError("the replicates hold no paths: run the engine with keep_paths=True")
    ax = plt.subplots(figsize=(5, 4))[1] if ax is None else ax
    for index in sorted(replicates.paths)[:n]:
        x, y = replicates.paths[index]
        if x.shape[1] == 1:
            line = ax.plot(np.arange(len(x)), x[:, 0])[0]
            shifted = replicates.settings.lag + np.arange(len(y))
            ax.plot(shifted, y[:, 0], linestyle="--", color=line.get_color())
        else:
            line = ax.plot(x[:, dims[0]], x[:, dims[1]])[0]
            ax.plot(y[:, dims[0]], y[:, dims[1]], linestyle="--", color=line.get_color())
    return ax


def _axes(axes: Any, n: int) -> list[Axes]:
    # The given axes, flattened, checked to number n.
    flat = list(np.ravel(axes))
    if len(flat) != n:
        raise ValueError(f"axes must hold {n} Axes, not {len(flat)}")
    return flat
