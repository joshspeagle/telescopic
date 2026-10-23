"""Plots of results: meeting times, the TV bound and signed histograms (needs matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from telescopic._types import FloatArray
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
    histogram, survival = plt.subplots(1, 2, figsize=(9, 3.5))[1] if axes is None else axes
    shifted = result.meeting_times - result.lag
    histogram.hist(shifted[np.isfinite(shifted)], bins="auto")
    histogram.set(xlabel=r"$\tau - L$", ylabel="pairs")
    cap = np.inf if result.max_iter is None else result.max_iter - result.lag
    stopped = np.sort(np.minimum(shifted, cap))
    t = np.arange(int(stopped[-1]) + 1)
    above = 1 - np.searchsorted(stopped, t, side="right") / len(stopped)
    survival.step(t, above, where="post")
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
    bound = result.tv_bound()
    positive = bound > 0
    ax.plot(np.arange(len(bound))[positive], bound[positive])
    ax.axvline(result.k, color="grey", linestyle="--", label="$k$")
    ax.set(yscale="log", xlabel="$t$", ylabel=r"TV bound on $X_t$")
    ax.legend()
    return ax


def marginals(
    result: Result,
    dims: Sequence[int] | None = None,
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
    dims : sequence of int, optional
        Coordinates; all by default.
    bins : int
    axes : sequence of Axes, optional
        One per coordinate; by default, those of a new figure.

    Returns
    -------
    list of Axes
    """
    dims = list(range(result.ndim)) if dims is None else list(dims)
    if axes is None:
        axes = list(plt.subplots(1, len(dims), figsize=(3.5 * len(dims), 3), squeeze=False)[1][0])
    for ax, dim in zip(axes, dims, strict=True):
        masses, errors, edges = result.histogram(dim, bins)
        widths = np.diff(edges)
        density, error = masses / widths, errors / widths
        ax.bar(edges[:-1], density, widths, align="edge", alpha=0.6)
        ax.bar(edges[:-1], 4 * error, widths, density - 2 * error, align="edge", fill=False)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(xlabel=result.names[dim], ylabel="density")
    return list(axes)


def corner(result: Result, dims: Sequence[int] | None = None, bins: int = 20) -> Any:
    """A corner plot of signed histograms: marginals on the diagonal, pairs below it.

    The pairs are coloured by signed density with a diverging map centred on 0
    (spec section 7.5). Telescopic never passes signed weights to ``corner`` or ArviZ,
    whose weighted quantiles assume non-negative weights.

    Parameters
    ----------
    result : Result
    dims : sequence of int, optional
        Coordinates; all by default.
    bins : int

    Returns
    -------
    ndarray of Axes, shape (len(dims), len(dims))
    """
    dims = list(range(result.ndim)) if dims is None else list(dims)
    n = len(dims)
    grid = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), squeeze=False)[1]
    cells = [_cells(result.measure.atoms[:, dim], bins) for dim in dims]
    for row in range(n):
        marginals(result, [dims[row]], bins, [grid[row, row]])
        for column in range(n):
            if column > row:
                grid[row, column].axis("off")
            elif column < row:
                (which_x, edges_x), (which_y, edges_y) = cells[column], cells[row]
                mass = result.measure.mass(which_x * bins + which_y, bins * bins).mean(axis=0)
                area = np.outer(np.diff(edges_x), np.diff(edges_y))
                density = (mass.reshape(bins, bins) / area).T
                limit = np.max(np.abs(density))
                grid[row, column].pcolormesh(
                    edges_x, edges_y, density, cmap="RdBu_r", vmin=-limit, vmax=limit
                )
                grid[row, column].set(xlabel=result.names[dims[column]])
                grid[row, column].set(ylabel=result.names[dims[row]])
    return grid


def paths(
    replicates: Replicates,
    lag: int,
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
    lag : int
        The lag of the run.
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
            ax.plot(lag + np.arange(len(y)), y[:, 0], linestyle="--", color=line.get_color())
        else:
            line = ax.plot(x[:, dims[0]], x[:, dims[1]])[0]
            ax.plot(y[:, dims[0]], y[:, dims[1]], linestyle="--", color=line.get_color())
    return ax


def _cells(values: FloatArray, bins: int) -> tuple[Any, FloatArray]:
    # The bin of each value, and the edges, the last nudged up to keep the largest inside.
    edges = np.histogram_bin_edges(values, bins=bins)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return np.searchsorted(edges, values, side="right") - 1, edges
