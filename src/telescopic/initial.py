"""Initial distributions: where each pair of chains starts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from telescopic._linalg import covariance
from telescopic._types import Distribution, FloatArray


class Normal:
    r"""Multivariate normal initial distribution :math:`\mathcal N(\mu, \Sigma)`.

    Parameters
    ----------
    mean : array_like, shape (d,)
    cov : array_like, shape (d, d) or (d,)
        Covariance matrix, or the variances of a diagonal one.
    """

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike) -> None:
        self.mean = np.array(mean, dtype=np.float64, ndmin=1)
        matrix = np.asarray(cov, dtype=np.float64)
        self.cov, self._chol = covariance(
            np.diag(matrix) if matrix.ndim == 1 else matrix, "cov", len(self.mean)
        )

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        """One draw, shape ``(d,)``."""
        return self.mean + self._chol @ random_state.standard_normal(len(self.mean))

    def __repr__(self) -> str:
        return f"Normal(mean={self.mean.tolist()}, cov={self.cov.tolist()})"


class Uniform:
    """Uniform initial distribution on the box ``[low, high)``.

    Parameters
    ----------
    low, high : array_like, shape (d,)
    """

    def __init__(self, low: npt.ArrayLike, high: npt.ArrayLike) -> None:
        self.low = np.array(low, dtype=np.float64, ndmin=1)
        self.high = np.array(high, dtype=np.float64, ndmin=1)
        if self.low.shape != self.high.shape or not np.all(self.low < self.high):
            raise ValueError("low and high must have the same shape, with low < high")

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        """One draw, shape ``(d,)``."""
        return random_state.uniform(self.low, self.high)

    def __repr__(self) -> str:
        return f"Uniform(low={self.low.tolist()}, high={self.high.tolist()})"


class Empirical:
    """The uniform distribution over a set of points, such as the last state of an ensemble.

    Any initial distribution keeps the estimators unbiased, so starting from the walkers
    of an earlier emcee run loses nothing.

    Parameters
    ----------
    points : array_like, shape (n, d)
    """

    def __init__(self, points: npt.ArrayLike) -> None:
        self.points = np.array(points, dtype=np.float64, ndmin=2)
        if self.points.ndim != 2 or len(self.points) == 0:
            raise ValueError(f"points must have shape (n, d) with n >= 1, not {self.points.shape}")

    def rvs(self, *, random_state: np.random.Generator) -> FloatArray:
        """One of the points, chosen uniformly, shape ``(d,)``."""
        point: FloatArray = self.points[random_state.integers(len(self.points))].copy()
        return point

    def __repr__(self) -> str:
        return f"Empirical({len(self.points)} points)"


class _Callable:
    def __init__(self, function: Callable[[np.random.Generator], Any]) -> None:
        self.function = function

    def rvs(self, *, random_state: np.random.Generator) -> Any:
        return self.function(random_state)

    def __repr__(self) -> str:
        return f"{getattr(self.function, '__name__', 'callable')}(rng)"


def as_initial(init: Any) -> Distribution:
    """Interpret ``init`` as an initial distribution (spec R-INPUT-2).

    Parameters
    ----------
    init
        An object with ``rvs(random_state=generator)``, such as a frozen SciPy
        distribution, :class:`Normal`, :class:`Uniform` or :class:`Empirical`; a callable
        ``init(rng)``; or an array of points of shape ``(n, d)``, used as
        :class:`Empirical`.

    Returns
    -------
    object with ``rvs(random_state=generator)``
    """
    if hasattr(init, "rvs"):
        distribution: Distribution = init
        return distribution
    if callable(init):
        return _Callable(init)
    points = np.asarray(init)
    if points.ndim == 2 and np.issubdtype(points.dtype, np.number):
        return Empirical(points)
    raise TypeError(
        "init must have rvs(random_state=...), like a frozen SciPy distribution, or be a "
        "callable init(rng) or an array of points of shape (n, d); a single point of shape "
        "(d,) is a point mass: pass Empirical([point]) to use one"
    )


def draw(
    initial: Distribution, generator: np.random.Generator, ndim: int | None = None
) -> FloatArray:
    """One validated draw from ``initial``: finite, of shape ``(d,)`` (or ``(ndim,)``)."""
    point = np.array(initial.rvs(random_state=generator), dtype=np.float64, ndmin=1)
    if point.ndim != 1 or (ndim is not None and len(point) != ndim):
        expected = "(d,)" if ndim is None else f"({ndim},)"
        raise ValueError(f"the initial distribution drew shape {point.shape}, not {expected}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"the initial distribution drew a non-finite point: {point.tolist()}")
    return point
