"""Array and protocol types shared by the modules, and checks of arguments."""

from __future__ import annotations

import numbers
import operator
import reprlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from numpy.random import Generator, SeedSequence

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]

# What every function that draws random numbers accepts as its rng (SPEC 7).
Seed = int | SeedSequence | Generator | None


class Distribution(Protocol):
    """A distribution that draws with ``rvs(random_state=generator)``, as frozen SciPy ones do.

    Initial distributions need only ``rvs``. The couplings of :mod:`telescopic.couplings`
    also need a normalized ``logpdf``, or ``logpmf`` for a discrete distribution.
    """

    def rvs(self, *, random_state: np.random.Generator) -> Any: ...


def integer(
    name: str, value: Any, expected: str = "an integer", minimum: int | None = None
) -> int:
    # A Python int by operator.index, as range() takes one: NumPy integers pass, and a
    # float, integral or not, or a bool raises TypeError naming the argument; one below
    # minimum raises ValueError (DECISIONS #44, #50).
    try:
        number = operator.index(value)
    except TypeError:
        number = None
    if number is None or isinstance(value, bool):
        raise TypeError(f"{name} must be {expected}, not {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, not {number}")
    return number


def flag(name: str, value: Any) -> bool:
    # True or False, NumPy's bools included. Anything else raises TypeError naming the
    # argument: a truthy string such as "no" would act as True (DECISIONS #104).
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be True or False, not {value!r}")
    return bool(value)


def instance(name: str, value: Any, kind: type) -> None:
    # A telescopic object, such as a Target or a Kernel, checked where it is first passed:
    # a function, a string or a class instead failed later, on an attribute (DECISIONS
    # #106).
    if not isinstance(value, kind):
        given = f"the class {value.__name__}" if isinstance(value, type) else type(value).__name__
        raise TypeError(f"{name} must be a {kind.__name__} instance, not {given}")


def real(
    name: str, value: Any, low: float = -np.inf, high: float = np.inf, *, closed: bool = False
) -> float:
    # A float in (low, high), or in (low, high] if closed, from a Python or NumPy real
    # number. A bool, a string, an array or None raises TypeError, and a number outside,
    # NaN included, ValueError, each naming the argument (DECISIONS #50).
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not {value!r}")
    number = float(value)
    if not (low < number < high or (closed and number == high)):
        bracket = "]" if closed else ")"
        raise ValueError(f"{name} must be in ({low:g}, {high:g}{bracket}, not {number}")
    return number


def integral(values: npt.NDArray[Any], minimum: int | None = None) -> bool:
    # Whether every value is an integer from minimum up: an integer, or a float with an
    # integral value. Integers skip the float tests, as on the engine's path.
    if values.dtype.kind not in "iu" and not np.all(
        np.isfinite(values) & (values == np.floor(values))
    ):
        return False
    return minimum is None or bool(np.all(values >= minimum))


def coordinates(dims: int | Sequence[int], ndim: int) -> npt.NDArray[np.intp]:
    # Coordinates given as one integer or a sequence of them, each in [0, ndim).
    columns = np.atleast_1d(dims)
    if columns.size == 0:
        raise ValueError("dims must name at least one coordinate")
    if columns.ndim != 1 or columns.dtype.kind not in "iu":
        raise TypeError(f"dims must be an integer or a sequence of integers, not {dims!r}")
    if np.any((columns < 0) | (columns >= ndim)):
        raise ValueError(f"dims must be in [0, {ndim}), not {dims}")
    return columns.astype(np.intp)


def real_array(
    name: str,
    values: Any,
    expected: str,
    *,
    dtype: type[np.floating[Any]] | None = np.float64,
    ndmin: int = 0,
) -> npt.NDArray[Any]:
    # values as a new array of real numbers, of dtype unless None, which keeps integers.
    # Strings, None, booleans, complex numbers and ragged sequences raise TypeError naming
    # the argument, with NumPy's error chained when it raised one (DECISIONS #103).
    cause = None
    try:
        array = np.array(values, ndmin=ndmin, order="C")
    except (TypeError, ValueError) as error:
        cause = error
    if cause is not None or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be {expected}, not {reprlib.repr(values)}") from cause
    return array if dtype is None else array.astype(dtype, copy=False)


def vector(name: str, values: npt.ArrayLike, ndim: int | None = None) -> FloatArray:
    # A finite vector of shape (d,), d >= 1, or a scalar as shape (1,).
    array = real_array(name, values, "real numbers of shape (d,)", ndmin=1)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must have shape (d,) with d >= 1, not {array.shape}")
    if ndim is not None and len(array) != ndim:
        raise ValueError(f"{name} must have shape ({ndim},), not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, not {array.tolist()}")
    return array


def log_densities(name: str, values: FloatArray, x: FloatArray) -> FloatArray:
    # values, the log-densities that name returned at the rows of x, unless one is NaN or
    # +inf, which raise ValueError with the first such point (spec R-INPUT-1). One
    # reduction finds them, since they make the maximum fail the test.
    if not np.max(values, initial=-np.inf) < np.inf:
        row = int(np.argmax(np.isnan(values) | (values == np.inf)))
        raise ValueError(f"{name} returned {values[row]} at x = {x[row].tolist()}")
    return values


def averaging(k: Any, m: Any, thin: Any) -> tuple[int, int, int]:
    # The settings of the averaging set S: 0 <= k <= m and thin >= 1, as ints.
    k, m = integer("k", k, minimum=0), integer("m", m, minimum=0)
    if m < k:
        raise ValueError(f"m must be at least k = {k}, not {m}")
    return k, m, integer("thin", thin, minimum=1)


def parameter_names(names: Sequence[str] | None, ndim: int) -> list[str]:
    # Distinct names of the ndim coordinates; x0, x1, ... by default.
    if names is None:
        return [f"x{i}" for i in range(ndim)]
    # Listed first, so that an iterator is read once; a string would list its letters.
    listed = None if isinstance(names, str) else list(names)
    if listed is None or not all(isinstance(name, str) for name in listed):
        raise TypeError(f"names must be a sequence of strings, not {names!r}")
    if len(listed) != ndim:
        raise ValueError(f"names must have {ndim} entries, not {len(listed)}")
    if len(set(listed)) != ndim:
        raise ValueError(f"names must be distinct, not {listed}")
    return listed


def read_only(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    # A view that the user's functions cannot write through.
    view = array.view()
    view.flags.writeable = False
    return view
