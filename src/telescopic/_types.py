"""Array and protocol types shared by the modules, and checks of arguments."""

from __future__ import annotations

import numbers
import operator
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
    also need a normalised ``logpdf``, or ``logpmf`` for a discrete distribution.
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


def vector(name: str, values: npt.ArrayLike) -> FloatArray:
    # A finite vector of shape (d,), d >= 1, or a scalar as shape (1,).
    array = np.array(values, dtype=np.float64, ndmin=1)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must have shape (d,) with d >= 1, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, not {array.tolist()}")
    return array


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
