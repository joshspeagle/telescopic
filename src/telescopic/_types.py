"""Array and protocol types shared by the modules, and checks of number arguments."""

from __future__ import annotations

import numbers
import operator
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


def integer(name: str, value: Any, expected: str = "an integer") -> int:
    # An int by operator.index, as range() takes one: NumPy integers pass, and a float,
    # integral or not, raises TypeError naming the argument (DECISIONS #44, #50).
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be {expected}, not {value!r}") from None


def real(name: str, value: Any) -> float:
    # A float from a Python or NumPy real number; a string, an array or None raises
    # TypeError naming the argument (DECISIONS #50).
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {value!r}")
    return float(value)
