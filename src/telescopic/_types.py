"""Array and protocol types shared by the modules."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


class Distribution(Protocol):
    """A distribution that draws with ``rvs(random_state=generator)``, as frozen SciPy ones do.

    Initial distributions need only ``rvs``. The couplings of :mod:`telescopic.couplings`
    also need a normalised ``logpdf``, or ``logpmf`` for a discrete distribution.
    """

    def rvs(self, *, random_state: np.random.Generator) -> Any: ...
