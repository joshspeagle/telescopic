"""The data of the line fit of Getting started, for the pages that reuse it."""

import numpy as np


def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x``, ``y`` and ``yerr``: 50 points of the line ``-0.9 x + 4.3`` with Gaussian errors.

    Getting started and the README write this code out in full. Only Getting started
    checks that its data equal these; nothing checks the README's copy.
    """
    rng = np.random.default_rng(0)
    x = np.sort(10 * rng.random(50))
    yerr = 0.1 + 0.5 * rng.random(50)
    y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)
    return x, y, yerr
