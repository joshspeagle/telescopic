"""Warning classes of telescopic, and the helper that emits them."""

from __future__ import annotations

import os
import sys
import warnings

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


class TelescopicWarning(UserWarning):
    """Base class of the warnings that telescopic emits."""


class ConvergenceWarning(TelescopicWarning):
    """A result may be unreliable.

    Raised for pairs that did not meet, heavy-tailed meeting times or estimates,
    a large share of replicates with a bias correction, and disagreeing pilots.
    """


class InputWarning(TelescopicWarning):
    """An input is legal but questionable, such as an initial draw with zero density."""


def warn(message: str, category: type[Warning]) -> None:
    """Emit a warning attributed to the innermost caller outside telescopic.

    Parameters
    ----------
    message : str
    category : Warning subclass
    """
    frame, level = sys._getframe(1), 2
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
