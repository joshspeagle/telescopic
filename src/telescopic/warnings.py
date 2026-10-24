"""Warning classes of telescopic."""


class TelescopicWarning(UserWarning):
    """Base class of the warnings that telescopic emits."""


class ConvergenceWarning(TelescopicWarning):
    """A result may be unreliable.

    Raised for pairs that did not meet, heavy-tailed meeting times or estimates,
    a large share of replicates with a bias correction, and disagreeing pilots.
    """


class InputWarning(TelescopicWarning):
    """An input is legal but questionable, such as an initial draw with zero density."""
