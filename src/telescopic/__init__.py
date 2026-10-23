"""Unbiased Markov chain Monte Carlo with coupled chains and telescoping sums."""

from telescopic import initial, kernels
from telescopic.measure import SignedMeasure
from telescopic.results import Estimate, Result, Summary
from telescopic.sampler import CoupledSampler, sample
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning, InputWarning, TelescopicWarning

__version__ = "0.1.0.dev0"

__all__ = [
    "ConvergenceWarning",
    "CoupledSampler",
    "Estimate",
    "InputWarning",
    "Result",
    "SignedMeasure",
    "Summary",
    "Target",
    "TelescopicWarning",
    "__version__",
    "initial",
    "kernels",
    "sample",
]
