"""Deterministic logistic regression designs for the research benchmark."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import special

from benchmarks import targets
from benchmarks.targets import Array

DATA_SEED = 1
PRIOR_VARIANCE = 25.0
ESTIMAND_INDICES = (0, 5)


@dataclass(frozen=True)
class LogisticDesign:
    """Synthetic binary responses and their design matrix, including an intercept."""

    matrix: Array
    outcomes: Array

    def log_prob(self, points: Array) -> Array:
        """Return log posterior densities at unitless points of shape (..., d)."""
        linear = points @ self.matrix.T
        margin = np.where(self.outcomes, -linear, linear)
        return -np.logaddexp(0.0, margin).sum(axis=-1) - (points**2).sum(axis=-1) / (
            2 * PRIOR_VARIANCE
        )

    def grad(self, points: Array) -> Array:
        """Return the log posterior score of the same shape as points."""
        residual = self.outcomes - special.expit(points @ self.matrix.T)
        return residual @ self.matrix - points / PRIOR_VARIANCE


def design(dimension: int) -> LogisticDesign:
    """Reproduce the six-parameter experiment and extend its signal to 300 dimensions."""
    if dimension not in (6, 300):
        raise ValueError("logistic dimension must be 6 or 300")
    size = 200 if dimension == 6 else 1500
    rng = np.random.default_rng(DATA_SEED)
    matrix = np.column_stack([np.ones(size), rng.standard_normal((size, dimension - 1))])
    signal = np.zeros(dimension)
    signal[:6] = [-0.5, 1, -1, 0.5, 0, 2]
    outcomes = (rng.random(size) < special.expit(matrix @ signal)).astype(np.float64)
    return LogisticDesign(matrix, outcomes)


def design_digest(data: LogisticDesign) -> str:
    """Hash the float64 design and responses in little-endian row order."""
    encoded = np.column_stack([data.matrix, data.outcomes]).astype("<f8")
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def logistic(dimension: int) -> targets.BenchmarkCase:
    """Build a logistic benchmark from independently measured posterior moments."""
    data = design(dimension)
    path = Path(__file__).with_name("results") / f"logistic-{dimension}.json"
    with path.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    if (
        not reference["converged"]
        or reference["dimension"] != dimension
        or reference["prior_variance"] != PRIOR_VARIANCE
        or reference["dataset_sha256"] != design_digest(data)
        or tuple(reference["indices"]) != ESTIMAND_INDICES
    ):
        raise ValueError(
            "logistic reference must match the design and estimands and pass diagnostics"
        )
    intercept, last_signal = ESTIMAND_INDICES
    estimands = (
        targets.Estimand(
            "intercept_mean",
            lambda points: points[..., intercept],
            reference["mean"][0],
            reference["variance"][0],
            reference["mean_standard_error"][0],
        ),
        targets.Estimand(
            "last_signal_mean",
            lambda points: points[..., last_signal],
            reference["mean"][1],
            reference["variance"][1],
            reference["mean_standard_error"][1],
        ),
    )
    return targets._case(
        "logistic",
        data.log_prob,
        data.grad,
        None,
        np.asarray(reference["start_mean"]),
        np.diag(reference["start_variance"]),
        estimands,
        covariance_source="independent_gibbs_marginal_variances",
    )
