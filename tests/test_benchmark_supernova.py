"""Verify the deterministic supernova target and numerical reference."""

import numpy as np
import pytest
from scipy import integrate, special

from benchmarks import supernova


@pytest.mark.slow
def test_synthetic_data_and_reference_are_reproducible_under_refinement() -> None:
    for original, repeated in zip(supernova._data(), supernova._data(), strict=True):
        np.testing.assert_array_equal(original, repeated)
    case = supernova.supernova(4, order=128, distance_order=24)
    refined = supernova.supernova(4, order=256, distance_order=48)
    np.testing.assert_allclose(case.matched.mean, refined.matched.mean, atol=5e-7, rtol=0)
    np.testing.assert_allclose(case.matched.cov, refined.matched.cov, atol=5e-7, rtol=0)
    np.testing.assert_allclose(case.wide.cov, 100 * case.matched.cov)
    for estimand, other in zip(case.estimands, refined.estimands, strict=True):
        np.testing.assert_allclose(
            [estimand.mean, estimand.variance], [other.mean, other.variance], atol=1e-7, rtol=0
        )


def test_distance_agrees_with_independent_adaptive_line_of_sight_integral() -> None:
    redshifts = np.array([0.02, 0.3, 1.1])
    omega, w = 0.32, -0.89

    def inverse_expansion(z: float) -> float:
        return float(1 / np.sqrt(omega * (1 + z) ** 3 + (1 - omega) * (1 + z) ** (3 * (1 + w))))

    expected = np.array(
        [
            5 * np.log10((1 + redshift) * integrate.quad(inverse_expansion, 0, redshift)[0])
            for redshift in redshifts
        ]
    )
    actual, _, _ = supernova.Distance(redshifts, 24).evaluate(
        np.asarray(special.logit(omega)), np.asarray(w)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)


def test_supernova_score_matches_finite_differences() -> None:
    case = supernova.supernova(4)
    points = np.array([[0.2, -0.4, 0.1, 0.5], [-1.5, 0.4, -0.3, -0.7]])
    step = 1e-5
    offsets = np.eye(4) * step
    central = (
        case.log_prob(points[:, None] + offsets) - case.log_prob(points[:, None] - offsets)
    ) / (2 * step)
    np.testing.assert_allclose(case.grad(points), central, atol=1e-7, rtol=1e-6)


def test_finite_tail_points_have_finite_likelihood_and_gradient() -> None:
    case = supernova.supernova(4)
    points = np.array(
        [[50.0, 800.0, 0.0, 0.0], [0.0, 0.0, 0.0, 800.0], [-900.0, -800.0, 0.0, 0.0]]
    )
    assert np.isfinite(case.log_prob(points)).all()
    assert np.isfinite(case.grad(points)).all()
