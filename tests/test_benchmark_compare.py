"""Check method selection and pointwise accounting for the shared pilot."""

from dataclasses import replace

import pytest

from benchmarks import compare, research, targets
from telescopic.kernels import HMC


def test_hmc_caps_density_and_gradient_callbacks_including_censored_work() -> None:
    case = targets.rotated_gaussian(2)
    density_rows = gradient_rows = 0

    def log_prob(points):
        nonlocal density_rows
        density_rows += len(points)
        return case.log_prob(points)

    def grad(points):
        nonlocal gradient_rows
        gradient_rows += len(points)
        return case.grad(points)

    counted = replace(case, log_prob=log_prob, grad=grad)
    kernel = HMC(cov=case.matched.cov, step_size=0.3, n_steps=3)
    records = research.meeting_pilot(counted, case.matched, kernel, pairs=8, budget=17, seed=702)
    assert sum(row.evaluations for row in records) == density_rows + gradient_rows
    assert sum(row.gradient_evaluations for row in records) == gradient_rows > 0
    assert all(row.evaluations <= 17 for row in records)
    assert any(row.meeting_cost is None for row in records)


def test_comparison_charges_shipped_hmc_tuning_and_preserves_the_random_walk() -> None:
    case = targets.rotated_gaussian(2)
    report = compare.compare_case(
        case,
        methods=["random_walk", "hmc"],
        start="matched",
        pairs=5,
        budget=500,
        tuning_seed=701,
        seed=702,
    )["methods"]
    assert report["random_walk"]["setup_density_evaluations"] == 0
    assert report["random_walk"]["sampling_gradient_evaluations"] == 0
    hmc = report["hmc"]
    assert hmc["setup_density_evaluations"] > 0
    assert hmc["setup_gradient_evaluations"] > 0
    assert hmc["sampling_gradient_evaluations"] > 0
    assert "jitter=0.5" in hmc["kernel"]
    assert hmc["total_evaluations"] == sum(
        hmc[key]
        for key in (
            "setup_density_evaluations",
            "setup_gradient_evaluations",
            "sampling_density_evaluations",
            "sampling_gradient_evaluations",
        )
    )


def test_comparison_refuses_tuning_seed_reuse() -> None:
    with pytest.raises(ValueError, match="seeds must differ"):
        compare.compare_case(
            targets.rotated_gaussian(2),
            methods=["hmc"],
            start="matched",
            pairs=1,
            budget=50,
            tuning_seed=701,
            seed=701,
        )


def test_optional_hmc_estimation_charges_tuning_once() -> None:
    from telescopic.engine import Settings

    case = targets.rotated_gaussian(2)
    report = compare.compare_case(
        case,
        methods=["hmc"],
        start="matched",
        pairs=4,
        budget=500,
        tuning_seed=701,
        seed=702,
        estimation=compare.EstimatorPlan(Settings(20, 100, 20, 1, 600, True), 3, 8, 703),
        reference=compare.ReferencePlan(4, 25, 704),
    )["methods"]["hmc"]
    summary = report["estimation"]["summary"]["slow_projection"]
    assert summary["available_intervals"] == 3
    assert report["estimation"]["reference_evaluations"] > 101
    assert summary["inefficiency_over_asymptotic_variance"] > 0
    assert summary["sampling_gradient_evaluations"] > 0
    assert summary["setup_evaluations"] == (
        report["setup_density_evaluations"] + report["setup_gradient_evaluations"]
    )
    assert report["total_evaluations"] == (
        summary["total_evaluations"] + report["summary"]["evaluations"]
    )
    assert report["total_evaluations"] == (
        summary["setup_evaluations"]
        + report["sampling_density_evaluations"]
        + report["sampling_gradient_evaluations"]
    )
    assert (
        summary["inefficiency_over_target_variance"]
        > summary["inefficiency_over_target_variance_without_setup"]
    )


def test_comparison_reports_all_callbacks_including_setup() -> None:
    case = targets.rotated_gaussian(2)
    density = gradient = 0

    def log_prob(points):
        nonlocal density
        density += len(points)
        return case.log_prob(points)

    def grad(points):
        nonlocal gradient
        gradient += len(points)
        return case.grad(points)

    report = compare.compare_case(
        replace(case, log_prob=log_prob, grad=grad),
        methods=["hmc"],
        start="matched",
        pairs=2,
        budget=500,
        tuning_seed=701,
        seed=702,
    )["methods"]["hmc"]
    assert density == report["setup_density_evaluations"] + report["sampling_density_evaluations"]
    assert (
        gradient == report["setup_gradient_evaluations"] + report["sampling_gradient_evaluations"]
    )


def test_additional_meeting_seeds_share_one_setup() -> None:
    case = targets.rotated_gaussian(2)
    common = {
        "methods": ["hmc"],
        "start": "matched",
        "pairs": 2,
        "budget": 500,
        "tuning_seed": 701,
        "seed": 702,
    }
    one = compare.compare_case(case, **common)["methods"]["hmc"]
    two = compare.compare_case(case, **common, additional_meeting_seeds=(705,))["methods"]["hmc"]
    assert two["setup_density_evaluations"] == one["setup_density_evaluations"]
    assert two["setup_gradient_evaluations"] == one["setup_gradient_evaluations"]
    assert two["records"][:2] == one["records"]
    assert two["summary"]["pairs"] == 4
    assert [row["seed"] for row in two["meeting_summary_by_seed"]] == [702, 705]


def test_primary_mode_reuses_the_reference_normalized_summary(monkeypatch) -> None:
    from benchmarks import estimator
    from telescopic.engine import Settings

    case = targets.mixture(2, separation=6.0, narrow_scale=1.0)
    record = estimator.RunRecord(
        100,
        0,
        [],
        [],
        {item.name: estimator.IntervalRecord([0.0, 1.0], True) for item in case.scored_estimands},
    )
    monkeypatch.setattr(estimator, "run_frozen", lambda *args, **kwargs: [record])
    monkeypatch.setattr(
        estimator,
        "batch_means_reference",
        lambda *args, **kwargs: ({"left_probability": 0.5}, 101),
    )
    report = compare.compare_case(
        case,
        methods=["random_walk"],
        start="matched",
        pairs=1,
        budget=500,
        tuning_seed=701,
        seed=702,
        estimation=compare.EstimatorPlan(Settings(20, 100, 20, 1, 600, True), 1, 2, 703),
        reference=compare.ReferencePlan(4, 25, 704),
    )["methods"]["random_walk"]["estimation"]
    assert report["summary"]["left_probability"]["inefficiency_over_asymptotic_variance"] > 0
    assert (
        report["summary"]["left_probability"] == report["mode_probabilities"]["left_probability"]
    )
    assert set(report["mode_probabilities"]) == {"left_probability", "right_probability"}


def test_identity_hmc_tunes_with_the_identity_covariance(monkeypatch) -> None:
    import numpy as np

    original = compare.HMC
    received = []

    def capture(**kwargs):
        received.append(kwargs["cov"])
        return original(**kwargs)

    monkeypatch.setattr(compare, "HMC", capture)
    case = targets.rotated_gaussian(2)
    result = compare.compare_case(
        case,
        methods=["hmc_identity"],
        start="matched",
        pairs=1,
        budget=500,
        tuning_seed=701,
        seed=702,
    )["methods"]["hmc_identity"]
    np.testing.assert_array_equal(received[0], np.eye(2))
    assert result["kernel_covariance_source"] == "identity"
    assert result["setup_gradient_evaluations"] > 0
    assert "jitter=0.5" in result["kernel"]
