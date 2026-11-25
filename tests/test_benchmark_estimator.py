"""Check fixed-replicate benchmark summaries and censoring."""

import numpy as np
import pytest

from benchmarks import estimator, targets
from telescopic.engine import Settings


def test_complete_intervals_use_all_replicates_for_cost_and_standard_error() -> None:
    estimand = targets.rotated_gaussian(2).estimands[0]
    records = [
        estimator.RunRecord(
            12,
            0,
            ["variance"],
            ["variance"],
            {
                estimand.name: estimator.IntervalRecord([-1.0, 1.0], True),
            },
        ),
        estimator.RunRecord(
            20,
            0,
            ["tail"],
            ["tail"],
            {
                estimand.name: estimator.IntervalRecord([2.0, 2.0], False),
            },
        ),
    ]
    summary = estimator.summarize(records, estimand, asymptotic_variance=4.0)
    assert summary["coverage"] == 0.5
    assert summary["coverage_standard_error"] == pytest.approx(np.sqrt(0.125))
    assert summary["pooled_mean"] == 1.0
    assert summary["pooled_standard_error"] == pytest.approx(np.sqrt(0.5))
    assert summary["pooled_z"] == pytest.approx(np.sqrt(2))
    assert summary["inefficiency_over_target_variance"] == 16.0
    assert summary["inefficiency_over_asymptotic_variance"] == 4.0
    assert summary["warning_eligible_runs"] == {"variance": 1, "tail": 1}
    assert summary["warning_counts_by_interval_outcome"] == {
        "variance": {"covered": 1, "missed": 0, "unavailable": 0},
        "tail": {"covered": 0, "missed": 1, "unavailable": 0},
    }


def test_censored_run_retains_cost_and_suppresses_aggregate_metrics() -> None:
    estimand = targets.rotated_gaussian(2).estimands[0]
    records = [
        estimator.RunRecord(
            12,
            0,
            [],
            [],
            {
                estimand.name: estimator.IntervalRecord([-1.0, 1.0], True),
            },
        ),
        estimator.RunRecord(20, 1, ["variance", "censored"], ["variance"], {}),
    ]
    summary = estimator.summarize(records, estimand)
    assert summary["runs"] == 2
    assert summary["available_intervals"] == summary["covered_intervals"] == 1
    assert summary["sampling_evaluations"] == 32
    assert summary["coverage"] is None
    assert summary["pooled_z"] is None
    assert summary["inefficiency_over_target_variance"] is None
    assert summary["warning_counts_by_interval_outcome"]["variance"]["unavailable"] == 1


def test_frozen_random_walk_runner_scores_deterministic_independent_runs() -> None:
    case = targets.rotated_gaussian(2)
    settings = Settings(20, 100, 20, 1, 500, True)
    kwargs = {"start": "matched", "proposal": "oracle", "runs": 3, "replicates": 15, "seed": 702}
    records = estimator.random_walk_runs(case, settings, **kwargs)
    assert records == estimator.random_walk_runs(case, settings, **kwargs)
    assert all(record.evaluations > 0 and record.censored == 0 for record in records)
    assert all(not record.assessed_warnings for record in records)
    for estimand in case.estimands:
        values = [record.estimates[estimand.name].replicates for record in records]
        assert all(len(row) == 15 for row in values)
        assert len({tuple(row) for row in values}) == 3


def test_frozen_random_walk_runner_keeps_a_censored_run_without_intervals() -> None:
    case = targets.rotated_gaussian(10)
    settings = Settings(0, 1, 1, 1, 2, True)
    (record,) = estimator.random_walk_runs(
        case, settings, start="wide", proposal="identity", runs=1, replicates=10, seed=702
    )
    assert record.censored == 10
    assert record.evaluations > 0
    assert "censored" in record.warnings
    assert not record.estimates


def test_stationary_batch_means_counts_reference_cost_separately() -> None:
    case = targets.rotated_gaussian(2)
    _, kernel = estimator.research.random_walk_setup(case, "matched", "oracle")
    result = estimator.batch_means_reference(case, kernel, batches=4, batch_size=25, seed=703)
    variances, cost = result
    assert result == estimator.batch_means_reference(
        case, kernel, batches=4, batch_size=25, seed=703
    )
    assert cost == 101
    assert variances.keys() == {estimand.name for estimand in case.estimands}
    assert all(variance > 0 for variance in variances.values())
