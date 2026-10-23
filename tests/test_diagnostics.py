import numpy as np
import pytest

from telescopic._random import root_seed
from telescopic.diagnostics import hill, issues, tv_bound, variance_share, w1_bound
from telescopic.engine import Replicates, Settings, run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.tuning import TuningReport


def synthetic(meeting_times, values, *, k=0, lag=1, max_iter=None, tuning=None):
    # One atom of weight 1 per replicate that met: its estimate is its value. A meeting
    # time of -1 marks a censored pair.
    tau = np.asarray(meeting_times, dtype=np.float64)
    tau[tau < 0] = np.inf
    met = np.flatnonzero(np.isfinite(tau))
    values = np.asarray(values, dtype=np.float64)
    return Result(
        Replicates(
            seed=root_seed(0),
            settings=Settings(k, k, lag, max_iter=max_iter),
            index=np.arange(len(tau)),
            meeting_time=tau,
            cost=np.ones(len(tau), dtype=np.int64),
            atoms=values[met].reshape(len(met), -1),
            weights=np.ones(len(met)),
            owner=met,
            distance_owner=np.zeros(0, dtype=np.int64),
            distance_time=np.zeros(0, dtype=np.int64),
            distance=np.zeros(0),
            zero_density=np.zeros(0, dtype=np.int64),
            requested=np.arange(len(tau)),
        ),
        tuning=tuning,
    )


def test_the_tv_bound_sums_the_survival_function_at_lagged_times():
    rng = np.random.default_rng(1)
    tau = rng.geometric(0.05, size=300) + 2
    for lag in (1, 3):
        t = np.arange(0, 80)
        expected = [sum(np.mean(tau > s + j * lag) for j in range(1, 200)) for s in t]
        np.testing.assert_allclose(tv_bound(tau, lag, t), np.minimum(1, expected), atol=1e-12)


def test_the_w1_bound_sums_the_distances_at_lagged_times():
    rng = np.random.default_rng(2)
    time, distance, lag, n = rng.integers(3, 40, size=200), rng.random(200), 3, 25
    t = np.arange(45)
    expected = [distance[(time > s) & ((time - s) % lag == 0)].sum() / n for s in t]
    np.testing.assert_allclose(w1_bound(time, distance, n, lag, t), expected, atol=1e-12)


@pytest.mark.parametrize("alpha", [1.5, 3.0])
def test_hill_recovers_a_pareto_tail_index(alpha):
    n = 10_000
    values = np.random.default_rng(3).pareto(alpha, size=n) + 1
    assert abs(hill(values, n // 10) - alpha) < 4 * alpha / np.sqrt(n // 10)


def test_the_tv_bound_refuses_censored_pairs_given_as_infinite():
    with pytest.raises(ValueError, match="censored pair as its cap"):
        tv_bound([3.0, np.inf], 1, [0])


def test_variance_share_is_zero_without_variance():
    np.testing.assert_array_equal(variance_share(np.ones((5, 2))), [0.0, 0.0])


@pytest.mark.parametrize(
    ("meeting_times", "values", "options", "message"),
    [
        ([3, -1, 4], [0.0, 0.0, 1.0], {"max_iter": 50}, "reached max_iter = 50"),
        ([2] * 70 + [9] * 30, np.arange(100.0), {"k": 5}, "30% of the pairs met after"),
        ([2] * 100, [0.0] * 99 + [100.0], {}, "one replicate holds 99%"),
        (
            np.random.default_rng(4).pareto(1.0, size=500).astype(int) + 2,
            np.random.default_rng(5).standard_normal(500),
            {},
            "look heavy-tailed",
        ),
        ([2, 20_000], [0.0, 1.0], {}, "still unmet after 10000"),
    ],
    ids=["censored", "late", "variance share", "tail index", "slow"],
)
def test_each_warning_fires_on_a_result_built_to_trigger_it(
    meeting_times, values, options, message
):
    found = issues(synthetic(meeting_times, values, **options))
    assert any(message in issue.message for issue in found), found


def test_a_pair_censored_after_the_slow_threshold_is_slow():
    found = issues(synthetic([2] * 9 + [-1], np.zeros(9), max_iter=20_000))
    assert {"censored", "slow"} <= {issue.kind for issue in found}


def test_changing_the_diagnostics_returned_leaves_the_results_own():
    result = synthetic([2] * 50, np.zeros(50), k=5)
    result.diagnostics()["late_fraction"] = 1.0
    assert result.diagnostics()["late_fraction"] == 0


def test_the_variance_warning_needs_one_replicate_to_hold_more_than_half_the_variance():
    # Shares of 0.5 and 0.58 in the variance of the mean or the second moment.
    assert issues(synthetic([2] * 100, [0.0] * 98 + [10.0, -10.0], k=5)) == []
    found = issues(synthetic([2] * 100, [0.0] * 98 + [11.0, -10.0], k=5))
    assert [issue.kind for issue in found] == ["variance"]


def test_tuning_chains_disagree_when_their_means_spread_wider_than_the_chains():
    def result(rhat):
        report = TuningReport(np.zeros(2), np.eye(2), np.full(4, 0.23), rhat, 100)
        return synthetic([2, 3], [0.0, 1.0], k=5, tuning=report)

    assert issues(result(np.array([1.0, 1.41]))) == []
    assert [issue.kind for issue in issues(result(np.array([1.0, 1.42])))] == ["tuning"]


def test_no_warning_fires_on_a_clean_gaussian_run():
    replicates = run(
        Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True),
        RandomWalk(np.eye(2)),
        Normal([1.0, 1.0], [1.0, 1.0]),
        range(100),
        rng=6,
        k=30,
        m=300,
        lag=30,
    )
    assert issues(Result(replicates)) == []


def test_the_variance_share_names_the_moment_and_coordinate_that_hold_it():
    # x1 is +-1 but for one replicate at 3, whose square, the only one apart from the 99
    # others, holds the share 1 - 1/R of their variance; it holds less of the mean's.
    rng = np.random.default_rng(7)
    normal = rng.standard_normal((100, 2))
    values = np.column_stack([normal[:, 0], np.r_[rng.choice([-1, 1], 99), 3], normal[:, 1]])
    found = synthetic([2] * 100, values, k=5).diagnostics()
    assert found["variance_share_of"] == "the second moment of x1"
    assert found["variance_share"] == pytest.approx(0.99)


def test_the_tail_index_needs_500_replicates():
    tau = np.random.default_rng(8).geometric(0.1, size=500) + 1
    assert np.isnan(synthetic(tau[:499], np.zeros(499)).diagnostics()["tail_index"])
    expected = hill(tau, 50)  # tau - L + 1 with L = 1
    assert synthetic(tau, np.zeros(500)).diagnostics()["tail_index"] == expected


def test_without_late_pairs_the_upper_bound_of_the_late_fraction_is_the_rule_of_three():
    found = synthetic([2] * 50, np.zeros(50), k=5).diagnostics()
    assert found["late_fraction"] == 0
    assert abs(found["late_fraction_upper"] - (1 - 0.05 ** (1 / 50))) < 1e-12


def test_hill_leaves_a_tail_unresolved_by_ties_undefined():
    # 92% ones and 8% twos: the threshold, the 51st largest value, is the smallest value.
    assert np.isnan(hill([1.0] * 460 + [2.0] * 40, 50))


def test_variance_share_ignores_spreads_of_rounding():
    assert variance_share(np.r_[np.ones(99), 1 + 2**-52]) == 0


@pytest.mark.parametrize(
    ("meeting_times", "lag", "t", "message"),
    [([], 1, [0], "non-empty"), ([3, 4], 0, [0], "lag must be"), ([3, 4], 1, [2.9], "integers")],
)
def test_the_tv_bound_refuses_invalid_arguments(meeting_times, lag, t, message):
    with pytest.raises(ValueError, match=message):
        tv_bound(meeting_times, lag, t)


def test_the_tv_bound_at_k_of_the_diagnostics_is_the_bounds():
    rng = np.random.default_rng(9)
    result = synthetic(rng.geometric(0.2, size=200) + 3, rng.standard_normal(200), k=4, lag=3)
    assert result.diagnostics()["tv_bound_at_k"] == pytest.approx(
        result.tv_bound([4])[0], rel=1e-12
    )


def test_hill_refuses_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        hill([0.0, 1.0, 2.0], 1)
