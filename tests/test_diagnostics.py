import numpy as np
import pytest

from telescopic._random import MAIN, root_seed
from telescopic.diagnostics import hill, issues, tv_bound, variance_share, w1_bound
from telescopic.engine import Replicates, run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target


def synthetic(meeting_times, values, *, k=0, lag=1, max_iter=None):
    # One atom of weight 1 per replicate that met: its estimate is its value. A meeting
    # time of -1 marks a censored pair.
    tau = np.asarray(meeting_times, dtype=np.float64)
    tau[tau < 0] = np.inf
    met = np.flatnonzero(np.isfinite(tau))
    values = np.asarray(values, dtype=np.float64)
    return Result(
        Replicates(
            index=np.arange(len(tau)),
            meeting_time=tau,
            cost=np.ones(len(tau), dtype=np.int64),
            atoms=values[met, None],
            weights=np.ones(len(met)),
            owner=met,
            distance_owner=np.zeros(0, dtype=np.int64),
            distance_time=np.zeros(0, dtype=np.int64),
            distance=np.zeros(0),
            zero_density=np.zeros(0, dtype=np.int64),
        ),
        k=k,
        m=k,
        lag=lag,
        max_iter=max_iter,
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
    assert any(message in issue for issue in found), found


def test_no_warning_fires_on_a_clean_gaussian_run():
    replicates = run(
        Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True),
        RandomWalk(np.eye(2)),
        Normal([1.0, 1.0], [1.0, 1.0]),
        range(100),
        root=root_seed(6),
        namespace=MAIN,
        k=30,
        m=300,
        lag=30,
    )
    assert issues(Result(replicates, k=30, m=300, lag=30)) == []


def test_without_late_pairs_the_upper_bound_of_the_late_fraction_is_the_rule_of_three():
    found = synthetic([2] * 50, np.zeros(50), k=5).diagnostics()
    assert found["late_fraction"] == 0
    assert abs(found["late_fraction_upper"] - (1 - 0.05 ** (1 / 50))) < 1e-12


def test_hill_refuses_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        hill([0.0, 1.0, 2.0], 1)
