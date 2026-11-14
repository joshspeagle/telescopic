import numpy as np
import pytest

from references import gaussian_log_prob, synthetic
from telescopic.diagnostics import (
    _lagged_tail,
    hill,
    issues,
    slow_pair_threshold,
    tuning_disagreement,
    tv_bound,
    variance_share,
    w1_bound,
)
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.tuning import TuningReport, adaptive_metropolis


def test_the_tv_bound_sums_the_survival_function_at_lagged_times():
    rng = np.random.default_rng(1)
    tau = rng.geometric(0.05, size=300) + 2
    for lag in (1, 3):
        t = np.arange(0, 80)
        expected = [sum(np.mean(tau > s + j * lag) for j in range(1, 200)) for s in t]
        np.testing.assert_allclose(tv_bound(tau, lag, t), np.minimum(1, expected), atol=1e-12)


@pytest.mark.parametrize("lag", [1, 3])
def test_the_tv_bound_from_the_exact_meeting_law_exceeds_the_exact_tv(lag):
    # A 3-state chain from state 0, coupled by the maximal coupling of discrete_maximal:
    # common moves from min(P_i, P_j), else independent moves from the residuals. The
    # pair (X_s, Y_{s-L}) starts from the law of X_L times that of Y_0 and evolves by the
    # coupled matrix, absorbed on the diagonal, which gives P(tau > s) exactly.
    P = np.array([[0.5, 0.4, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
    coupled = np.zeros((3, 3, 3, 3))
    for i, j in np.ndindex(3, 3):
        common = np.minimum(P[i], P[j])
        coupled[i, j] = np.diag(common)
        if i != j:
            coupled[i, j] += np.outer(P[i] - common, P[j] - common) / (1 - common.sum())
    coupled = coupled.reshape(9, 9)
    start = np.eye(3)[0]
    apart = ~np.eye(3, dtype=bool).ravel()
    pair = np.outer(start @ np.linalg.matrix_power(P, lag), start).ravel()
    survival = np.ones(lag + 300)
    for s in range(lag, len(survival)):
        survival[s] = pair[apart].sum()
        pair = pair @ coupled
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    pi = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    pi /= pi.sum()
    t = np.arange(21)
    tv = [0.5 * np.abs(start @ np.linalg.matrix_power(P, u) - pi).sum() for u in t]
    # From t = 16 the two agree within 3e-16, rounding; the bound exceeds the distance by
    # 4e-14 at t = 13, so the tolerance leaves t <= 13 checked.
    assert np.all(_lagged_tail(survival, lag, t) >= np.array(tv) - 1e-14)


def test_the_w1_bound_sums_the_distances_at_lagged_times():
    rng = np.random.default_rng(2)
    time, distance, lag, n = rng.integers(3, 40, size=200), rng.random(200), 3, 25
    t = np.arange(45)
    expected = [distance[(time > s) & ((time - s) % lag == 0)].sum() / n for s in t]
    np.testing.assert_allclose(w1_bound(time, distance, n, lag, t), expected, atol=1e-12)
    with pytest.raises(ValueError, match="n_replicates must be at least 1"):
        w1_bound(time, distance, 0, lag, t)


@pytest.mark.parametrize("alpha", [1.5, 3.0])
def test_hill_recovers_a_pareto_tail_index(alpha):
    n = 10_000
    values = np.random.default_rng(3).pareto(alpha, size=n) + 1
    assert abs(hill(values, n // 10) - alpha) < 4 * alpha / np.sqrt(n // 10)


def test_hill_measures_the_tail_from_the_largest_value_below_it():
    # The 2 largest of [1, 2, 4, 8] over the next, 2: log 4 and log 2. In uint8, -n_tail
    # would wrap.
    for n_tail in (2, np.uint8(2)):
        assert hill([1.0, 2.0, 4.0, 8.0], n_tail) == pytest.approx(2 / (3 * np.log(2)), rel=1e-12)


def test_the_tv_bound_refuses_censored_pairs_given_as_infinite():
    with pytest.raises(ValueError, match="censored pair as its cap plus 1"):
        tv_bound([3.0, np.inf], 1, [0])


@pytest.mark.parametrize(
    ("meeting_times", "values", "options", "message"),
    [
        ([3, -1, 4], [0.0, 0.0, 1.0], {"max_iter": 50}, "reached max_iter = 50"),
        ([2] * 70 + [9] * 30, np.arange(100.0), {"k": 5}, "30% of the pairs met after"),
        ([2] * 100, [0.0] * 99 + [100.0], {}, "replicate 99 holds 99%"),
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


@pytest.mark.parametrize(
    ("meeting_times", "values", "options", "remedy"),
    [
        ([2] * 70 + [9] * 30, np.arange(100.0), {"k": 5}, "; run a new sampler with a larger k"),
        (
            [2] * 100,
            [0.0] * 99 + [100.0],
            {},
            "; check the initial distribution, and look for a mode that the other",
        ),
        (
            np.random.default_rng(4).pareto(1.0, size=500).astype(int) + 2,
            np.random.default_rng(5).standard_normal(500),
            {},
            "; look at the survival function of plot_meeting_times, and at the kernel",
        ),
        ([2, 20_000], [0.0, 1.0], {}, "; look at the kernel and the initial distribution"),
    ],
    ids=["late", "variance share", "tail index", "slow"],
)
def test_each_warning_ends_with_its_remedy(meeting_times, values, options, remedy):
    found = issues(synthetic(meeting_times, values, **options))
    assert any(remedy in issue.message for issue in found), found


@pytest.mark.parametrize("met", [9, 0], ids=["some met", "none met"])
def test_a_pair_censored_at_the_slow_threshold_is_slow(met):
    # Its chains were compared at the cap, 10^4, so it was unmet there. Where no pair
    # met, the largest meeting time was nan, and max(nan, max_iter) nan.
    found = issues(synthetic([2] * met + [-1] * 3, np.zeros(met), max_iter=10_000))
    assert {"censored", "slow"} <= {issue.kind for issue in found}


def test_the_censored_warning_does_not_advise_no_cap_after_zero_density_starts():
    result = synthetic([3, -1, 4], [0.0, 0.0, 1.0], max_iter=50)
    result._replicates.zero_density = np.array([1])
    (message,) = [issue.message for issue in issues(result) if issue.kind == "censored"]
    assert "or none" not in message
    assert "1 of them started where the target density is zero" in message


def test_the_censored_warning_advises_no_cap_when_the_zero_density_starts_met():
    result = synthetic([3, -1, 4], [0.0, 0.0, 1.0], max_iter=50)
    result._replicates.zero_density = np.array([0, 2])
    (message,) = [issue.message for issue in issues(result) if issue.kind == "censored"]
    assert message.endswith("or none")


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


@pytest.mark.parametrize(
    ("rhat", "disagree"), [([np.nan, 1.0], False), ([np.nan, 3.0, np.nan], True)]
)
def test_the_tuning_warning_names_coordinates_whose_r_hat_is_nan(rhat, disagree):
    # A NaN, where all four chains sat at one value of a coordinate, made the largest R-hat
    # NaN, which passed the check and hid a disagreement in another coordinate.
    report = TuningReport(np.zeros(2), np.eye(2), np.full(4, 0.23), np.array(rhat), 100)
    message = tuning_disagreement(report)
    assert ("disagree (R-hat up to 3.00)" in message) == disagree
    assert ("several modes" in message) == disagree
    # Conditional: the initial distribution may already cover every mode.
    conditional = "; unless the initial distribution already covers every mode, use one that does"
    assert (conditional in message) == disagree
    nan = [i for i, value in enumerate(rhat) if np.isnan(value)]
    assert f"never moved in coordinates {nan} (R-hat NaN)" in message


def test_the_tuning_warning_names_a_coordinate_whose_steps_round_away():
    # At 2^70 a float64 is a multiple of 2^18, so steps of the target's scale, 1e3, round
    # away: every chain stays at 2^70, whose mean and variance are exact, and R-hat is NaN.
    start = np.column_stack([np.linspace(-2.0, 2.0, 100), np.full(100, 2.0**70)])

    def log_prob(x):
        return -0.5 * (x[..., 0] ** 2 + ((x[..., 1] - 2.0**70) / 1e3) ** 2)

    pilot = adaptive_metropolis(Target(log_prob, vectorize=True), start, 1, n_steps=200)
    message = tuning_disagreement(pilot.report())
    assert message.startswith("the tuning chains never moved in coordinates [1]")


def test_no_warning_fires_on_a_clean_gaussian_run():
    replicates = run(
        Target(gaussian_log_prob, vectorize=True),
        RandomWalk(np.eye(2)),
        Normal([1.0, 1.0], [1.0, 1.0]),
        range(100),
        rng=6,
        k=30,
        m=300,
        lag=30,
    )
    assert issues(Result(replicates)) == []


@pytest.mark.parametrize("column", [0, 1])
def test_the_variance_share_names_the_moment_and_coordinate_that_hold_it(column):
    # The coordinate is 1 or -1 about its mean 0 but for one replicate at 3, whose squared
    # deviation, the only one apart from the 99 others, holds the share 1 - 1/R of their
    # variance; it holds less of the mean's. Far from 0 the share is the same.
    normal = np.random.default_rng(7).standard_normal((100, 2))
    outlier = np.r_[np.repeat([1.0, -1.0], [48, 51]), 3.0]
    for offset in (0.0, 1000.0):
        values = np.insert(normal, column, outlier + offset, axis=1)
        found = synthetic([2] * 100, values, k=5).diagnostics()
        assert found["variance_share_of"] == f"the variance of x{column}"
        assert found["variance_share"] == pytest.approx(0.99)


def test_the_variance_share_names_the_replicate_that_holds_it():
    # Replicate 3 is the outlier; the censored pair 0 shifts the rows of the replicates
    # that met by one, so a row would read 2.
    values = np.r_[0.0, np.repeat([1.0, -1.0], [50, 49])]
    values = np.insert(values, 3, 30.0)
    result = synthetic([-1] + [2] * 100, values, k=5, max_iter=50)
    assert result.diagnostics()["variance_share_replicate"] == 3
    assert any(issue.message.startswith("replicate 3 holds") for issue in issues(result))


def test_the_tail_index_is_that_of_tau_minus_the_lag_from_500_replicates():
    # With L = 30, tau - L + 1 = tau' for meeting times tau' + 29.
    tau = np.random.default_rng(8).geometric(0.1, size=500) + 1
    found = synthetic(tau[:499] + 29, np.zeros(499), lag=30).diagnostics()["tail_index"]
    assert np.isnan(found)
    found = synthetic(tau + 29, np.zeros(500), lag=30).diagnostics()["tail_index"]
    assert found == hill(tau, 50)


def test_without_late_pairs_the_upper_bound_of_the_late_fraction_is_the_rule_of_three():
    # Pairs that meet at k + L need no bias correction, so they are not late.
    found = synthetic([6] * 50, np.zeros(50), k=5).diagnostics()
    assert found["late_fraction"] == 0
    assert abs(found["late_fraction_upper"] - (1 - 0.05 ** (1 / 50))) < 1e-12


def test_censored_pairs_count_as_late():
    found = synthetic([2] * 8 + [-1] * 2, np.zeros(8), k=5, max_iter=50).diagnostics()
    assert found["late_fraction"] == 0.2


def test_the_slow_pair_threshold_is_ten_times_the_larger_of_the_lag_and_m():
    assert slow_pair_threshold(1, 10) == 10_000
    assert slow_pair_threshold(1, 2_000) == slow_pair_threshold(2_000, 1) == 20_000
    with pytest.raises(TypeError, match="lag must be an integer"):
        slow_pair_threshold("1", 2)


def test_the_relative_cost_divides_the_mean_cost_by_the_averaged_times():
    # Costs max(tau, m) + tau - L: 13, 15, 23 and 39.
    found = synthetic([3, 5, 12, 20], np.zeros(4), k=2, m=11).diagnostics()
    assert (found["mean_cost"], found["relative_cost"]) == (22.5, 2.25)


def test_hill_is_undefined_where_ties_make_the_smallest_value_the_threshold():
    # 92% ones and 8% twos gave 1.80, and a threshold moved to 2 gave inf; a light tail
    # above 98% twos read as heavy (DECISIONS #57).
    assert np.isnan(hill([1.0] * 460 + [2.0] * 40, 50))
    assert np.isnan(hill(np.r_[np.full(490, 2.0), np.arange(3.0, 13.0)], 50))
    # Without ties, the smallest value is the threshold at n_tail = n - 1.
    assert hill([1.0, 2.0, 4.0, 8.0], 3) == pytest.approx(1 / (2 * np.log(2)), rel=1e-12)


def test_variance_share_ignores_spreads_of_rounding():
    assert variance_share(np.r_[np.ones(99), 1 + 2**-52]) == 0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        # NaN read as no variance, a share of 0.
        ([0.0, 0.0, np.nan, 100.0], "values must be finite"),
        (np.zeros((0, 2)), "R >= 1"),
        (np.zeros((2, 2, 2)), "shape"),
    ],
)
def test_variance_share_refuses_invalid_values(values, message):
    with pytest.raises(ValueError, match=message):
        variance_share(values)


@pytest.mark.parametrize(
    ("meeting_times", "lag", "t", "message"),
    [
        ([], 1, [0], "non-empty"),
        ([3, 4], 0, [0], "lag must be"),
        ([3, 4], 1, [2.9], "integers"),
        # Truncated to 2, silently.
        ([2.9, 3.0], 1, [1], "integers"),
        # A cast warning, then an IndexError.
        ([3, 4], 1, [np.inf], "integers"),
    ],
)
def test_the_tv_bound_refuses_invalid_arguments(meeting_times, lag, t, message):
    with pytest.raises(ValueError, match=message):
        tv_bound(meeting_times, lag, t)


def test_the_tv_bound_is_zero_long_after_every_meeting():
    # t = 1e20 overflowed the cast to int64 and raised IndexError.
    np.testing.assert_array_equal(tv_bound([3, 4], 1, [10, 1e20]), [0.0, 0.0])


@pytest.mark.parametrize(
    ("times", "distances", "message"),
    [
        ([-1, 2], [0.5, 0.5], "non-negative integers"),
        ([1.5, 2.5], [1.0, 1.0], "non-negative integers"),
        ([1, 2, 3], [0.5, 0.5], "one shape"),
        ([[1, 2]], [[1.0, 1.0]], "one shape"),
        ([1, 2], [np.nan, 1.0], "finite and non-negative"),
        ([1, 2], [-5.0, 1.0], "finite and non-negative"),
    ],
)
def test_the_w1_bound_refuses_invalid_distances(times, distances, message):
    with pytest.raises(ValueError, match=message):
        w1_bound(times, distances, 2, 1, [0])


def test_the_tv_bound_at_k_of_the_diagnostics_is_the_bounds():
    rng = np.random.default_rng(9)
    result = synthetic(rng.geometric(0.2, size=200) + 3, rng.standard_normal(200), k=4, lag=3)
    assert result.diagnostics()["tv_bound_at_k"] == pytest.approx(
        result.tv_bound([4])[0], rel=1e-12
    )


@pytest.mark.parametrize(
    ("values", "n_tail", "error", "message"),
    [
        ([0.0, 1.0, 2.0], 1, ValueError, "values must be positive and finite"),
        ([1.0, 2.0, np.inf], 1, ValueError, "values must be positive and finite"),
        ([1.0, 2.0, np.nan], 1, ValueError, "values must be positive and finite"),
        ([], 1, ValueError, "below the number of values, 0"),
        ([1.0, 2.0, 3.0], 1.0, TypeError, "n_tail must be an integer"),
        (np.ones((4, 3)), 1, ValueError, r"values must have shape \(n,\)"),
        (2.0, 1, ValueError, r"values must have shape \(n,\)"),
    ],
)
def test_hill_refuses_invalid_arguments(values, n_tail, error, message):
    with pytest.raises(error, match=message):
        hill(values, n_tail)


@pytest.mark.parametrize(("lag", "m", "name"), [(0, 5, "lag"), (1, -5, "m")])
def test_slow_pair_threshold_refuses_a_lag_below_1_or_a_negative_m(lag, m, name):
    # slow_pair_threshold(0, -5) returned 10000.
    with pytest.raises(ValueError, match=f"{name} must be at least"):
        slow_pair_threshold(lag, m)
