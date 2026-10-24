import itertools

import numpy as np
import pytest
from scipy import stats

from telescopic._random import MAIN, root_seed
from telescopic.engine import run
from telescopic.estimators import unbiased_covariance
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning

TARGET = Target(lambda x: -0.5 * np.sum(x**2, axis=1), vectorize=True)
SETTINGS = {"k": 2, "m": 12, "lag": 2}


def gaussian_run(indices, seed=1, **settings):
    return run(
        TARGET,
        RandomWalk(np.eye(2)),
        Normal([2.0, -1.0], [1.0, 1.0]),
        indices,
        root=root_seed(seed),
        namespace=MAIN,
        **settings,
    )


def gaussian_result(n=40, **options):
    settings = SETTINGS | options
    return Result(gaussian_run(range(n), **settings), names=["a", "b"], **settings)


RESULT = gaussian_result()


def test_an_estimate_averages_the_replicates_with_a_normal_interval():
    estimate = RESULT.estimate(lambda x: x**2, level=0.9)
    values = RESULT.measure.per_replicate(lambda x: x**2)
    error = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    np.testing.assert_array_equal(estimate.value, values.mean(axis=0))
    np.testing.assert_allclose(estimate.standard_error, error, rtol=1e-12)
    np.testing.assert_allclose(
        estimate.interval[1] - estimate.value, stats.norm.ppf(0.95) * error, rtol=1e-12
    )


def test_the_covariance_combines_first_and_second_moments_coordinate_by_coordinate():
    first = RESULT.measure.per_replicate(lambda x: x)
    second = np.array(
        [
            [
                RESULT.measure.per_replicate(lambda x, i=i, j=j: x[:, i] * x[:, j]).mean()
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    covariance = RESULT.cov()
    np.testing.assert_allclose(covariance, unbiased_covariance(first, second), atol=1e-12)
    np.testing.assert_array_equal(covariance, covariance.T)
    np.testing.assert_allclose(RESULT.std(), np.sqrt(np.maximum(0, np.diag(covariance))))


def test_without_bias_corrections_quantiles_are_those_of_the_pooled_draws():
    # With k past every meeting, all weights are 1 / |S|: an ordinary empirical measure.
    # The levels avoid multiples of 1 / N, where rounding of the cumulative weights
    # decides between two neighbouring atoms.
    result = gaussian_result(k=40, m=45, lag=1, max_iter=None)
    assert not result.measure.weights.min() < 0
    levels = [0.137, 0.503, 0.911]
    for dim in range(2):
        expected = np.quantile(result.measure.atoms[:, dim], levels, method="inverted_cdf")
        np.testing.assert_array_equal(result.quantile(levels)[:, dim], expected)


def test_histogram_bins_average_each_replicates_signed_mass_in_half_open_bins():
    masses, errors, edges = RESULT.histogram(0, bins=7, range=(-2.0, 3.0))
    atoms, weights, owner = (
        RESULT.measure.atoms[:, 0],
        RESULT.measure.weights,
        RESULT.measure.replicate,
    )
    per_replicate = np.array(
        [
            [
                weights[(owner == r) & (atoms >= a) & (atoms < b)].sum()
                for a, b in itertools.pairwise(edges)
            ]
            for r in range(RESULT.n_replicates)
        ]
    )
    np.testing.assert_allclose(masses, per_replicate.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(errors, per_replicate.std(axis=0, ddof=1) / np.sqrt(40), atol=1e-12)
    total, _, _ = RESULT.histogram(0, bins=7)
    assert abs(total.sum() - 1) < 1e-12


def test_censored_pairs_are_listed_as_infinite_and_left_out_of_the_estimates():
    result = gaussian_result(n=30, k=3, m=3, lag=1, max_iter=4)
    censored = np.isinf(result.meeting_times)
    assert 0 < result.n_censored == censored.sum() < 30
    assert result.n_replicates == 30 - result.n_censored
    assert result.measure.per_replicate(lambda x: x[:, 0]).shape == (result.n_replicates,)
    with pytest.warns(ConvergenceWarning) as record:  # censored pairs are also late
        result.check()
    assert any("reached max_iter = 4" in str(warning.message) for warning in record)
    with pytest.raises(RuntimeError, match="reached max_iter"):
        result.check(strict=True)


def test_the_summary_reports_every_parameter_and_the_run():
    summary = RESULT.summary()
    as_dict = summary.to_dict()
    assert set(as_dict["parameters"]) == {"a", "b"}
    assert as_dict["parameters"]["a"]["mean"] == RESULT.mean().value[0]
    assert as_dict["n_replicates"] == 40
    assert "meeting times: median" in repr(summary)


def test_the_summary_converts_to_pandas():
    pytest.importorskip("pandas")
    assert list(RESULT.summary().to_pandas().index) == ["a", "b"]


def test_an_interrupted_run_says_its_estimates_are_not_exactly_unbiased():
    result = Result(RESULT._replicates, interrupted=True, **SETTINGS)
    assert "not exactly unbiased" in repr(result.summary())


def test_names_must_match_the_dimension():
    with pytest.raises(ValueError, match="names must have 2 entries"):
        Result(RESULT._replicates, names=["a"], **SETTINGS)


def test_merging_two_runs_equals_one_run_over_both_sets_of_replicates():
    def part(indices, seed=4):
        return Result(gaussian_run(indices, seed, **SETTINGS), seed=root_seed(seed), **SETTINGS)

    merged, whole = part(range(5)).merge(part(range(5, 12))), part(range(12))
    np.testing.assert_array_equal(merged.meeting_times, whole.meeting_times)
    np.testing.assert_array_equal(merged.measure.atoms, whole.measure.atoms)
    np.testing.assert_array_equal(merged.measure.replicate, whole.measure.replicate)
    with pytest.raises(ValueError, match="different lag"):
        part(range(3)).merge(
            Result(part(range(3, 5))._replicates, k=2, m=12, lag=1, seed=root_seed(4))
        )
    with pytest.raises(ValueError, match="different seeds"):
        part(range(3)).merge(part(range(3, 5), seed=5))


def test_results_from_equal_array_seeds_merge():
    def part(indices):
        seed = np.random.SeedSequence(np.array([1, 2, 3, 4]))
        return Result(gaussian_run(indices, seed, **SETTINGS), seed=seed, **SETTINGS)

    assert part(range(3)).merge(part(range(3, 5))).n_replicates == 5


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda r: r.histogram(5), "dim must be in"),
        (lambda r: r.histogram(0, bins=0), "bins must be at least 1"),
        (lambda r: r.tv_bound([-1]), "t must be non-negative"),
        (lambda r: r.w1_bound([-1]), "t must be non-negative"),
    ],
)
def test_invalid_arguments_of_the_result_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call(RESULT)


def test_the_default_bound_times_end_where_the_bound_vanishes():
    bound = RESULT.tv_bound()
    assert len(bound) == int(RESULT.meeting_times.max()) - RESULT.lag
    assert bound[-1] > 0
