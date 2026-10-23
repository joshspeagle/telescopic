import itertools
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from references import synthetic
from telescopic.diagnostics import w1_bound
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
        rng=seed,
        **settings,
    )


def gaussian_result(n=40, **options):
    settings = SETTINGS | options
    return Result(gaussian_run(range(n), **settings), names=["a", "b"])


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
    np.testing.assert_allclose(RESULT.var(), np.diag(covariance), rtol=1e-12)
    np.testing.assert_allclose(RESULT.std(), np.sqrt(np.maximum(0, np.diag(covariance))))


def test_variances_stay_exact_far_from_the_origin():
    # Moments about 0 would cancel catastrophically at an offset of 1e8 for a spread of 1.
    replicates = gaussian_run(range(40), **SETTINGS)
    shifted = Result(replace(replicates, atoms=replicates.atoms + 1e8))
    np.testing.assert_allclose(shifted.cov(), RESULT.cov(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shifted.var(), RESULT.var(), rtol=1e-6)


def test_without_bias_corrections_quantiles_are_those_of_the_pooled_draws():
    # With k past every meeting, all weights are 1 / |S|: an ordinary empirical measure.
    # The levels avoid multiples of 1 / N, where rounding of the cumulative weights
    # decides between two neighbouring atoms; the interval's are 0.2225 and 0.7775.
    result = gaussian_result(k=40, m=45, lag=1, max_iter=None)
    assert not result.measure.weights.min() < 0
    levels = [0.137, 0.503, 0.911]
    for dim in range(2):
        atoms = result.measure.atoms[:, dim]
        expected = np.quantile(atoms, levels, method="inverted_cdf")
        np.testing.assert_array_equal(result.quantile(levels)[:, dim], expected)
        central = np.quantile(atoms, [(1 - 0.555) / 2, (1 + 0.555) / 2], method="inverted_cdf")
        np.testing.assert_array_equal(result.interval(0.555)[:, dim], central)


def test_quantiles_keep_apart_atoms_that_share_some_coordinates():
    # A Gibbs sweep changes some coordinates of consecutive atoms and keeps the others.
    values = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]
    np.testing.assert_array_equal(synthetic([2] * 4, values).quantile(0.6), [0.0, 3.0])


@pytest.mark.parametrize(
    ("dims", "limits"), [(0, (-2.0, 3.0)), ([1, 0], [(-3.0, 2.0), (-2.0, 3.0)])]
)
def test_histogram_bins_average_each_replicates_signed_mass_in_half_open_bins(dims, limits):
    masses, errors, edges = RESULT.histogram(dims, bins=7, range=limits)
    if np.ndim(dims) == 0:
        dims, edges, limits = [dims], [edges], [limits]
    for edge, limit in zip(edges, limits, strict=True):
        assert (edge[0], edge[-1]) == limit
    atoms, weights, owner = (
        RESULT.measure.atoms[:, dims],
        RESULT.measure.weights,
        RESULT.measure.replicate,
    )
    cells = itertools.product(*(itertools.pairwise(edge) for edge in edges))
    inside = [
        np.all([(atoms[:, i] >= a) & (atoms[:, i] < b) for i, (a, b) in enumerate(cell)], axis=0)
        for cell in cells
    ]
    per_replicate = np.array(
        [[weights[(owner == r) & cell].sum() for cell in inside] for r in range(40)]
    ).reshape(40, *masses.shape)
    np.testing.assert_allclose(masses, per_replicate.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(errors, per_replicate.std(axis=0, ddof=1) / np.sqrt(40), atol=1e-12)
    total, _, _ = RESULT.histogram(dims, bins=7)
    assert abs(total.sum() - 1) < 1e-12


def test_a_numpy_integer_of_bins_computes_as_a_python_integer():
    # 20 ** 2 wraps to 144 in uint8.
    masses, _, _ = RESULT.histogram([0, 1], bins=np.uint8(20))
    assert masses.shape == (20, 20)


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


def test_without_pandas_the_summary_names_the_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)  # import pandas then fails
    with pytest.raises(ImportError, match=r"telescopic\[pandas\]"):
        RESULT.summary().to_pandas()


def test_a_result_missing_requested_replicates_is_interrupted_until_they_run():
    # Replicates 0 to 39 ran of 0 to 59 requested; a run of 40 to 59 completes them.
    interrupted = Result(replace(RESULT._replicates, requested=np.arange(60)))
    assert interrupted.interrupted
    assert "not exactly unbiased" in repr(interrupted.summary())
    rest = Result(gaussian_run(range(40, 60), **SETTINGS))
    assert not interrupted.merge(rest).interrupted


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        (["a"], ValueError, "names must have 2 entries"),
        (["a", "a"], ValueError, "names must be distinct"),
        ([0, 1], TypeError, "names must be a sequence of strings"),
        ("ab", TypeError, "names must be a sequence of strings"),
    ],
)
def test_names_must_be_distinct_strings_one_per_coordinate(names, error, message):
    with pytest.raises(error, match=message):
        Result(RESULT._replicates, names=names)


def test_merging_two_runs_equals_one_run_over_both_sets_of_replicates():
    def part(indices, seed=4):
        return Result(gaussian_run(indices, seed, **SETTINGS))

    merged, whole = part(range(5)).merge(part(range(5, 12))), part(range(12))
    np.testing.assert_array_equal(merged.meeting_times, whole.meeting_times)
    np.testing.assert_array_equal(merged.measure.atoms, whole.measure.atoms)
    np.testing.assert_array_equal(merged.measure.replicate, whole.measure.replicate)


def test_a_result_needs_the_signed_measures_of_its_replicates():
    # Every pair censored, so no missing atom shows that none were stored.
    replicates = run(
        TARGET,
        RandomWalk(np.eye(2), scale=1e-6),
        Normal([2.0, -1.0], [1.0, 1.0]),
        range(3),
        k=0,
        m=0,
        lag=1,
        max_iter=5,
        rng=1,
        store=False,
    )
    assert np.all(np.isinf(replicates.meeting_time))
    with pytest.raises(ValueError, match="store=True"):
        Result(replicates)


def test_censored_pairs_need_the_cap_of_their_run():
    tau = RESULT._replicates.meeting_time.copy()
    tau[0] = np.inf
    with pytest.raises(ValueError, match="max_iter of their run"):
        Result(replace(RESULT._replicates, meeting_time=tau))


def test_results_from_equal_array_seeds_merge():
    def part(indices):
        seed = np.random.SeedSequence(np.array([1, 2, 3, 4]))
        return Result(gaussian_run(indices, seed, **SETTINGS))

    assert part(range(3)).merge(part(range(3, 5))).n_replicates == 5


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda r: r.histogram(5), ValueError, r"dims must be in \[0, 2\)"),
        (lambda r: r.histogram([0, 1.0]), TypeError, "dims must be an integer"),
        (lambda r: r.histogram(0, bins=0), ValueError, "bins must be at least 1"),
        (lambda r: r.histogram([0, 1], range=(0.0, 1.0)), ValueError, r"range must have shape"),
        (lambda r: r.tv_bound([-1]), ValueError, "t must be non-negative"),
        (lambda r: r.w1_bound([-1]), ValueError, "t must be non-negative"),
        (lambda r: r.quantile(np.nan), ValueError, "q must be in"),
    ],
)
def test_invalid_arguments_of_the_result_raise(call, error, message):
    with pytest.raises(error, match=message):
        call(RESULT)


def test_runs_with_equal_scipy_initial_distributions_merge():
    # Each prints its own address, as in two processes.
    first, second = (
        Result(
            gaussian_run(indices, **SETTINGS),
            initial=stats.multivariate_normal([2.0, -1.0], np.eye(2)),
        )
        for indices in (range(3), range(3, 5))
    )
    assert first.merge(second).n_replicates == 5


def test_the_default_w1_times_reach_a_censored_pairs_last_distance():
    replicates = gaussian_run(range(40), **SETTINGS, max_iter=13)
    result = Result(replicates)
    assert result.n_censored
    bound = result.w1_bound()
    assert len(bound) == replicates.distance_time.max() - result.lag + 1
    assert bound[-1] > 0
    # The bound averages over every replicate, censored ones included, as the TV bound.
    times = np.arange(len(bound))
    expected = w1_bound(replicates.distance_time, replicates.distance, 40, result.lag, times)
    np.testing.assert_array_equal(bound, expected)


def test_the_default_bound_times_end_where_the_bound_vanishes():
    bound = RESULT.tv_bound()
    assert len(bound) == int(RESULT.meeting_times.max()) - RESULT.lag
    assert bound[-1] > 0
