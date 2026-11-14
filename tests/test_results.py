import itertools
import sys
from dataclasses import replace
from xml.etree import ElementTree

import numpy as np
import pytest
from scipy import stats

from references import gaussian_log_prob, synthetic
from telescopic import examples, sample
from telescopic.diagnostics import w1_bound
from telescopic.engine import run
from telescopic.estimators import unbiased_covariance, unbiased_variance, weighted_quantile
from telescopic.initial import Normal
from telescopic.kernels import Independence, RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning

TARGET = Target(gaussian_log_prob, vectorize=True)
SETTINGS = {"k": 2, "m": 12, "lag": 2}


def gaussian_run(indices, seed=1, target=TARGET, **settings):
    return run(
        target,
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


@pytest.mark.parametrize(
    "result",
    [gaussian_result(2), gaussian_result(30, k=3, m=3, lag=1, max_iter=4)],
    ids=["two replicates", "some censored"],
)
def test_an_estimate_averages_the_replicates_with_a_student_t_interval(result):
    estimate = result.estimate(lambda x: x**2, level=0.9)
    values = result.measure.per_replicate(lambda x: x**2)
    error = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    half_width = stats.t.ppf(0.95, len(values) - 1) * error
    np.testing.assert_array_equal(estimate.value, values.mean(axis=0))
    np.testing.assert_allclose(estimate.standard_error, error, rtol=1e-12)
    # One array, lower ends first, as the credible interval of the result.
    assert estimate.interval.shape == (2, 2)
    np.testing.assert_allclose(estimate.interval[0], estimate.value - half_width, rtol=1e-12)
    np.testing.assert_allclose(estimate.interval[1], estimate.value + half_width, rtol=1e-12)


def test_result_repr_names_its_settings_without_computing_diagnostics():
    result = gaussian_result(30, k=3, m=3, lag=1, max_iter=4)
    expected = (
        f"Result(R={result.n_replicates}, d=2, k=3, m=3, lag=1, "
        f"censored={result.n_censored}, interrupted=False)"
    )
    assert repr(result) == expected
    assert "_diagnostics" not in vars(result)


def test_an_estimate_from_one_replicate_raises_instead_of_an_undefined_interval():
    with pytest.raises(ValueError, match="at least 2 replicates that met, not 1"):
        gaussian_result(1).estimate(lambda x: x)


@pytest.mark.parametrize(
    ("meeting_times", "method", "needed"),
    [
        ([2, -1], lambda result: result.mean(), 2),
        ([2, -1], lambda result: result.cov(), 2),
        ([2, -1], lambda result: result.var(), 2),
        ([-1, -1], lambda result: result.quantile(0.5), 1),
    ],
    ids=["mean", "cov", "var", "quantile"],
)
def test_estimates_from_too_few_replicates_that_met_raise(meeting_times, method, needed):
    # Censored pairs do not count: without the check, cov and var raised the estimator's
    # message and quantile an IndexError.
    result = synthetic(meeting_times, np.ones((2, 2)), max_iter=8)
    message = f"at least {needed} replicates that met, not {needed - 1}"
    with pytest.raises(ValueError, match=message):
        method(result)


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
    replicates = RESULT._replicates
    shifted = Result(replace(replicates, atoms=replicates.atoms + 1e8))
    np.testing.assert_allclose(shifted.cov(), RESULT.cov(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shifted.var(), RESULT.var(), rtol=1e-6)


def test_without_bias_corrections_quantiles_are_those_of_the_pooled_draws():
    # With k past every meeting, all weights are 1 / |S|: an ordinary empirical measure.
    # The levels avoid multiples of 1 / N, where rounding of the cumulative weights
    # decides between two neighboring atoms; the interval's are 0.2225 and 0.7775.
    result = gaussian_result(k=40, m=45, lag=1, max_iter=None)
    assert not result.measure.weights.min() < 0
    levels = [0.137, 0.503, 0.911]
    for dim in range(2):
        atoms = result.measure.atoms[:, dim]
        expected = np.quantile(atoms, levels, method="inverted_cdf")
        np.testing.assert_array_equal(result.quantile(levels)[:, dim], expected)
        central = np.quantile(atoms, [(1 - 0.555) / 2, (1 + 0.555) / 2], method="inverted_cdf")
        np.testing.assert_array_equal(result.credible_interval(0.555)[:, dim], central)


def test_quantile_of_h_equals_weighted_quantile_recipe():
    # The recipe applies h to every atom and divides the pooled weights by R; RESULT has
    # bias corrections, so some weights are negative.
    assert RESULT.measure.weights.min() < 0
    weights = RESULT.measure.weights / RESULT.n_replicates
    atoms = RESULT.measure.atoms
    product = atoms[:, 0] * atoms[:, 1]
    levels = [0.137, 0.503, 0.911]
    expected = weighted_quantile(product, weights, levels)
    np.testing.assert_array_equal(RESULT.quantile(levels, lambda x: x[:, 0] * x[:, 1]), expected)
    one_level = RESULT.quantile(0.503, lambda x: x[:, 0] * x[:, 1])
    assert one_level.shape == ()
    assert one_level == expected[1]
    # Values of shape (n, p) give a column per value, as the coordinates do.
    both = RESULT.quantile(levels, lambda x: np.column_stack([x[:, 0] * x[:, 1], x[:, 0]]))
    np.testing.assert_array_equal(both, np.column_stack([expected, RESULT.quantile(levels)[:, 0]]))
    central = weighted_quantile(product, weights, [0.16, 0.84])
    np.testing.assert_array_equal(RESULT.credible_interval(h=lambda x: x[:, 0] * x[:, 1]), central)


def test_quantiles_keep_apart_atoms_that_share_some_coordinates():
    # A Gibbs sweep changes some coordinates of consecutive atoms and keeps the others.
    values = [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]
    np.testing.assert_array_equal(synthetic([2] * 4, values).quantile(0.6), [0.0, 3.0])


@pytest.mark.parametrize(
    ("dims", "limits"), [(0, (-2.0, 3.0)), ([1, 0], [(-3.0, 2.0), (-2.0, 3.0)])]
)
def test_histogram_bins_average_each_replicates_signed_mass_closing_only_the_last_bin(
    dims, limits
):
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
        np.all(
            [
                (atoms[:, i] >= a) & ((atoms[:, i] < b) | (b == edges[i][-1]) & (atoms[:, i] == b))
                for i, (a, b) in enumerate(cell)
            ],
            axis=0,
        )
        for cell in cells
    ]
    per_replicate = np.array(
        [[weights[(owner == r) & cell].sum() for cell in inside] for r in range(40)]
    ).reshape(40, *masses.shape)
    np.testing.assert_allclose(masses, per_replicate.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(errors, per_replicate.std(axis=0, ddof=1) / np.sqrt(40), atol=1e-12)
    total, _, _ = RESULT.histogram(dims, bins=7)
    assert abs(total.sum() - 1) < 1e-12


@pytest.mark.parametrize("dims", [0, [0, 1]])
def test_histogram_keeps_an_atom_on_the_upper_edge_of_its_range(dims):
    # Each replicate's one atom of weight 1 lies on the edges [0, 1] of the range, so
    # closing the last bin counts all of the mass, as numpy.histogram does.
    values = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 1.0], [1.0, 0.25]])
    result = synthetic([1] * 4, values)
    limits = (0.0, 1.0) if np.ndim(dims) == 0 else [(0.0, 1.0)] * 2
    masses, _, _ = result.histogram(dims, bins=4, range=limits)
    assert masses.sum() == 1
    expected, _ = np.histogramdd(
        values[:, np.atleast_1d(dims)], bins=4, range=np.reshape(limits, (-1, 2))
    )
    np.testing.assert_array_equal(masses, expected / 4)
    # The default range ends at the largest atom, which the same rule keeps.
    automatic, _, _ = result.histogram(dims, bins=4)
    np.testing.assert_array_equal(automatic, masses)


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


def test_variance_and_quantiles_of_a_censored_run_use_the_pairs_that_met():
    # 7 of 10 pairs met, each with one atom of weight 1: the moments are those of the 7
    # values, divided by the 7 replicates that met, not the 10 that ran.
    met = np.random.default_rng(9).standard_normal((7, 2))
    result = synthetic(
        [1, 2, 3, 4, 5, 6, 7, -1, -1, -1], np.vstack([met, np.zeros((3, 2))]), max_iter=8
    )
    expected = unbiased_variance(met, np.mean(met**2, axis=0))
    np.testing.assert_allclose(result.var(), expected, rtol=1e-12)
    np.testing.assert_allclose(result.var(), np.diag(result.cov()), rtol=1e-12)
    np.testing.assert_allclose(result.std(), np.sqrt(expected), rtol=1e-12)
    np.testing.assert_array_equal(result.quantile(0.5), np.median(met, axis=0))


def test_the_summary_reports_every_parameter_and_the_run():
    summary = RESULT.summary()
    as_dict = summary.to_dict()
    assert set(as_dict["parameters"]) == {"a", "b"}
    assert as_dict["parameters"]["a"]["mean"] == RESULT.mean().value[0]
    assert as_dict["n_replicates"] == 40
    assert "meeting times: median" in repr(summary)


def test_summary_repr_html_holds_every_column():
    summary = RESULT.summary()
    page = ElementTree.fromstring(summary._repr_html_())
    header, *rows = page.iter("tr")
    assert [cell.text for cell in header] == [None, *summary.table]
    for i, (row, name) in enumerate(zip(rows, summary.names, strict=True)):
        values = [f"{column[i]:.6g}" for column in summary.table.values()]
        assert [cell.text for cell in row] == [name, *values]
    items = [item.text for item in page.iter("li")]
    assert items == summary._footer_lines()
    assert any(item.startswith("meeting times: median") for item in items)


def test_summary_repr_html_escapes_the_names():
    summary = Result(RESULT._replicates, names=["<b>", "a&b"]).summary()
    text = summary._repr_html_()
    assert "<b>" not in text
    page = ElementTree.fromstring(text)
    assert [row[0].text for row in list(page.iter("tr"))[1:]] == ["<b>", "a&b"]


@pytest.mark.parametrize("d", [2, 10])
def test_summary_of_the_default_runs_fits_in_80_characters(d):
    # The issue's two-dimensional run printed a line of 130 characters, past the 83 that
    # a code output of the documentation holds at 1440 px. At d = 2 and 10 the longest
    # lines are now 75 and 76 characters (notes/experiments/i85_summary_width.py).
    example = examples.gaussian(d)
    result = sample(example.log_prob, example.initial, vectorize=True, rng=0)
    assert max(len(line) for line in repr(result.summary()).splitlines()) <= 80


def test_summary_prints_the_kernel_on_its_own_line_to_4_significant_figures():
    kernel = RandomWalk(np.eye(2), scale=2.3812345678)
    lines = repr(Result(RESULT._replicates, kernel=kernel).summary()).splitlines()
    assert "kernel: RandomWalk(cov=<2x2>, scale=2.381, coupling='reflection')" in lines
    assert "scale=2.3812345678" in repr(kernel)


def test_summary_rounds_floats_but_not_dotted_tokens_in_the_kernel():
    # A version in a proposal's repr printed as '3.1.12', and 123456.7 as 123500.0.
    class Fitted:
        def sample(self, draws):
            raise NotImplementedError

        def log_density(self, x):
            raise NotImplementedError

        def __repr__(self):
            return "Fitted(version='3.10.12', scale=123456.7, shift=0.8989355)"

    summary = Result(RESULT._replicates, kernel=Independence(Fitted())).summary()
    expected = "Independence(proposal=Fitted(version='3.10.12', scale=1.235e+05, shift=0.8989))"
    assert f"kernel: {expected}" in summary._footer_lines()


def test_the_summary_prints_the_tail_index_on_its_own_line_from_500_replicates():
    tau = np.random.default_rng(8).geometric(0.1, size=500) + 1
    result = synthetic(tau, np.zeros(500))
    line = f"tail index of the meeting times: {result.diagnostics()['tail_index']:.3g}"
    assert line in repr(result.summary()).splitlines()
    assert "tail index" not in repr(synthetic(tau[:499], np.zeros(499)).summary())


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
        (["a"], ValueError, "the atoms are points of dimension 2, but names gives 1"),
        (["a", "a"], ValueError, "names must be distinct"),
        ([0, 1], TypeError, "names must be a sequence of strings"),
        ("ab", TypeError, "names must be a sequence of strings"),
    ],
)
def test_names_must_be_distinct_strings_one_per_coordinate(names, error, message):
    with pytest.raises(error, match=message):
        Result(RESULT._replicates, names=names)


def test_names_from_an_iterator_are_read_once():
    assert Result(RESULT._replicates, names=iter(["a", "b"])).names == ["a", "b"]


def test_merging_two_runs_equals_one_run_over_both_sets_of_replicates():
    # Point by point, so that a replicate's bits do not depend on the rest of its run.
    def part(indices):
        return Result(gaussian_run(indices, 4, Target(gaussian_log_prob), **SETTINGS))

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


def test_a_result_with_no_replicates_raises_at_construction():
    # diagnostics() and check() divided by zero, and tv_bound() reduced an empty array.
    with pytest.raises(ValueError, match="at least one replicate"):
        synthetic([], [])


def test_meeting_time_quantiles_sort_censored_pairs_above_the_cap():
    # 4 of 10 pairs censored: the median is the 5th smallest meeting time, and the 90%
    # and 99% quantiles and the maximum fall on censored pairs. Over the pairs that met,
    # the median was 3.5 and the others at most 6.
    result = synthetic([1, 2, 3, 4, 5, 6, -1, -1, -1, -1], np.arange(6.0), max_iter=8)
    values = result.diagnostics()
    assert values["meeting_time_50"] == 5
    for key in ("meeting_time_90", "meeting_time_99", "meeting_time_max"):
        assert values[key] == np.inf
    assert "median 5, 90% > 8, 99% > 8, max > 8; censored 4" in repr(result.summary())


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
        (lambda r: r.histogram([]), ValueError, "dims must name at least one coordinate"),
        (lambda r: r.histogram(0, bins=0), ValueError, "bins must be at least 1"),
        (lambda r: r.histogram([0, 1], range=(0.0, 1.0)), ValueError, r"range must have shape"),
        (lambda r: r.tv_bound([-1]), ValueError, "t must be non-negative"),
        (lambda r: r.w1_bound([-1]), ValueError, "t must be non-negative"),
        (lambda r: r.quantile(np.nan), ValueError, "q must be in"),
        (lambda r: r.quantile(0.5, lambda x: x[:1]), ValueError, "h must map atoms"),
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


def test_a_merge_keeps_this_results_records_and_fills_the_missing_ones_from_the_other():
    kernel, initial, pilot = RandomWalk(np.eye(2)), Normal([2.0, -1.0], [1.0, 1.0]), np.ones(3)
    bare = Result(gaussian_run(range(3), **SETTINGS))
    full = Result(gaussian_run(range(3, 5), **SETTINGS), kernel=kernel, pilot=pilot)
    other = Result(gaussian_run(range(5, 7), **SETTINGS), kernel=RandomWalk(), initial=initial)
    merged = bare.merge(full).merge(other)
    assert merged.kernel is kernel
    assert merged.initial is initial
    assert merged.pilot is pilot
    assert merged.tuning is None


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


def test_tv_bound_counts_a_censored_pair_as_meeting_after_its_cap():
    # The engine compares the chains at the cap, 25, before censoring a pair, so its
    # meeting time is at least 26. With L = 3 the term of a pair is
    # max(0, ceil((tau - L - t) / L)), and 26 adds 1 to the cap's where 3 divides 22 - t.
    tau = np.array([4, 7, 9, 12, 20, -1, -1])
    result = synthetic(tau, np.zeros(7), k=19, lag=3, max_iter=25)
    known = np.where(tau < 0, 26, tau)
    t = np.arange(30)
    terms = np.maximum(0, np.ceil((known[:, None] - 3 - t) / 3))
    expected = np.minimum(1, terms.mean(axis=0))
    np.testing.assert_allclose(result.tv_bound(t), expected, rtol=1e-12)
    np.testing.assert_allclose(result.tv_bound(), expected[: 26 - 3], rtol=1e-12)
    assert result.diagnostics()["tv_bound_at_k"] == pytest.approx(4 / 7, rel=1e-12)


def test_the_default_bound_times_end_where_the_bound_vanishes():
    bound = RESULT.tv_bound()
    assert len(bound) == int(RESULT.meeting_times.max()) - RESULT.lag
    assert bound[-1] > 0
