from dataclasses import replace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import telescopic
from references import gaussian_log_prob
from telescopic import plotting
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])
# A cap that censors some pairs, so that the plots show them.
with pytest.warns(ConvergenceWarning):
    RESULT = telescopic.sample(
        gaussian_log_prob, INIT, 30, kernel=RandomWalk(np.eye(2)), k=5, m=10, max_iter=12, rng=1
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_the_survival_curve_counts_censored_pairs_up_to_the_cap():
    histogram, survival = RESULT.plot_meeting_times()
    shifted = RESULT.meeting_times - RESULT.lag
    assert sum(patch.get_height() for patch in histogram.patches) == RESULT.n_replicates
    assert histogram.patches[0].get_x() == np.min(shifted)
    t, above = survival.lines[0].get_data()
    cap = RESULT.max_iter - RESULT.lag
    assert RESULT.n_censored > 0
    assert t[-1] == cap
    np.testing.assert_allclose(above, [np.mean(np.minimum(shifted, cap) > s) for s in t])
    assert survival.get_yscale() == "log"


def test_the_tv_bound_plot_draws_the_bound_and_marks_k():
    ax = RESULT.plot_tv_bound()
    t, bound = ax.lines[0].get_data()
    assert t[0] == 0
    np.testing.assert_array_equal(bound, RESULT.tv_bound(t))
    assert ax.get_yscale() == "log"
    assert ax.lines[-1].get_xdata()[0] == RESULT.k


def test_pairs_that_all_met_at_the_lag_plot_without_a_log_scale():
    replicates = RESULT._replicates
    met = np.full(len(replicates.index), float(RESULT.lag))
    result = telescopic.Result(replace(replicates, meeting_time=met))
    assert result.plot_meeting_times()[1].get_yscale() == "linear"
    assert result.plot_tv_bound().get_yscale() == "linear"


def test_marginal_plots_draw_signed_densities_with_error_boxes():
    axes = RESULT.plot_marginals(bins=10)
    assert [ax.get_xlabel() for ax in axes] == RESULT.names
    for dim, ax in enumerate(axes):
        masses, errors, edges = RESULT.histogram(dim, bins=10)
        widths = np.diff(edges)
        bars, boxes = ax.patches[:10], ax.patches[10:]
        np.testing.assert_allclose([bar.get_height() for bar in bars], masses / widths)
        np.testing.assert_allclose([box.get_y() for box in boxes], (masses - 2 * errors) / widths)
        np.testing.assert_allclose([box.get_height() for box in boxes], 4 * errors / widths)


def test_corner_plots_put_marginals_on_the_diagonal_and_joint_densities_below():
    grid = RESULT.plot_corner(bins=8)
    assert grid.shape == (2, 2)
    assert grid[0, 0].patches
    assert not grid[0, 1].axison
    density = np.asarray(grid[1, 0].collections[0].get_array()).reshape(8, 8)
    (masses_x, _, edges_x), (masses_y, _, edges_y) = (RESULT.histogram(dim, 8) for dim in (0, 1))
    masses = density * np.outer(np.diff(edges_y), np.diff(edges_x))
    np.testing.assert_allclose(masses.sum(axis=0), masses_x, atol=1e-12)
    np.testing.assert_allclose(masses.sum(axis=1), masses_y, atol=1e-12)


def test_plots_draw_on_the_axes_they_are_given():
    axes = plt.subplots(1, 4)[1]
    assert RESULT.plot_meeting_times(axes[:2]) == list(axes[:2])
    assert RESULT.plot_tv_bound(axes[2]) is axes[2]
    assert RESULT.plot_marginals(1, axes=axes[3]) == [axes[3]]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda axes: RESULT.plot_meeting_times(axes[:3]), "axes must hold 2 Axes, not 3"),
        (lambda axes: RESULT.plot_marginals([0, 1], axes=axes[:1]), "axes must hold 2 Axes"),
        (lambda axes: RESULT.plot_marginals([0, 2]), r"dims must be in \[0, 2\)"),
        (lambda axes: RESULT.plot_corner([-1]), r"dims must be in \[0, 2\)"),
    ],
)
def test_invalid_plot_arguments_raise_before_drawing(call, message):
    axes = plt.subplots(1, 3)[1]
    with pytest.raises(ValueError, match=message):
        call(axes)
    assert not any(ax.has_data() for ax in axes)


def test_paths_need_the_paths_of_the_engine():
    replicates = run(
        Target(gaussian_log_prob),
        RandomWalk(np.eye(2)),
        INIT,
        range(3),
        rng=2,
        k=1,
        m=5,
        lag=1,
        keep_paths=True,
    )
    ax = plotting.paths(replicates)
    assert len(ax.lines) == 6
    with pytest.raises(ValueError, match="keep_paths=True"):
        plotting.paths(RESULT._replicates)
    with pytest.raises(ValueError, match="n must be at least 1"):
        plotting.paths(replicates, n=0)
    with pytest.raises(ValueError, match=r"dims must be two coordinates in \[0, 2\)"):
        plotting.paths(replicates, dims=(0, 2))


def test_paths_in_one_dimension_shift_y_by_the_lag():
    initial = Normal([3.0], [[1.0]])
    replicates = run(
        Target(gaussian_log_prob),
        RandomWalk(np.eye(1)),
        initial,
        [0],
        rng=3,
        k=1,
        m=5,
        lag=3,
        keep_paths=True,
    )
    x, y = plotting.paths(replicates).lines
    assert (x.get_xdata()[0], y.get_xdata()[0]) == (0, 3)
