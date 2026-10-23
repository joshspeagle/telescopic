import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import telescopic
from references import gaussian_log_prob
from telescopic import plotting
from telescopic._random import MAIN, root_seed
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.target import Target

INIT = Normal([2.0, -1.0], [4.0, 4.0])
RESULT = telescopic.sample(gaussian_log_prob, INIT, 30, kernel=RandomWalk(np.eye(2)), k=40, rng=1)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_meeting_time_plots_return_their_two_axes():
    histogram, survival = RESULT.plot_meeting_times()
    assert histogram.patches
    assert survival.get_yscale() == "log"


def test_the_tv_bound_plot_marks_k():
    ax = RESULT.plot_tv_bound()
    assert ax.get_yscale() == "log"
    assert ax.lines[-1].get_xdata()[0] == RESULT.k


def test_marginal_plots_show_each_coordinate_with_error_boxes():
    axes = RESULT.plot_marginals(bins=10)
    assert [ax.get_xlabel() for ax in axes] == RESULT.names
    assert all(len(ax.patches) == 20 for ax in axes)  # 10 bars and 10 boxes


def test_corner_plots_put_marginals_on_the_diagonal_and_pairs_below():
    grid = RESULT.plot_corner(bins=8)
    assert grid.shape == (2, 2)
    assert len(grid[1, 0].collections) == 1
    assert not grid[0, 1].axison


def test_paths_need_the_paths_of_the_engine():
    replicates = run(
        Target(gaussian_log_prob),
        RandomWalk(np.eye(2)),
        INIT,
        range(3),
        root=root_seed(2),
        namespace=MAIN,
        k=1,
        m=5,
        lag=1,
        keep_paths=True,
    )
    ax = plotting.paths(replicates, lag=1)
    assert len(ax.lines) == 6
    with pytest.raises(ValueError, match="keep_paths=True"):
        plotting.paths(RESULT._replicates, lag=1)
