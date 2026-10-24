import numpy as np
import pytest

from references import gaussian_log_prob
from telescopic._random import MAIN, root_seed
from telescopic.engine import Status, merge, run
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.initial import Empirical, Normal
from telescopic.kernels import RandomWalk, State
from telescopic.target import Target

# Point by point, so that a replicate's results are bitwise independent of the batch.
TARGET = Target(gaussian_log_prob)
FAR = Normal(mean=[3.0, -1.0], cov=[1.0, 1.0])


def simulate(kernel=None, initial=FAR, n=20, indices=None, **options):
    settings = {"k": 2, "m": 10, "lag": 1, "lanes": 8} | options
    return run(
        TARGET,
        RandomWalk(np.eye(2), scale=1.0) if kernel is None else kernel,
        initial,
        range(n) if indices is None else indices,
        root=root_seed(1),
        namespace=MAIN,
        **settings,
    )


@pytest.mark.parametrize(
    ("k", "m", "lag", "thin"),
    [(0, 0, 1, 1), (2, 10, 1, 1), (3, 12, 4, 1), (5, 20, 3, 2), (0, 7, 7, 3)],
)
def test_signed_measures_match_their_definition_on_the_recorded_paths(k, m, lag, thin):
    replicates = simulate(k=k, m=m, lag=lag, thin=thin, keep_paths=True)
    averaged = averaging_times(k, m, thin)
    for index, tau in zip(replicates.index, replicates.meeting_time.astype(int), strict=True):
        x, y = replicates.paths[index]
        t = np.arange(averaged[0] + lag, tau)
        c = correction_coefficients(t, k, m, lag, thin)
        rows = replicates.owner == index
        atoms, weights = replicates.atoms[rows], replicates.weights[rows]
        assert len(atoms) == len(averaged) + 2 * np.count_nonzero(c)
        for h in (lambda z: z[:, 0], lambda z: np.sin(z).sum(axis=1)):
            expected = h(x[averaged]).mean() + np.sum(c * (h(x[t]) - h(y[t - lag])))
            assert abs(np.sum(weights * h(atoms)) - expected) < 1e-12


@pytest.mark.parametrize("initial", [FAR, Empirical([[0.0, 0.0]])], ids=["far", "point mass"])
@pytest.mark.parametrize("lag", [1, 3])
def test_the_meeting_time_is_the_first_time_the_chains_agree(initial, lag):
    # With a point mass, X_L = Y_0 when X rejects L proposals in a row: tau = L.
    k, m = 1, 6
    replicates = simulate(
        RandomWalk(np.eye(2), scale=3.0), initial, k=k, m=m, lag=lag, keep_paths=True
    )
    for index, tau, cost in zip(
        replicates.index, replicates.meeting_time.astype(int), replicates.cost, strict=True
    ):
        x, y = replicates.paths[index]
        assert len(x) == max(tau, m) + 1
        assert len(y) == tau - lag + 1
        equal = np.all(x[lag : tau + 1] == y, axis=1)
        assert equal[-1]
        assert not equal[:-1].any()
        assert cost == lag + 2 * (tau - lag) + max(0, m - tau)
        rows = replicates.distance_owner == index
        np.testing.assert_array_equal(replicates.distance_time[rows], np.arange(lag, tau))
        np.testing.assert_allclose(
            replicates.distance[rows], np.linalg.norm(x[lag:tau] - y[:-1], axis=1)
        )
    assert np.any(replicates.meeting_time == lag) == (initial is not FAR)


@pytest.mark.parametrize("coupling", ["reflection", "maximal"])
def test_a_replicates_output_does_not_depend_on_the_lanes(coupling):
    # Spec R-EXEC-2, with a coupling whose draw count varies ("maximal").
    kernel = RandomWalk(np.eye(2), scale=1.0, coupling=coupling)
    runs = [simulate(kernel, n=30, k=3, m=8, lag=2, lanes=lanes) for lanes in (1, 7, 64)]
    np.testing.assert_array_equal(runs[0].index, np.arange(30))
    for other in runs[1:]:
        for name, value in vars(other).items():
            if name != "seconds":
                np.testing.assert_array_equal(value, getattr(runs[0], name))


class Tagged(RandomWalk):
    """X moves by 1 and Y by 100 per step, so a chain that takes the other's move shows."""

    def step(self, target, state, draws):
        x = state.x + 1.0
        return State(x, target.log_prob(x)), np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        y = state_y.x + 100.0
        return self.step(target, state_x, draws)[0], State(y, target.log_prob(y))


def test_each_chain_receives_its_own_transition():
    start = Empirical([[0.0, 0.0]])  # integer positions, so the differences are exact
    replicates = simulate(Tagged(np.eye(2)), start, n=3, lag=2, max_iter=12, keep_paths=True)
    for x, y in replicates.paths.values():
        np.testing.assert_array_equal(np.diff(x, axis=0), 1.0)
        np.testing.assert_array_equal(np.diff(y, axis=0), 100.0)


def test_a_cap_censors_unmet_pairs_and_drops_their_measures():
    replicates = simulate(Tagged(np.eye(2)), n=6, k=2, m=10, max_iter=15)
    np.testing.assert_array_equal(replicates.meeting_time, np.inf)
    assert len(replicates.atoms) == 0
    np.testing.assert_array_equal(np.bincount(replicates.distance_owner), 15)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"k": 3, "m": 2}, "0 <= k <= m"),
        ({"lag": 0}, "at least 1"),
        ({"thin": 0}, "at least 1"),
        ({"max_iter": 10}, "max_iter must exceed"),
        ({"lanes": 0}, "lanes must be"),
        ({"n": 0}, "indices must be a non-empty sequence"),
        ({"indices": [0.5, 1.7]}, "distinct integers"),
        ({"indices": [3, 3]}, "distinct integers"),
        ({"indices": [-1, 0]}, "distinct integers"),
    ],
)
def test_invalid_settings_raise(options, message):
    with pytest.raises(ValueError, match=message):
        simulate(**options)


def test_pilot_mode_returns_meeting_times_only():
    replicates = simulate(k=0, m=0, lag=1, store=False)
    assert (replicates.meeting_time >= 1).all()
    assert len(replicates.atoms) == len(replicates.distance) == 0


def test_a_monitor_sees_the_progress_and_can_stop_new_replicates():
    seen = []

    def monitor(status):
        seen.append(status)
        return len(seen) >= 3

    replicates = simulate(n=100, lanes=4, monitor=monitor)
    np.testing.assert_array_equal(replicates.index, np.arange(4))
    assert seen[0] == Status(started=4, finished=0, slowest=seen[0].slowest, t=1)
    assert seen[-1].finished == 4


def test_the_run_records_its_evaluations_and_the_starts_outside_the_support():
    def half_line(x):
        return 0.0 if x[0] > 0 else -np.inf

    target = Target(half_line)
    replicates = run(
        target,
        RandomWalk(np.eye(1)),
        Empirical([[-1.0], [1.0]]),
        range(10),
        root=root_seed(2),
        namespace=MAIN,
        k=0,
        m=2,
        lag=1,
        max_iter=50,
    )
    assert replicates.n_log_prob == target.n_log_prob > 0
    assert 0 < len(replicates.zero_density) < 10


def test_a_kernel_of_another_dimension_than_the_points_raises():
    with pytest.raises(ValueError, match="kernel works in 1 dimensions"):
        simulate(RandomWalk(np.eye(1)))


def test_merging_refuses_parts_that_share_replicates():
    part = simulate(n=3)
    with pytest.raises(ValueError, match="share replicate indices"):
        merge([part, part])
