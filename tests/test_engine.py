import numpy as np
import pytest

from telescopic._random import MAIN, root_seed
from telescopic.engine import run
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.initial import Empirical, Normal
from telescopic.kernels import Kernel, RandomWalk, State
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning, InputWarning


def gaussian_log_prob(x):
    return -0.5 * float(np.sum(x**2))


# Point by point, so that a replicate's results are bitwise independent of the batch.
TARGET = Target(gaussian_log_prob)
FAR = Normal(mean=[3.0, -1.0], cov=[1.0, 1.0])


def simulate(kernel=None, initial=FAR, n=20, **options):
    settings = {"k": 2, "m": 10, "lag": 1, "lanes": 8} | options
    return run(
        TARGET,
        RandomWalk(np.eye(2), scale=1.0) if kernel is None else kernel,
        initial,
        range(n),
        root=root_seed(1),
        namespace=MAIN,
        **settings,
    )


def owned(replicates, index):
    rows = replicates.owner == index
    return replicates.atoms[rows], replicates.weights[rows]


@pytest.mark.parametrize(
    ("k", "m", "lag", "thin"),
    [(0, 0, 1, 1), (2, 10, 1, 1), (3, 12, 4, 1), (5, 20, 3, 2), (0, 7, 7, 3)],
)
def test_signed_measures_match_their_definition_on_the_recorded_paths(k, m, lag, thin):
    replicates = simulate(k=k, m=m, lag=lag, thin=thin, keep_paths=True)
    averaged = averaging_times(k, m, thin)
    for index, tau in zip(replicates.index, replicates.meeting_time, strict=True):
        x, y = replicates.paths[index]
        t = np.arange(averaged[0] + lag, tau)
        c = correction_coefficients(t, k, m, lag, thin)
        atoms, weights = owned(replicates, index)
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
        replicates.index, replicates.meeting_time, replicates.cost, strict=True
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


class Perfect(Kernel):
    """P(x, .) = pi for every x, coupled by common draws: pairs meet at the first coupled step."""

    def step(self, target, state, draws):
        x = draws.normal(state.x.shape[1])
        return State(x, target.log_prob(x)), np.ones(len(x), dtype=bool)

    def coupled_step(self, target, state_x, state_y, draws):
        new, _ = self.step(target, state_x, draws)
        return new, State(new.x.copy(), new.log_prob.copy())


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_a_perfect_kernel_meets_at_the_first_coupled_step(lag):
    replicates = simulate(Perfect(), k=3, m=4, lag=lag)
    np.testing.assert_array_equal(replicates.meeting_time, lag + 1)
    np.testing.assert_array_equal(replicates.cost, lag + 2 + max(0, 4 - lag - 1))


class Careful(RandomWalk):
    """Refuses a coupled step on a pair that has already met."""

    def coupled_step(self, target, state_x, state_y, draws):
        assert not np.all(state_x.x == state_y.x, axis=1).any()
        return super().coupled_step(target, state_x, state_y, draws)


def test_pairs_that_met_are_never_coupled_again():
    simulate(Careful(np.eye(2), scale=3.0), Empirical([[0.0, 0.0]]), n=40, lag=2)


@pytest.mark.parametrize("coupling", ["reflection", "maximal"])
def test_a_replicates_output_does_not_depend_on_the_lanes(coupling):
    # Spec R-EXEC-2, with a coupling whose draw count varies ("maximal").
    kernel = RandomWalk(np.eye(2), scale=1.0, coupling=coupling)
    runs = [simulate(kernel, n=30, k=3, m=8, lag=2, lanes=lanes) for lanes in (1, 7, 64)]
    for other in runs[1:]:
        for name in (
            "index",
            "meeting_time",
            "cost",
            "atoms",
            "weights",
            "owner",
            "correction",
            "distance_owner",
            "distance_time",
            "distance",
        ):
            np.testing.assert_array_equal(getattr(other, name), getattr(runs[0], name))


def test_every_index_runs_exactly_once():
    replicates = simulate(n=50, lanes=7)
    np.testing.assert_array_equal(replicates.index, np.arange(50))
    assert (replicates.meeting_time >= 1).all()


class Apart(RandomWalk):
    """Couples nothing: the chains move independently and never meet."""

    def coupled_step(self, target, state_x, state_y, draws):
        return self.step(target, state_x, draws)[0], self.step(target, state_y, draws)[0]


def test_a_cap_censors_unmet_pairs_and_drops_their_measures():
    replicates = simulate(Apart(np.eye(2)), n=6, k=2, m=10, max_iter=15)
    np.testing.assert_array_equal(replicates.meeting_time, -1)
    assert len(replicates.atoms) == 0
    np.testing.assert_array_equal(np.bincount(replicates.distance_owner), 15)


def test_a_cap_must_exceed_lag_and_m():
    with pytest.raises(ValueError, match="max_iter must exceed"):
        simulate(k=2, m=10, max_iter=10)


def test_a_slow_pair_warns_once():
    with pytest.warns(ConvergenceWarning, match="has not met after 5 iterations") as record:
        simulate(Apart(np.eye(2)), n=4, max_iter=30, slow_after=5)
    assert len(record) == 1


def test_pilot_mode_returns_meeting_times_only():
    replicates = simulate(k=0, m=0, lag=1, store=False)
    assert (replicates.meeting_time >= 1).all()
    assert len(replicates.atoms) == len(replicates.distance) == 0


def test_a_stop_request_finishes_the_replicates_in_progress():
    calls = iter(range(1000))
    replicates = simulate(n=100, lanes=4, stop=lambda: next(calls) >= 3)
    np.testing.assert_array_equal(replicates.index, np.arange(4))


def test_a_start_outside_the_support_warns():
    def half_line(x):
        return 0.0 if x[0] > 0 else -np.inf

    with pytest.warns(InputWarning, match="zero target density"):
        run(
            Target(half_line),
            RandomWalk(np.eye(1)),
            Empirical([[-1.0]]),
            range(4),
            root=root_seed(2),
            namespace=MAIN,
            k=0,
            m=2,
            lag=1,
            max_iter=50,
            slow_after=10**6,
        )
