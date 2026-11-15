import numpy as np
import pytest
from scipy import stats

from references import (
    Failing,
    assert_same_outputs,
    assert_same_replicates,
    dense,
    gaussian_log_prob,
)
from telescopic._io import record_times
from telescopic._random import MAIN, replicate_generator
from telescopic.engine import Settings, Status, _Batch, merge, run, stored_bytes
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.initial import Empirical, Normal
from telescopic.kernels import MALA, Gibbs, RandomWalk, State
from telescopic.measure import SignedMeasure
from telescopic.target import Target

# Point by point, so that a replicate's results are bitwise independent of the batch.
TARGET = Target(gaussian_log_prob)
FAR = Normal(mean=[3.0, -1.0], cov=[1.0, 1.0])


def simulate(kernel=None, initial=FAR, n=20, indices=None, target=TARGET, **options):
    settings = {"k": 2, "m": 10, "lag": 1, "lanes": 8, "rng": 1} | options
    return run(
        target,
        RandomWalk(np.eye(2), scale=1.0) if kernel is None else kernel,
        initial,
        range(n) if indices is None else indices,
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
        assert cost == len(x) + len(y) - 2
        rows = replicates.distance_owner == index
        np.testing.assert_array_equal(replicates.distance_time[rows], np.arange(lag, tau))
        np.testing.assert_allclose(
            replicates.distance[rows], np.linalg.norm(x[lag:tau] - y[:-1], axis=1)
        )
    assert np.any(replicates.meeting_time == lag) == (initial is not FAR)


def test_each_replicate_draws_x0_then_y0_from_its_own_stream():
    # Drawing Y0 from another replicate's stream keeps each estimate unbiased but makes
    # the replicates dependent, against what the standard error assumes.
    indices = [0, 3, 4, 9]
    replicates = simulate(indices=indices, keep_paths=True)
    for index in indices:
        generator = replicate_generator(replicates.seed, MAIN, index)
        x, y = replicates.paths[index]
        np.testing.assert_array_equal(x[0], FAR.rvs(random_state=generator))
        np.testing.assert_array_equal(y[0], FAR.rvs(random_state=generator))


@pytest.mark.parametrize("coupling", ["reflection", "maximal"])
def test_a_replicates_output_does_not_depend_on_the_lanes(coupling):
    # Spec R-EXEC-2, with a coupling whose draw count varies ("maximal").
    kernel = RandomWalk(np.eye(2), scale=1.0, coupling=coupling)
    runs = [simulate(kernel, n=30, k=3, m=8, lag=2, lanes=lanes) for lanes in (1, 7, 64)]
    np.testing.assert_array_equal(runs[0].index, np.arange(30))
    for other in runs[1:]:
        assert_same_replicates(other, runs[0])


@pytest.mark.parametrize("kernel", [RandomWalk(dense(33)), MALA(0.5, cov=dense(33))])
def test_a_vectorized_run_repeats_bitwise_with_the_same_seed_and_lanes(kernel):
    # Spec R-EXEC-2: its kernels' products are BLAS's, whose bits depend on the batch but
    # not on chance. A dense covariance, so that the products are not exact in any order.
    target = Target(gaussian_log_prob, grad=lambda x: -x, vectorize=True)
    initial = Normal(np.zeros(33), np.ones(33))
    runs = [
        run(target, kernel, initial, range(12), k=2, m=10, lag=1, lanes=5, rng=3) for _ in range(2)
    ]
    assert_same_replicates(runs[1], runs[0])


def switching(point, block):
    # A normal conditional with ppf where the other coordinate is positive, and without
    # it elsewhere.
    mean = 0.5 * point[1 - block]
    return stats.norm(mean, 1.0) if mean > 0 else stats.multivariate_normal(mean)


def test_a_replicates_output_does_not_depend_on_the_lanes_under_crn_gibbs_couplings():
    # Spec R-EXEC-2: a row's coupling of a block followed the other rows' conditionals.
    kernel = Gibbs(switching, [[0], [1]], coupling="crn")
    runs = [simulate(kernel, n=14, lanes=lanes, max_iter=400) for lanes in (1, 7)]
    assert_same_replicates(runs[1], runs[0])


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
    # X's L steps alone, then max_iter - L coupled steps of both chains.
    np.testing.assert_array_equal(replicates.cost, 1 + 2 * (15 - 1))


class Beside(Tagged):
    """Y lands beside X: equal in the first coordinate, 1e-12 away in the second."""

    def coupled_step(self, target, state_x, state_y, draws):
        moved, _ = self.step(target, state_x, draws)
        y = moved.x + np.array([0.0, 1e-12])
        return moved, State(y, target.log_prob(y))


def test_pairs_meet_only_when_every_coordinate_is_exactly_equal():
    replicates = simulate(Beside(np.eye(2)), n=3, max_iter=12)
    np.testing.assert_array_equal(replicates.meeting_time, np.inf)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"k": 3, "m": 2}, "m must be at least k = 3, not 2"),
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


@pytest.mark.parametrize("name", ["k", "m", "lag", "thin", "max_iter"])
def test_a_setting_that_is_not_an_integer_raises(name):
    # A float k or m would give biased measures without an error.
    options = {"k": 2, "m": 10, "lag": 1, "thin": 1, "max_iter": 50}
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        simulate(**options | {name: options[name] + 0.5})


def test_numpy_integer_settings_run_as_python_integers():
    numpy = simulate(
        k=np.int64(2), m=np.int32(10), lag=np.uint8(2), thin=np.int16(3), max_iter=np.int64(60)
    )
    assert all(type(value) is int for value in numpy.settings[:5])
    assert_same_replicates(numpy, simulate(k=2, m=10, lag=2, thin=3, max_iter=60))


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
    assert seen[0] == Status(finished=0, slowest=seen[0].slowest, t=1)
    assert seen[-1].finished == 4


class Still(RandomWalk):
    """Neither chain moves, so pairs from a continuous initial distribution never meet."""

    def step(self, target, state, draws):
        return state, np.zeros(len(state))

    def coupled_step(self, target, state_x, state_y, draws):
        return state_x, state_y


def test_the_status_names_the_oldest_unmet_replicate():
    # Replicates 0 and 1 run on the two lanes until max_iter censors them; replicate 2
    # then runs on lane 0 beside an idle lane.
    seen = []

    def monitor(status):
        seen.append(status)
        return False

    simulate(Still(np.eye(2)), n=3, lanes=2, k=0, m=1, lag=1, max_iter=3, monitor=monitor)
    assert seen == [
        Status(0, 0, 1),
        Status(0, 0, 2),
        Status(2, -1, 0),
        Status(2, 2, 1),
        Status(2, 2, 2),
        Status(3, -1, 0),
    ]


def test_the_status_counts_the_starts_outside_the_support_across_reused_lanes():
    # Pairs that never move run on 2 lanes in turn, each until max_iter censors it.
    seen = []

    def monitor(status):
        seen.append(status)
        return False

    replicates = run(
        Target(lambda x: 0.0 if x[0] > 0 else -np.inf),
        Still(np.eye(1)),
        Normal([0.5], [1.0]),
        range(12),
        rng=3,
        k=0,
        m=1,
        lag=1,
        max_iter=3,
        lanes=2,
        monitor=monitor,
    )
    outside = replicates.zero_density.tolist()
    assert 0 < len(outside) < 12
    assert seen[-1].zero_density == len(outside)
    assert {status.first_zero_density for status in seen} <= {-1, outside[0]}
    for status in seen:
        assert status.slowest_zero_density == (status.slowest in outside)


def test_the_run_records_its_evaluations_and_the_starts_outside_the_support():
    def half_line(x):
        return 0.0 if x[0] > 0 else -np.inf

    target = Target(half_line)
    replicates = run(
        target,
        RandomWalk(np.eye(1)),
        Empirical([[-1.0], [1.0]]),
        range(30),
        rng=2,
        k=0,
        m=2,
        lag=1,
        max_iter=50,
        keep_paths=True,
    )
    assert replicates.n_log_prob == target.n_log_prob > 0
    # A pair with either start outside the support is recorded, whichever chain it is.
    outside = [i for i, (x, y) in replicates.paths.items() if x[0, 0] < 0 or y[0, 0] < 0]
    np.testing.assert_array_equal(replicates.zero_density, sorted(outside))


def test_the_engine_takes_an_array_of_points_as_its_initial_distribution():
    # As sample and tune do, through as_initial.
    replicates = simulate(initial=np.array([[3.0, -1.0], [2.0, 0.0]]), n=3)
    assert len(replicates.index) == 3


def test_a_kernel_of_another_dimension_than_the_points_raises():
    with pytest.raises(ValueError, match="kernel works in 1 dimensions"):
        simulate(RandomWalk(np.eye(1)))


def test_merging_refuses_no_parts_and_parts_that_share_replicates():
    with pytest.raises(ValueError, match="at least one"):
        merge([])
    part = simulate(n=3)
    with pytest.raises(ValueError, match="share replicate indices"):
        merge([part, part])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rng": 2}, "different seeds"),
        ({"lag": 2}, "different settings"),
        ({"store": False}, "different settings"),
        (
            {"kernel": RandomWalk(np.eye(1)), "initial": Normal([3.0], [[1.0]])},
            "dimensions 2 and 1",
        ),
    ],
    ids=["seed", "settings", "store", "dimension"],
)
def test_merging_refuses_parts_of_another_run(options, message):
    with pytest.raises(ValueError, match=message):
        merge([simulate(n=3), simulate(indices=[3, 4], **options)])


def test_the_recorded_seed_reproduces_a_run_from_a_generator():
    first = simulate(n=4, rng=np.random.default_rng(1))
    assert_same_replicates(simulate(n=4, rng=first.seed), first)


def test_the_replicates_count_the_gradients_taken_as_zero():
    target = Target(gaussian_log_prob, grad=lambda x: np.where(x > 1, np.nan, -x))

    def part(indices):
        kernel = MALA(1.0, cov=np.eye(1))
        return run(target, kernel, Normal([0.0], [0.01]), indices, rng=3, k=1, m=5, lag=1)

    whole = part(range(10))
    assert whole.n_grad_zeroed > 0
    assert_same_replicates(merge([part(range(5)), part(range(5, 10))]), whole)


@pytest.mark.parametrize("keep", [None, 13])
@pytest.mark.parametrize("lag", [1, 3])
def test_the_storage_estimate_equals_the_bytes_a_run_stores(lag, keep):
    # Pairs met at different times, some before and some after k + L, so that the
    # correction atoms and the distances both count; with keep, the moments too.
    replicates = simulate(n=40, k=2, m=6, lag=lag, keep=keep)
    assert len(np.unique(replicates.meeting_time)) > 3
    columns = ("atoms", "weights", "owner", "time", "distance_owner", "distance_time", "distance")
    stored = sum(getattr(replicates, name).nbytes for name in (*columns, "center", "moments"))
    coupled = replicates.meeting_time - lag
    assert stored_bytes(replicates.settings, 2, coupled) == stored


@pytest.mark.parametrize("coupled", [[3.0, np.inf], [2.5]], ids=["censored", "fraction"])
def test_the_storage_estimate_refuses_steps_that_are_not_non_negative_integers(coupled):
    with pytest.raises(ValueError, match="coupled must be non-negative integers"):
        stored_bytes(Settings(2, 6, 1), 2, coupled)


def test_the_storage_estimate_is_zero_for_a_run_that_stores_nothing():
    assert stored_bytes(Settings(2, 6, 1, store=False), 2, [10]) == 0


def bounded_log_prob(x):
    # The standard normal outside x_0 > 3.5, where a third of FAR's starts fall.
    return -np.inf if x[0] > 3.5 else gaussian_log_prob(x)


def failed_run(at, log_prob=gaussian_log_prob, **options):
    # The error of a run whose log-density raises at its evaluation number `at`.
    failing = Failing(at, log_prob=log_prob)
    with pytest.raises(ValueError, match=f"failed at evaluation {at}") as caught:
        simulate(target=Target(failing), **options)
    assert caught.value is failing.error
    return caught.value


def test_engine_attaches_the_replicates_finished_before_an_error():
    error = failed_run(300)
    finished = error.telescopic_replicates
    assert 0 < len(finished.index) < 20
    assert len(finished.abandoned)
    assert not np.isin(finished.abandoned, finished.index).any()
    np.testing.assert_array_equal(finished.requested, np.arange(20))
    note = (
        f"telescopic: this error stopped the run with {len(finished.index)} of its 20 "
        f"replicates finished and {len(finished.abandoned)} in progress abandoned; "
        "error.telescopic_replicates holds the finished ones"
    )
    assert [line for line in error.__notes__ if line.startswith("telescopic:")] == [note]
    alone = simulate(indices=finished.index)
    for name in ("index", "meeting_time", "atoms", "weights", "owner", "distance", "zero_density"):
        np.testing.assert_array_equal(getattr(finished, name), getattr(alone, name), name)


@pytest.mark.parametrize("at", [200, 300, 400])
def test_records_after_an_error_hold_no_atoms_distances_zero_density_or_paths_of_unfinished_replicates(
    at,
):
    # Some abandoned replicates started at zero density, and all have distances and paths.
    whole = simulate(target=Target(bounded_log_prob), keep_paths=True)
    finished = failed_run(at, bounded_log_prob, keep_paths=True).telescopic_replicates
    assert np.isin(finished.abandoned, whole.zero_density).any()
    for name in ("owner", "distance_owner", "zero_density"):
        assert np.isin(getattr(finished, name), finished.index).all(), name
    assert set(finished.paths) == set(finished.index.tolist())


SETTINGS_CASES = [
    {"k": 0, "m": 0, "lag": 1},
    {"k": 2, "m": 10, "lag": 1},
    {"k": 3, "m": 12, "lag": 4},
    {"k": 5, "m": 20, "lag": 3, "thin": 2},
    {"k": 0, "m": 7, "lag": 7, "thin": 3},
    {"k": 4, "m": 6, "lag": 2, "max_iter": 9},
]


@pytest.mark.parametrize("options", SETTINGS_CASES)
def test_each_atom_records_the_time_it_was_recorded(options):
    # By t, and at each t the averaged atom X_t first, then X_t and Y_{t-L} of the
    # correction, read off the recorded paths.
    replicates = simulate(keep_paths=True, **options)
    k, m, lag, thin = options["k"], options["m"], options["lag"], options.get("thin", 1)
    averaged = set(averaging_times(k, m, thin).tolist())
    for index, tau in zip(replicates.index, replicates.meeting_time, strict=True):
        if np.isinf(tau):
            continue
        x, y = replicates.paths[index]
        rows = []
        for t in range(int(max(tau, m)) + 1):
            if t in averaged:
                rows.append((t, x[t], 1 / len(averaged)))
            c = correction_coefficients(t, k, m, lag, thin) if lag <= t < tau else 0.0
            if c > 0:
                rows += [(t, x[t], c), (t, y[t - lag], -c)]
        mine = replicates.owner == index
        np.testing.assert_array_equal(replicates.time[mine], [row[0] for row in rows])
        np.testing.assert_array_equal(replicates.atoms[mine], [row[1] for row in rows])
        np.testing.assert_allclose(replicates.weights[mine], [row[2] for row in rows], rtol=1e-15)
    assert replicates.time.dtype == np.int32


@pytest.mark.parametrize("options", SETTINGS_CASES)
def test_the_record_order_rebuilds_the_times_of_every_atom(options):
    replicates = simulate(n=30, **options)
    times = record_times(
        replicates.settings, replicates.index, replicates.meeting_time, replicates.owner
    )
    np.testing.assert_array_equal(times, replicates.time)
    with pytest.raises(ValueError, match="do not follow the record order"):
        record_times(
            replicates.settings, replicates.index, replicates.meeting_time, replicates.owner[1:]
        )


def test_max_iter_beyond_int32_raises_only_where_times_are_stored(monkeypatch):
    with pytest.raises(ValueError, match=r"max_iter must be at most 2\*\*31 - 1"):
        simulate(max_iter=2**31)
    simulate(k=0, m=0, store=False, max_iter=2**31)
    simulate(keep=0, max_iter=2**31)
    # Without a cap, a replicate that reaches the largest time raises, keeping the others.
    monkeypatch.setattr("telescopic.engine.MAX_TIME", 4)
    with pytest.raises(OverflowError, match=r"replicate \d+ reached t = 5") as caught:
        simulate(k=0, m=2)
    assert np.all(caught.value.telescopic_replicates.meeting_time <= 5)


def test_keep_must_be_none_or_a_non_negative_integer_with_store():
    with pytest.raises(TypeError, match=r"keep must be None or an integer >= 0, not 2\.5"):
        simulate(keep=2.5)
    with pytest.raises(ValueError, match="keep must be at least 0, not -1"):
        simulate(keep=-1)
    with pytest.raises(ValueError, match="keep needs store=True"):
        simulate(k=0, m=0, store=False, keep=3)


@pytest.mark.parametrize(
    "target",
    [TARGET, Target(gaussian_log_prob, vectorize=True)],
    ids=["point by point", "vectorized"],
)
def test_kept_rows_equal_the_same_rows_of_a_run_that_keeps_every_atom(target):
    # Replicate 12, the last kept, met; the others' rows are left out.
    options = {"n": 40, "k": 2, "m": 10, "lag": 2, "max_iter": 30, "target": target}
    full, reduced = simulate(**options), simulate(keep=13, **options)
    assert 12 in full.owner
    for name in ("index", "meeting_time"):
        np.testing.assert_array_equal(getattr(reduced, name), getattr(full, name), name)
    rows, distances = full.owner < 13, full.distance_owner < 13
    for name in ("atoms", "weights", "owner", "time"):
        np.testing.assert_array_equal(getattr(reduced, name), getattr(full, name)[rows], name)
    for name in ("distance_owner", "distance_time", "distance"):
        np.testing.assert_array_equal(getattr(reduced, name), getattr(full, name)[distances], name)
    assert full.moments.shape == (40, 0)
    assert reduced.moments.shape == (40, 5)


def moment_values(replicates):
    # Each replicate's values of h_c(x) = (x - c_r, products over triu_indices(d)), the
    # sums of |w h_c(x)|, and the atom counts n, for the replicates that met.
    met = np.isfinite(replicates.meeting_time)
    position = np.searchsorted(replicates.index[met], replicates.owner)
    center = replicates.center[met]
    shifted = replicates.atoms - center[position]
    i, j = np.triu_indices(replicates.atoms.shape[1])
    h = np.concatenate([shifted, shifted[:, i] * shifted[:, j]], axis=1)
    measure = SignedMeasure(replicates.atoms, replicates.weights, position, int(met.sum()))
    magnitude = SignedMeasure(replicates.atoms, np.abs(replicates.weights), position, len(center))
    count = np.bincount(position, minlength=len(center))[:, None]
    return measure.per_replicate(lambda _: h), magnitude.per_replicate(lambda _: np.abs(h)), count


def test_moments_equal_per_replicate_values_of_the_kept_atoms_within_rounding():
    # Claim M2: |moments - pi_r(h_c)| <= 2 gamma_{n+1} sum |w h_c(x)| elementwise, and for
    # the means |a_r - pi_r(x)| <= |s_r - 1| |c_r| + 2 gamma_{n+1} sum |w x| + eps |a_r|,
    # with s_r the sum of the replicate's weights. Censored pairs have NaN rows, and c_r is
    # the replicate's first atom, X_k.
    replicates = simulate(n=30, k=3, m=12, lag=4, max_iter=14, keep=30)
    met = np.isfinite(replicates.meeting_time)
    assert 0 < met.sum() < 30
    assert np.isnan(replicates.moments[~met]).all()
    assert np.isnan(replicates.center[~met]).all()
    first = np.searchsorted(replicates.owner, replicates.index[met])
    np.testing.assert_array_equal(replicates.center[met], replicates.atoms[first])
    assert (replicates.time[first] == 3).all()
    expected, magnitude, n = moment_values(replicates)
    eps = np.finfo(float).eps
    gamma = (n + 1) * eps / (1 - (n + 1) * eps)
    moments = replicates.moments[met]
    assert np.all(np.abs(moments - expected) <= 2 * gamma * magnitude)
    position = np.searchsorted(replicates.index[met], replicates.owner)
    measure = SignedMeasure(replicates.atoms, replicates.weights, position, int(met.sum()))
    plain = measure.per_replicate(lambda x: x)
    size = SignedMeasure(replicates.atoms, np.abs(replicates.weights), position, len(plain))
    total = np.bincount(position, replicates.weights)[:, None]
    mean = replicates.center[met] + moments[:, :2]
    bound = (
        np.abs(total - 1) * np.abs(replicates.center[met])
        + 2 * gamma * size.per_replicate(np.abs)
        + eps * np.abs(mean)
    )
    assert np.all(np.abs(mean - plain) <= bound)


def test_moments_do_not_depend_on_the_number_of_lanes():
    runs = [simulate(n=30, k=3, m=8, lag=2, lanes=lanes, keep=10) for lanes in (1, 7, 64)]
    for other in runs[1:]:
        np.testing.assert_array_equal(other.center, runs[0].center)
        np.testing.assert_array_equal(other.moments, runs[0].moments)


@pytest.mark.parametrize(("ndim", "width"), [(100, 100 + 5050), (101, 202)])
def test_moments_are_variances_above_100_dimensions(ndim, width):
    # From a point mass, a pair meets when X rejects its first proposal; those that do not
    # are censored at 20.
    replicates = run(
        Target(gaussian_log_prob, vectorize=True),
        RandomWalk(np.eye(ndim)),
        Empirical(np.zeros((1, ndim))),
        range(8),
        k=0,
        m=2,
        lag=1,
        max_iter=20,
        keep=8,
        rng=2,
    )
    assert replicates.moments.shape == (8, width)
    met = np.isfinite(replicates.meeting_time)
    assert met.any()
    position = np.searchsorted(replicates.index[met], replicates.owner)
    shifted = replicates.atoms - replicates.center[met][position]
    measure = SignedMeasure(replicates.atoms, replicates.weights, position, int(met.sum()))
    i, j = np.triu_indices(ndim) if ndim <= 100 else (np.arange(ndim), np.arange(ndim))
    squares = replicates.moments[met][:, ndim:][:, i == j]
    np.testing.assert_allclose(
        squares, measure.per_replicate(lambda _: shifted**2), rtol=1e-12, atol=1e-12
    )


def test_store_false_records_nothing():
    # The meeting-time pilot: no rows, no times and no moments.
    replicates = simulate(k=0, m=0, store=False)
    for name in ("atoms", "weights", "owner", "time", "distance_owner", "distance"):
        assert not len(getattr(replicates, name)), name
    assert replicates.center.shape == replicates.moments.shape == (20, 0)


def test_merged_moments_and_centers_follow_their_replicates():
    whole = simulate(keep=10)
    merged = merge([simulate(indices=range(12, 20), keep=10), simulate(n=12, keep=10)])
    assert_same_outputs(merged, whole)
    assert whole.moments.shape == (20, 5)


def run_notes(error):
    return [line for line in error.__notes__ if line.startswith("telescopic:")]


def test_the_note_says_whether_no_replicate_finished_or_their_record_was_lost(monkeypatch):
    # No replicate ran: the initial distribution failed on the first draw.
    failed = RuntimeError("no draw")

    def initial(rng):
        raise failed

    with pytest.raises(RuntimeError, match="no draw"):
        simulate(initial=initial)
    assert failed.telescopic_replicates is None
    assert run_notes(failed) == [
        "telescopic: this error stopped the run before any of its replicates finished"
    ]
    # The replicates finished, but their record failed; the same object raised twice
    # keeps one note.
    lost = MemoryError("no memory")

    def result(*args):
        raise lost

    monkeypatch.setattr(_Batch, "result", result)
    for _ in range(2):
        with pytest.raises(MemoryError):
            simulate()
    assert lost.telescopic_replicates is None
    note = (
        "telescopic: this error stopped the run, and no record of the replicates finished "
        "before it could be made, so they were lost"
    )
    assert run_notes(lost) == [note]


class Momentum(RandomWalk):
    """Sends both chains to 0 and counts in a Markov field: 1 per step alone, and per
    coupled step 1 for X and 2 for Y. After X's L steps alone, the positions agree from the
    first coupled step on, and the counts after L coupled steps."""

    @property
    def markov_fields(self):
        return ("momentum",)

    def init_extra(self, state):
        return {"momentum": np.zeros((len(state), 1))}

    def step(self, target, state, draws):
        x = np.zeros_like(state.x)
        moved = State(x, target.log_prob(x), None, {"momentum": state.extra["momentum"] + 1})
        return moved, np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        moved_y = self.step(target, state_y, draws)[0]
        moved_y.extra["momentum"] = state_y.extra["momentum"] + 2
        return self.step(target, state_x, draws)[0], moved_y


def test_engine_declares_a_meeting_only_when_markov_fields_agree():
    # With lag 2, X steps alone at t = 1 and 2; the positions agree from t = 3 on, and the
    # counts at t = 4 (4 for X, 2 + 2 for Y).
    replicates = simulate(Momentum(np.eye(2)), n=5, k=0, m=6, lag=2, max_iter=20)
    np.testing.assert_array_equal(replicates.meeting_time, 4)


class Forgetful(Momentum):
    def coupled_step(self, target, state_x, state_y, draws):
        moved_x, moved_y = super().coupled_step(target, state_x, state_y, draws)
        return moved_x, State(moved_y.x, moved_y.log_prob)


def test_engine_raises_when_a_step_drops_a_markov_field():
    with pytest.raises(ValueError, match=r"without its Markov fields \['momentum'\]"):
        simulate(Forgetful(np.eye(2)), n=2, max_iter=20)
