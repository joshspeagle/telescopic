import numpy as np
import pytest

from references import P_MIN
from telescopic import examples
from telescopic.kernels import MALA, Gibbs, RandomWalk, State
from telescopic.target import Target
from telescopic.testing import check_coupling, random_draws

GAUSSIAN = examples.gaussian(2, 0.5)
TARGET = Target(GAUSSIAN.log_prob, grad=GAUSSIAN.grad, vectorize=True)
COV = GAUSSIAN.cov
PUMPS = examples.pumps()


# test_kernels runs check_coupling on every built-in coupling, in three dimensions.
def test_check_coupling_passes_the_gibbs_sampler_of_the_pump_model():
    target = Target(PUMPS.log_prob, vectorize=True)
    kernel = Gibbs(PUMPS.conditional, PUMPS.blocks, eta=0.8)
    report = check_coupling(kernel, target, PUMPS.mean, 1.1 * PUMPS.mean, n=500, rng=2)
    assert report.passed(P_MIN), report


class WrongY(RandomWalk):
    # Y moves with twice the kernel's scale.
    def coupled_step(self, target, state_x, state_y, draws):
        wide = RandomWalk(self.cov, scale=2 * self.scale)
        return self.step(target, state_x, draws)[0], wide.step(target, state_y, draws)[0]


class Recomputing(RandomWalk):
    # Where the proposals meet, Y gets X's proposal recomputed, off in the last bits, in
    # every other row; the rest meet.
    def coupled_step(self, target, state_x, state_y, draws):
        new_x, new_y = super().coupled_step(target, state_x, state_y, draws)
        recomputed = np.all(new_x.x == new_y.x, axis=1)
        recomputed[::2] = False
        y = np.where(recomputed[:, None], new_x.x * (1 + 4e-16), new_y.x)
        return new_x, State(y, target.log_prob(y))


class Stale(RandomWalk):
    # Y carries X's log-density.
    def coupled_step(self, target, state_x, state_y, draws):
        new_x, new_y = super().coupled_step(target, state_x, state_y, draws)
        return new_x, State(new_y.x, new_x.log_prob)


class StaleGradient(MALA):
    # Y carries X's gradient.
    def coupled_step(self, target, state_x, state_y, draws):
        new_x, new_y = super().coupled_step(target, state_x, state_y, draws)
        return new_x, State(new_y.x, new_y.log_prob, new_x.grad)


def positive_first(x):
    # A standard normal restricted to x_0 > 0.
    return np.where(x[..., 0] > 0, -0.5 * np.sum(x**2, axis=-1), -np.inf)


class Independent(RandomWalk):
    # Each chain steps alone: the marginals are right, but the chains never meet.
    def coupled_step(self, target, state_x, state_y, draws):
        return self.step(target, state_x, draws)[0], self.step(target, state_y, draws)[0]


def test_check_coupling_flags_a_wrong_marginal():
    report = check_coupling(
        WrongY(COV, scale=1.0), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=3
    )
    assert report.stay[1] < P_MIN < report.stay[0]
    assert report.moves[1] < P_MIN < report.moves[0]


def test_check_coupling_flags_a_coupling_that_recomputes_y_where_the_chains_meet():
    report = check_coupling(Recomputing(COV), TARGET, [0.1, 0.2], [0.2, 0.1], n=2_000, rng=4)
    assert report.near_misses > 0
    assert report.met > 0
    assert not report.passed()


@pytest.mark.parametrize(
    ("kernel", "target", "y"),
    [
        (Stale(COV), TARGET, [0.3, -0.1]),
        (StaleGradient(0.8, cov=COV), TARGET, [0.3, -0.1]),
        # Y stays at a point of zero density, with X's finite log-density.
        (Stale(0.01 * np.eye(2)), Target(positive_first, vectorize=True), [-3.0, 0.0]),
    ],
    ids=["log-density", "gradient", "zero-density"],
)
def test_check_coupling_flags_a_stale_cache(kernel, target, y):
    report = check_coupling(kernel, target, [0.1, 0.2], y, n=2_000, rng=5)
    assert report.stale > 0
    assert not report.passed()


def test_check_coupling_accepts_chains_held_at_points_of_zero_density():
    target = Target(positive_first, vectorize=True)
    kernel = RandomWalk(0.01 * np.eye(2))
    report = check_coupling(kernel, target, [-3.0, 0.0], [-3.0, 0.1], n=100, rng=7)
    assert report.stale == 0


def test_check_coupling_leaves_a_seed_sequence_unchanged():
    seed = np.random.SeedSequence(7)
    first, second = (
        check_coupling(RandomWalk(COV), TARGET, [0.1, 0.2], [0.3, -0.1], n=500, rng=seed)
        for _ in range(2)
    )
    assert first == second


def test_check_coupling_flags_a_coupling_that_never_meets():
    report = check_coupling(Independent(COV), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=6)
    assert report.met == 0
    assert min(*report.stay, *report.moves) > P_MIN
    assert not report.passed()
    with pytest.raises(TypeError, match="alpha must be a real number"):
        report.passed("0.1")


@pytest.mark.parametrize(
    ("x", "y", "n", "error", "message"),
    [
        ([0.0, 0.0], [0.0], 10, ValueError, "x and y must be points"),
        ([0.0], [0.0], 1, ValueError, "n must be at least 2"),
        ([0.0], [0.0], 1e4, TypeError, "n must be an integer, not 10000.0"),
    ],
)
def test_check_coupling_rejects_invalid_arguments(x, y, n, error, message):
    with pytest.raises(error, match=message):
        check_coupling(RandomWalk(np.eye(len(x))), TARGET, x, y, n=n)


@pytest.mark.parametrize(("n_rows", "error"), [(0, ValueError), (2.5, TypeError)])
def test_random_draws_need_a_positive_integer_number_of_rows(n_rows, error):
    with pytest.raises(error, match="n_rows must be"):
        random_draws(n_rows)
