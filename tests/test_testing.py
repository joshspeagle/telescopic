import numpy as np
import pytest

from references import P_MIN
from telescopic import examples
from telescopic.kernels import Gibbs, RandomWalk, State
from telescopic.target import Target
from telescopic.testing import check_coupling

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


def test_check_coupling_flags_a_stale_log_density():
    report = check_coupling(Stale(COV), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=5)
    assert report.stale > 0
    assert not report.passed()


def test_check_coupling_flags_a_coupling_that_never_meets():
    report = check_coupling(Independent(COV), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=6)
    assert report.met == 0
    assert min(*report.stay, *report.moves) > P_MIN
    assert not report.passed()


@pytest.mark.parametrize(
    ("x", "y", "n", "message"),
    [([0.0, 0.0], [0.0], 10, "x and y must be points"), ([0.0], [0.0], 1, "n must be at least 2")],
)
def test_check_coupling_rejects_invalid_arguments(x, y, n, message):
    with pytest.raises(ValueError, match=message):
        check_coupling(RandomWalk(np.eye(len(x))), TARGET, x, y, n=n)
