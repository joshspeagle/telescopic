import numpy as np
import pytest

from references import P_MIN
from telescopic import examples
from telescopic.kernels import (
    HMC,
    MALA,
    Gibbs,
    Independence,
    Mixture,
    MultivariateT,
    RandomWalk,
    State,
)
from telescopic.target import Target
from telescopic.testing import check_coupling

GAUSSIAN = examples.gaussian(2, 0.5)
TARGET = Target(GAUSSIAN.log_prob, grad=GAUSSIAN.grad, vectorize=True)
COV = GAUSSIAN.cov
PUMPS = examples.pumps()


@pytest.mark.parametrize(
    "kernel",
    [
        RandomWalk(COV),
        MALA(0.8, cov=COV),
        HMC(0.3, 4, cov=COV, meeting_weight=0.3, meeting_scale=0.5),
        Independence(MultivariateT(np.zeros(2), 1.5 * COV, df=None)),
        Mixture([RandomWalk(COV, scale=0.5), MALA(0.8, cov=COV)], [0.5, 0.5]),
    ],
    ids=["rwmh", "mala", "hmc", "independence", "mixture"],
)
def test_check_coupling_passes_the_built_in_kernels(kernel):
    report = check_coupling(kernel, TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=1)
    assert report.passed(P_MIN), report


def test_check_coupling_passes_gibbs_samplers():
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
    # Where the proposals meet, Y gets X's proposal recomputed, off in the last bits.
    def coupled_step(self, target, state_x, state_y, draws):
        new_x, new_y = super().coupled_step(target, state_x, state_y, draws)
        met = np.all(new_x.x == new_y.x, axis=1)
        y = np.where(met[:, None], new_x.x * (1 + 4e-16), new_y.x)
        return new_x, State(y, target.log_prob(y))


class Independent(RandomWalk):
    # Each chain steps alone: the marginals are right, but equal states split.
    def coupled_step(self, target, state_x, state_y, draws):
        return self.step(target, state_x, draws)[0], self.step(target, state_y, draws)[0]


def test_check_coupling_flags_a_wrong_marginal():
    report = check_coupling(
        WrongY(COV, scale=1.0), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=3
    )
    assert report.moves[1] < P_MIN
    assert report.moves[0] > P_MIN


def test_check_coupling_flags_a_coupling_that_recomputes_y_where_the_chains_meet():
    report = check_coupling(Recomputing(COV), TARGET, [0.1, 0.2], [0.2, 0.1], n=2_000, rng=4)
    assert report.near_misses > 0
    assert not report.passed()


def test_check_coupling_flags_a_coupling_that_splits_equal_states():
    report = check_coupling(Independent(COV), TARGET, [0.1, 0.2], [0.3, -0.1], n=2_000, rng=5)
    assert not report.faithful
    assert min(*report.stay, *report.moves) > P_MIN


@pytest.mark.parametrize(
    ("x", "y", "n", "message"),
    [([0.0, 0.0], [0.0], 10, "x and y must be points"), ([0.0], [0.0], 1, "n must be at least 2")],
)
def test_check_coupling_rejects_invalid_arguments(x, y, n, message):
    with pytest.raises(ValueError, match=message):
        check_coupling(RandomWalk(np.eye(len(x))), TARGET, x, y, n=n)
