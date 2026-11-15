import _frozen_kernels as frozen
import numpy as np
import pytest

from references import dense
from telescopic._random import Streams
from telescopic.kernels import State
from telescopic.kernels.metropolis import Maximal, MetropolisHastings, RandomWalkProposal
from telescopic.target import Target


def gaussian(x):
    return -0.5 * np.sum(x * x, axis=-1)


def gaussian_grad(x):
    return -x


def streams(n, seed):
    # Draws as a run makes them: each row reads its normals and its uniforms from two
    # streams of its own, so a change in the order of either kind changes the numbers.
    lanes = Streams(n)
    for row in range(n):
        lanes.assign(row, np.random.default_rng([seed, row]))
    return lanes.view(np.arange(n))


def starts(n, d, seed, grad, target):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    # A tenth of the pairs equal, and a tenth close, so that every branch of a coupling
    # runs.
    y = rng.standard_normal((n, d))
    y[: n // 10] = x[: n // 10]
    y[n // 10 : n // 5] = x[n // 10 : n // 5] + 1e-3 * rng.standard_normal((n // 10, d))
    if grad:
        return State(x, *target.log_prob_and_grad(x)), State(y, *target.log_prob_and_grad(y))
    return State(x, target.log_prob(x)), State(y, target.log_prob(y))


def assert_same_states(new, old):
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.log_prob, b.log_prob)
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            np.testing.assert_array_equal(a.grad, b.grad)


def assert_same_next_numbers(draws, reference):
    # Both read the same count of each kind: their next numbers agree.
    np.testing.assert_array_equal(draws.normal(1), reference.normal(1))
    np.testing.assert_array_equal(draws.uniform(), reference.uniform())
    assert draws.generator(0).random() == reference.generator(0).random()


@pytest.mark.parametrize("vectorize", [False, True])
def test_maximal_coupling_object_reproduces_random_walk_maximal_bitwise(vectorize):
    d, n = 3, 300
    target = Target(gaussian, gaussian_grad, vectorize=vectorize)
    old = frozen.RandomWalk(dense(d), scale=1.3, coupling="maximal")
    new = MetropolisHastings(RandomWalkProposal(*old._factors), Maximal())
    state_x, state_y = starts(n, d, 1, False, target)
    draws, reference = streams(n, 2), streams(n, 2)
    for _ in range(3):
        expected = old.coupled_step(target, state_x, state_y, reference)
        state_x, state_y = new.coupled_step(target, state_x, state_y, draws)
        assert_same_states((state_x, state_y), expected)
        assert_same_next_numbers(draws, reference)
    assert repr(Maximal()) == "Maximal()"
