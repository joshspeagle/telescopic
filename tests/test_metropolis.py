import pickle

import _frozen_kernels as frozen
import numpy as np
import pytest

from references import dense
from telescopic.couplings import Maximal, Reflection, Shared
from telescopic.kernels import (
    MALA,
    Independence,
    IndependenceProposal,
    LangevinProposal,
    MetropolisHastings,
    MultivariateT,
    RandomWalk,
    RandomWalkProposal,
    State,
)
from telescopic.target import Target
from telescopic.testing import fixed_draws


def gaussian(x):
    return -0.5 * np.sum(x * x, axis=-1)


def gaussian_grad(x):
    return -x


def twin_draws(n, d, seed):
    # Two copies of the same draws, as a run reads them: each row's normals and uniforms
    # are two sequences of their own, so a change in the order of either kind changes
    # the numbers, and each row has a generator of its own for the maximal coupling.
    rng = np.random.default_rng(seed)
    normals, uniforms = rng.standard_normal((n, 4 * d + 25)), rng.random((n, 8))
    return [
        fixed_draws(normals, uniforms, [np.random.default_rng([seed, row]) for row in range(n)])
        for _ in range(2)
    ]


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
    draws, reference = twin_draws(n, d, 2)
    for _ in range(3):
        expected = old.coupled_step(target, state_x, state_y, reference)
        state_x, state_y = new.coupled_step(target, state_x, state_y, draws)
        assert_same_states((state_x, state_y), expected)
        assert_same_next_numbers(draws, reference)
    assert repr(Maximal()) == "Maximal()"


def bounded_target(d, vectorize):
    # N(0, Sigma) cut at x_0 = -1: a log-density of -inf on part of the space, and a BLAS
    # product in a vectorized call.
    precision = np.linalg.inv(dense(d))

    def log_prob(x):
        return np.where(x[..., 0] > -1.0, -0.5 * np.sum((x @ precision) * x, axis=-1), -np.inf)

    def grad(x):
        return np.where(x[..., :1] > -1.0, -(x @ precision), 0.0)

    return Target(log_prob, grad, vectorize=vectorize)


def shipped_and_composed(name, d):
    # The frozen kernel, the shipped kernel, and the same kernel composed from a proposal
    # and a coupling.
    cov = dense(d)
    if name in ("reflection", "maximal"):
        old = frozen.RandomWalk(cov, scale=1.0, coupling=name)
        coupling = Reflection() if name == "reflection" else Maximal()
        composed = MetropolisHastings(RandomWalkProposal(*old._factors), coupling)
        return old, RandomWalk(cov, scale=1.0, coupling=name), composed
    if name == "mala":
        old = frozen.MALA(0.9, cov=cov)
        composed = MetropolisHastings(LangevinProposal(*old._factors), Reflection())
        return old, MALA(0.9, cov=cov), composed
    q = MultivariateT(np.zeros(d), 1.5 * cov)
    old = frozen.Independence(q)
    return old, Independence(q), MetropolisHastings(IndependenceProposal(q), Shared())


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("d", [1, 12])
@pytest.mark.parametrize(
    ("name", "grad"),
    # MALA's states always carry gradients; the others' carry them when a gradient kernel
    # shares their states.
    [(name, grad) for name in ("reflection", "maximal", "independence") for grad in (False, True)]
    + [("mala", True)],
)
def test_metropolis_hastings_reproduces_shipped_kernels_bitwise(name, grad, d, vectorize):
    n = 3000
    target = bounded_target(d, vectorize)
    old, shipped, composed = shipped_and_composed(name, d)
    state_x, state_y = starts(n, d, 3, grad, target)
    rng = np.random.default_rng(4)
    for state in (state_x, state_y):
        # A tenth of the starts outside the support, some in both chains of a pair.
        outside = rng.random(n) < 0.1
        state.x[outside, 0] = -1.5
        state.log_prob[outside] = -np.inf
        if grad:
            state.grad[outside] = 0.0
    for kernel in (shipped, composed):
        x, y = state_x, state_y
        draws, reference = twin_draws(n, d, 5)
        for _ in range(2):
            moved, probability = kernel.step(target, x, draws)
            expected, expected_probability = old.step(target, x, reference)
            assert_same_states((moved,), (expected,))
            np.testing.assert_array_equal(probability, expected_probability)
            expected = old.coupled_step(target, x, y, reference)
            x, y = kernel.coupled_step(target, x, y, draws)
            assert_same_states((x, y), expected)
            assert_same_next_numbers(draws, reference)


@pytest.mark.parametrize(("name", "coupling"), [("reflection", Reflection), ("maximal", Maximal)])
def test_random_walk_gives_the_same_bits_with_a_coupling_object_and_its_string(name, coupling):
    d, n = 3, 200
    target = bounded_target(d, False)
    by_name = RandomWalk(dense(d), coupling=name)
    by_object = RandomWalk(dense(d), coupling=coupling())
    state_x, state_y = starts(n, d, 6, False, target)
    draws, reference = twin_draws(n, d, 7)
    for _ in range(3):
        expected = by_name.coupled_step(target, state_x, state_y, reference)
        state_x, state_y = by_object.coupled_step(target, state_x, state_y, draws)
        assert_same_states((state_x, state_y), expected)
    assert_same_next_numbers(draws, reference)
    assert isinstance(by_object.coupling, coupling)
    assert by_name.coupling == name


class ShiftedWalk(RandomWalkProposal):
    """A subclass whose draw differs from its parent's: Reflection's coupled step would
    not have its law."""

    def sample(self, target, state, draws):
        x, forward = super().sample(target, state, draws)
        return x + 1.0, forward


@pytest.mark.parametrize(
    ("proposal", "coupling"),
    [
        (LangevinProposal(np.eye(2), np.eye(2)), Maximal()),
        (LangevinProposal(np.eye(2), np.eye(2)), Shared()),
        (RandomWalkProposal(np.eye(2), np.eye(2)), Shared()),
        (IndependenceProposal(MultivariateT(np.zeros(2), np.eye(2))), Reflection()),
        (IndependenceProposal(MultivariateT(np.zeros(2), np.eye(2))), Maximal()),
        (ShiftedWalk(np.eye(2), np.eye(2)), Reflection()),
        (ShiftedWalk(np.eye(2), np.eye(2)), Maximal()),
    ],
)
def test_metropolis_hastings_refuses_a_coupling_for_a_proposal_or_subclass_it_does_not_support(
    proposal, coupling
):
    with pytest.raises(ValueError, match="does not couple"):
        MetropolisHastings(proposal, coupling)


def test_random_walk_refuses_a_coupling_object_of_another_proposal():
    with pytest.raises(ValueError, match="coupling must be"):
        RandomWalk(np.eye(2), coupling=Shared())


def test_kernels_with_coupling_objects_survive_pickling():
    d, n = 2, 50
    target = bounded_target(d, False)
    q = MultivariateT(np.zeros(d), 2 * np.eye(d))
    kernels = [
        RandomWalk(np.eye(d), coupling=Maximal()),
        RandomWalk(np.eye(d), coupling=Reflection()),
        MALA(0.5, cov=np.eye(d)),
        Independence(q),
        MetropolisHastings(RandomWalkProposal(np.eye(d), np.eye(d)), Maximal()),
        MetropolisHastings(IndependenceProposal(q), Shared()),
    ]
    for kernel in kernels:
        copy = pickle.loads(pickle.dumps(kernel))
        assert repr(copy) == repr(kernel)
        grad = kernel.needs_grad
        state_x, state_y = starts(n, d, 8, grad, target)
        draws, reference = twin_draws(n, d, 9)
        expected = kernel.coupled_step(target, state_x, state_y, reference)
        assert_same_states(copy.coupled_step(target, state_x, state_y, draws), expected)
