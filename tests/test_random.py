import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.random import PCG64DXSM, Generator, SeedSequence

from telescopic._random import MAIN, PILOT, Streams, replicate_generator, root_seed


def test_root_seed_reproduces_from_an_integer():
    first, second = root_seed(7), root_seed(7)
    assert first.entropy == second.entropy
    assert first.spawn_key == second.spawn_key


def test_root_seed_from_none_draws_fresh_entropy():
    assert root_seed(None).entropy != root_seed(None).entropy


def test_root_seed_keeps_a_seed_sequence():
    seed = SeedSequence(3)
    assert root_seed(seed) is seed


def test_root_seed_from_the_same_generator_twice_differs():
    rng = np.random.default_rng(5)
    first, second = root_seed(rng), root_seed(rng)
    assert (first.entropy, first.spawn_key) != (second.entropy, second.spawn_key)


@pytest.mark.parametrize("rng", [True, 1.5, "seed"])
def test_root_seed_rejects_other_types(rng):
    with pytest.raises(TypeError, match="rng must be"):
        root_seed(rng)


def test_replicate_generator_follows_the_spawn_key_convention():
    root = SeedSequence(11, spawn_key=(2,))
    expected = Generator(PCG64DXSM(SeedSequence(11, spawn_key=(2, MAIN, 4))))
    assert replicate_generator(root, MAIN, 4).random() == expected.random()


def test_namespaces_and_indices_give_different_streams():
    root = root_seed(1)
    draws = {
        (ns, i): replicate_generator(root, ns, i).random() for ns in (MAIN, PILOT) for i in (0, 1)
    }
    assert len(set(draws.values())) == 4


requests = st.lists(st.tuples(st.booleans(), st.integers(1, 40)), min_size=1, max_size=30)


@given(
    target=requests,
    others=st.lists(requests, max_size=4),
    position=st.integers(0, 4),
    block=st.integers(1, 6),
)
def test_a_lanes_numbers_depend_only_on_its_generator_and_requests(
    target, others, position, block
):
    # The lane of interest shares a batch with other lanes that make their own requests
    # in between; its numbers must equal those of a lone lane with the same generator.
    ndim = 3
    position = min(position, len(others))
    batch = Streams(len(others) + 1, ndim, block=block)
    lanes = list(range(len(others) + 1))
    target_lane = lanes.pop(position)
    batch.assign(target_lane, replicate_generator(root_seed(9), MAIN, 0))
    for lane in lanes:
        batch.assign(lane, replicate_generator(root_seed(9), MAIN, lane + 1))
    alone = Streams(1, ndim, block=block)
    alone.assign(0, replicate_generator(root_seed(9), MAIN, 0))

    def ask(draws, normal, size):
        return (
            draws.normal(size) if normal else np.stack([draws.uniform() for _ in range(size)], 1)
        )

    for step, (normal, size) in enumerate(target):
        for lane, other in zip(lanes, others, strict=True):
            if step < len(other):
                ask(batch.view([lane]), *other[step])
        together = ask(batch.view([target_lane, *lanes]), normal, size)[0]
        np.testing.assert_array_equal(together, ask(alone.view([0]), normal, size)[0])


def test_subset_rows_keep_their_lanes():
    streams = Streams(3, 2)
    for lane in range(3):
        streams.assign(lane, replicate_generator(root_seed(4), MAIN, lane))
    reference = Streams(3, 2)
    for lane in range(3):
        reference.assign(lane, replicate_generator(root_seed(4), MAIN, lane))
    picked = streams.view([2, 0, 1]).subset([0, 2]).normal(2)
    full = reference.view([0, 1, 2]).normal(2)
    np.testing.assert_array_equal(picked, full[[2, 1]])


def test_requests_larger_than_a_block_are_reproducible():
    first, second = Streams(1, 2, block=2), Streams(1, 2, block=2)
    for streams in (first, second):
        streams.assign(0, replicate_generator(root_seed(8), MAIN, 0))
    big = [s.view([0]).normal(50) for s in (first, second)]
    np.testing.assert_array_equal(big[0], big[1])
    assert big[0].shape == (1, 50)


def test_injected_numbers_replay_in_order_then_run_out():
    draws = Streams.from_arrays(normals=[[1.0, 2.0, 3.0]], uniforms=[[0.5]]).view([0])
    np.testing.assert_array_equal(draws.normal(2), [[1.0, 2.0]])
    assert draws.uniform()[0] == 0.5
    np.testing.assert_array_equal(draws.normal(1), [[3.0]])
    with pytest.raises(IndexError, match="exhausted"):
        draws.uniform()


def test_log_uniform_of_zero_is_minus_infinity():
    draws = Streams.from_arrays(uniforms=[[0.0]]).view([0])
    assert draws.log_uniform()[0] == -np.inf


def test_a_lane_without_a_generator_raises():
    with pytest.raises(RuntimeError, match="no generator"):
        Streams(1, 1).view([0]).normal(1)
