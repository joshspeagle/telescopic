import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.random import PCG64DXSM, Generator, SeedSequence

from telescopic._random import (
    MAIN,
    PILOT,
    TUNING,
    Streams,
    namespace_seed,
    replicate_generator,
    root_seed,
)
from telescopic.testing import fixed_draws


def test_root_seed_reproduces_from_an_integer():
    assert root_seed(7).state == root_seed(7).state


def test_root_seed_from_none_draws_fresh_entropy():
    assert root_seed(None).entropy != root_seed(None).entropy


def test_root_seed_copies_a_seed_sequence_that_its_owner_spawns_from_later():
    seed = SeedSequence(3)
    root = root_seed(seed)
    seed.spawn(2)
    assert root.state == SeedSequence(3).state


def test_root_seed_from_a_generator_follows_its_state():
    rng = np.random.default_rng(5)
    first, second = root_seed(rng), root_seed(rng)
    assert first.entropy != second.entropy
    assert root_seed(np.random.default_rng(5)).entropy == first.entropy


@pytest.mark.parametrize("rng", [True, 1.5, "seed"])
def test_root_seed_rejects_other_types(rng):
    with pytest.raises(TypeError, match="rng must be"):
        root_seed(rng)


def test_root_seed_rejects_a_negative_integer_by_name():
    with pytest.raises(ValueError, match="rng must be a non-negative integer, not -1"):
        root_seed(-1)


def test_replicate_generator_follows_the_spawn_key_convention():
    root = SeedSequence(11, spawn_key=(2,))
    expected = Generator(PCG64DXSM(SeedSequence(11, spawn_key=(2, 0x74656C, MAIN, 4))))
    assert replicate_generator(root, MAIN, 4).random() == expected.random()
    pilot = SeedSequence(11, spawn_key=(2, 0x74656C, PILOT))
    assert namespace_seed(root, PILOT).state == pilot.state


def test_replicate_streams_avoid_the_callers_own_spawned_streams():
    root = SeedSequence(11)
    grandchild = Generator(PCG64DXSM(root.spawn(1)[0].spawn(1)[0]))
    assert replicate_generator(SeedSequence(11), MAIN, 0).random() != grandchild.random()


def test_namespaces_and_indices_give_different_streams():
    # Main replicates, tuning chains, and the replicates of the meeting-time pilot.
    root = root_seed(1)
    seeds = [(root, MAIN), (root, TUNING), (namespace_seed(root, PILOT), MAIN)]
    draws = {replicate_generator(seed, ns, i).random() for seed, ns in seeds for i in (0, 1)}
    assert len(draws) == 6


N_LANES = 4
requests = st.tuples(
    st.lists(st.integers(0, N_LANES - 1), min_size=1, max_size=N_LANES, unique=True),
    st.booleans(),
    st.integers(1, 40),
)


@given(steps=st.lists(st.one_of(requests, st.integers(0, N_LANES - 1)), max_size=30))
@pytest.mark.parametrize("width", [0, 3, 1024])
def test_each_row_reads_its_replicates_streams_in_order(width, steps):
    # Every row of every request must equal the next numbers of its replicate's two
    # child generators, whatever the buffer width, the order of lanes in the request,
    # the other lanes' requests, and the lanes reassigned in between (an integer step).
    root = root_seed(9)
    streams = Streams(N_LANES, width=width)
    children = {}

    def assign(lane, index):
        streams.assign(lane, replicate_generator(root, MAIN, index))
        children[lane] = replicate_generator(root, MAIN, index).spawn(2)

    for lane in range(N_LANES):
        assign(lane, lane)
    for index, step in enumerate(steps, start=N_LANES):
        if isinstance(step, int):
            assign(step, index)
            continue
        lanes, normal, size = step
        draws = streams.view(lanes)
        if normal:
            taken = draws.normal(size)
            expected = [children[lane][0].standard_normal(size) for lane in lanes]
        else:
            taken = np.stack([draws.uniform() for _ in range(size)], axis=1)
            expected = [children[lane][1].random(size) for lane in lanes]
        np.testing.assert_array_equal(taken, expected)


def test_subset_rows_keep_their_lanes():
    root = root_seed(4)
    streams = Streams(3)
    for lane in range(3):
        streams.assign(lane, replicate_generator(root, MAIN, lane))
    picked = streams.view([2, 0, 1]).subset([0, 2]).normal(2)
    expected = [replicate_generator(root, MAIN, i).spawn(2)[0].standard_normal(2) for i in (2, 1)]
    np.testing.assert_array_equal(picked, expected)


def test_injected_numbers_replay_in_order_then_run_out():
    draws = fixed_draws(normals=[[1.0, 2.0, 3.0]], uniforms=[[0.5]])
    np.testing.assert_array_equal(draws.normal(2), [[1.0, 2.0]])
    assert draws.uniform()[0] == 0.5
    np.testing.assert_array_equal(draws.normal(1), [[3.0]])
    with pytest.raises(IndexError, match="exhausted"):
        draws.uniform()


def test_injected_arrays_must_have_one_row_per_chain():
    with pytest.raises(ValueError, match="same number of rows"):
        fixed_draws(normals=np.zeros((2, 3)), uniforms=np.zeros((3, 1)))


def test_log_uniform_of_zero_is_minus_infinity():
    assert fixed_draws(uniforms=[[0.0]]).log_uniform()[0] == -np.inf


def test_injected_draws_without_generators_refuse_to_serve_one():
    with pytest.raises(RuntimeError, match="no generator"):
        fixed_draws(normals=[[0.0]]).generator(0)
