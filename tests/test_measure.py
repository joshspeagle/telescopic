import numpy as np
import pytest

from telescopic.measure import SignedMeasure

MEASURE = SignedMeasure(
    atoms=np.array([[0.0, 1.0], [2.0, 0.5], [1.0, 1.0], [3.0, -1.0], [4.0, 2.0]]),
    weights=np.array([0.5, 0.5, 1.5, -0.5, 1.0]),
    replicate=np.array([0, 0, 1, 1, 2]),
    n_replicates=3,
)


def test_per_replicate_values_are_weighted_sums_over_each_replicates_atoms():
    np.testing.assert_allclose(MEASURE.per_replicate(lambda x: x[:, 0]), [1.0, 0.0, 4.0])
    np.testing.assert_allclose(
        MEASURE.per_replicate(lambda x: x), [[1.0, 0.75], [0.0, 2.0], [4.0, 2.0]]
    )


@pytest.mark.parametrize("h", [lambda x: 1.0, lambda x: x[:2, 0], lambda x: x[:, :, None]])
def test_a_test_function_of_the_wrong_shape_raises(h):
    with pytest.raises(ValueError, match="h must map"):
        MEASURE.per_replicate(h)


@pytest.mark.parametrize("cell", [[0, 1, 2, 0, 1], [0, 1, 0, -2, 1], [0, 1], [0, 0.5, 1, 0, 1]])
def test_masses_refuse_cells_outside_the_range_or_not_integers(cell):
    # A cell of n_cells or more went into the next replicate's row, and a fractional one
    # was truncated to the cell below.
    with pytest.raises(ValueError, match="cell must give each"):
        MEASURE.mass(np.array(cell), 2)


def test_a_test_function_cannot_change_the_atoms():
    def shift(x):
        x -= 100.0
        return x[:, 0]

    with pytest.raises(ValueError, match="read-only"):
        MEASURE.per_replicate(shift)
    np.testing.assert_allclose(MEASURE.per_replicate(lambda x: x[:, 0]), [1.0, 0.0, 4.0])


@pytest.mark.parametrize(
    ("replicate", "message"),
    [
        ([0.0, 1.5, 1.0], "replicate must give each of the 3 atoms an integer"),
        ([0, 5, 1], "replicate must give"),
        ([0, -1, 1], "replicate must give"),
    ],
)
def test_signed_measure_refuses_replicate_indices_that_are_not_integers_in_range(
    replicate, message
):
    # 1.5 was truncated to replicate 1, and 5 or -1 raised SciPy's message about a matrix.
    with pytest.raises(ValueError, match=message):
        SignedMeasure(np.ones((3, 1)), np.ones(3), np.array(replicate), 2)


def test_signed_measure_rejects_unsorted_replicate():
    # The sums are built from each replicate's range of consecutive atoms, so an unsorted
    # replicate would assign atoms to the wrong replicates.
    with pytest.raises(ValueError, match="replicate must be in nondecreasing order"):
        SignedMeasure(np.ones((3, 1)), np.ones(3), np.array([1, 0, 1]), 2)


@pytest.mark.parametrize(
    ("atoms", "weights", "message"),
    [
        (np.ones((3, 1)), np.ones(2), r"weights must have shape \(3,\)"),
        (np.ones(3), np.ones(3), r"atoms must have shape \(N, d\)"),
    ],
)
def test_signed_measure_refuses_arrays_of_the_wrong_shape(atoms, weights, message):
    # Weights of length 2 for 3 atoms raised SciPy's "all index and data arrays must
    # have the same length" at the first estimate.
    with pytest.raises(ValueError, match=message):
        SignedMeasure(atoms, weights, np.array([0, 1, 1]), 2)


@pytest.mark.parametrize(
    ("atoms", "weights", "name"),
    [
        (np.array([["a"], ["b"], ["c"]]), np.ones(3), "atoms"),
        (np.ones((3, 1)), np.array(["a", "b", "c"]), "weights"),
    ],
)
def test_signed_measure_refuses_arrays_that_are_not_real_numbers(atoms, weights, name):
    # A string array was stored.
    with pytest.raises(TypeError, match=f"^{name} must be real numbers, not an array of dtype"):
        SignedMeasure(atoms, weights, np.array([0, 1, 1]), 2)


def test_signed_measure_arrays_are_read_only():
    # Weights changed after one per_replicate call changed cov() but not per_replicate(),
    # whose sums were cached.
    weights = np.ones(3)
    measure = SignedMeasure(np.ones((3, 1)), weights, np.array([0, 1, 1]), 2)
    for array in (measure.atoms, measure.weights, measure.replicate):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 2
    assert weights.flags.writeable  # the caller's array is left alone
