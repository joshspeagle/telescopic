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
