import io
import warnings

import numpy as np
import pytest

import telescopic
from references import gaussian_log_prob
from telescopic import io as telescopic_io
from telescopic.initial import Normal
from telescopic.results import Result
from telescopic.warnings import ConvergenceWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])


@pytest.fixture(scope="module", params=[3, np.random.default_rng(4)], ids=["int", "generator"])
def result(request):
    # Tuned, with a pilot and censored pairs; a generator gives a seed of four integers.
    sampler = telescopic.CoupledSampler(gaussian_log_prob, INIT, rng=request.param)
    sampler.meeting_times(20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # the censored pairs
        result = sampler.run(20, k=2, m=10, lag=2, max_iter=12)
    assert result.n_censored
    return result


def reloaded(result):
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    return Result.load(buffer)


def test_save_and_load_round_trip_every_field(result):
    loaded = reloaded(result)
    for name, value in vars(result._replicates).items():
        if name != "paths":
            np.testing.assert_array_equal(getattr(loaded._replicates, name), value, err_msg=name)
    for name in ("k", "m", "lag", "thin", "max_iter", "names", "interrupted"):
        assert getattr(loaded, name) == getattr(result, name)
    np.testing.assert_array_equal(loaded.pilot, result.pilot)
    for name, value in vars(result.tuning).items():
        np.testing.assert_array_equal(getattr(loaded.tuning, name), value, err_msg=name)
    assert loaded.seed.entropy == result.seed.entropy
    assert loaded.seed.spawn_key == result.seed.spawn_key
    assert (loaded.kernel, loaded.initial) == (repr(result.kernel), repr(result.initial))
    np.testing.assert_array_equal(loaded.mean().value, result.mean().value)
    assert repr(loaded.summary()) == repr(result.summary())


def test_a_file_from_a_later_schema_raises(result, monkeypatch):
    monkeypatch.setattr(telescopic_io, "SCHEMA", telescopic_io.SCHEMA + 1)
    buffer = io.BytesIO()
    result.save(buffer)
    monkeypatch.undo()
    buffer.seek(0)
    with pytest.raises(ValueError, match="newer than this telescopic"):
        Result.load(buffer)


def test_saved_files_hold_no_pickled_objects(result):
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
