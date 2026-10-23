import io
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import telescopic
from references import gaussian_log_prob
from telescopic import _io
from telescopic.engine import run
from telescopic.initial import Normal
from telescopic.kernels import RandomWalk
from telescopic.results import Result
from telescopic.target import Target
from telescopic.warnings import ConvergenceWarning

INIT = Normal([2.0, -1.0], [4.0, 4.0])


@pytest.fixture(
    scope="module", params=[3, np.random.default_rng(4), None], ids=["int", "generator", "bare"]
)
def result(request):
    if request.param is None:
        # From the engine alone: no kernel, initial distribution, tuning or pilot.
        replicates = run(
            Target(gaussian_log_prob),
            RandomWalk(np.eye(2)),
            INIT,
            range(5),
            rng=5,
            k=2,
            m=6,
            lag=1,
        )
        return Result(replicates)
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


def assert_same(loaded, original, name):
    # Compares seeds by their state, and the fields of the replicates, the measure and the
    # tuning report one by one.
    if isinstance(original, np.random.SeedSequence):
        assert loaded.state == original.state, name
    elif hasattr(original, "__dataclass_fields__"):
        for field, value in vars(original).items():
            if field != "paths":
                assert_same(getattr(loaded, field), value, f"{name}.{field}")
    else:
        np.testing.assert_array_equal(loaded, original, err_msg=name)


def test_save_and_load_round_trip_every_field(result):
    loaded = reloaded(result)
    assert vars(loaded).keys() == vars(result).keys()
    for name, value in vars(result).items():
        if name in ("kernel", "initial"):
            assert getattr(loaded, name) == (None if value is None else str(value))
        else:
            assert_same(getattr(loaded, name), value, name)
    assert repr(loaded.summary()) == repr(result.summary())


def test_a_name_without_the_suffix_loads_what_saving_wrote(result):
    # NumPy appends .npz when saving; the directory is inside the repository, and removed.
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "run"
        result.save(path)
        np.testing.assert_array_equal(Result.load(path).meeting_times, result.meeting_times)


def test_a_file_from_a_later_schema_raises(result, monkeypatch):
    monkeypatch.setattr(_io, "SCHEMA", _io.SCHEMA + 1)
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
        for name in data.files:
            data[name]  # raises for a pickled array
