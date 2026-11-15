import dataclasses
import io
import json
import tempfile
import warnings
from functools import cached_property
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
    # tuning report one by one; cached properties are not fields.
    if isinstance(original, np.random.SeedSequence):
        assert loaded.state == original.state, name
    elif dataclasses.is_dataclass(original):
        for field in dataclasses.fields(original):
            if field.name != "paths":
                value = getattr(original, field.name)
                assert_same(getattr(loaded, field.name), value, f"{name}.{field.name}")
    else:
        np.testing.assert_array_equal(loaded, original, err_msg=name)


def stored(result):
    # A result's attributes, without the values cached from them.
    cached = {name for name, member in vars(Result).items() if isinstance(member, cached_property)}
    return {name: value for name, value in vars(result).items() if name not in cached}


def test_save_and_load_round_trip_every_field(result):
    loaded = reloaded(result)
    assert stored(loaded).keys() == stored(result).keys()
    for name, value in stored(result).items():
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


def test_a_file_that_is_not_a_saved_result_raises():
    buffer = io.BytesIO()
    np.savez(buffer, x=np.zeros(3))
    buffer.seek(0)
    with pytest.raises(ValueError, match="not a saved telescopic result"):
        Result.load(buffer)


def test_saved_files_hold_no_pickled_objects(result):
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as data:
        for name in data.files:
            data[name]  # raises for a pickled array


def reduced_result():
    # A run that keeps replicates 0 to 6, with censored pairs.
    sampler = telescopic.CoupledSampler(
        gaussian_log_prob, INIT, kernel=RandomWalk(np.eye(2)), rng=6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # the censored pairs
        result = sampler.run(20, k=2, m=10, lag=2, max_iter=12, keep=7)
    assert result.n_censored
    assert 0 < result.n_kept < result.n_replicates
    return result


def test_reduced_results_round_trip_bitwise_through_a_file():
    result = reduced_result()
    loaded = reloaded(result)
    for name, value in stored(result).items():
        if name not in ("kernel", "initial"):
            assert_same(getattr(loaded, name), value, name)
    assert loaded._replicates.moments.shape == (20, 5)
    assert repr(loaded.summary()) == repr(result.summary())


def schema_1_file(result):
    # The file that schema 1 wrote for a result without keep: no family, no times, no
    # moments and no abandoned replicates, every array under "replicates.".
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    with np.load(buffer) as data:
        metadata = json.loads(str(data["metadata"]))
        arrays = {name: data[name] for name in data.files if name != "metadata"}
    del metadata["family"], metadata["settings"]["keep"]
    metadata["schema"] = 1
    for name in _io.COMMON_ARRAYS:
        array = arrays.pop(f"replicates.{name}")
        if name in _io.SCHEMA_1_ARRAYS:
            arrays[f"replicates.{name}"] = array
    for name in _io.COUPLED_ARRAYS:
        arrays[f"replicates.{name}"] = arrays.pop(f"coupled.{name}")
    old = io.BytesIO()
    np.savez(old, metadata=np.array(json.dumps(metadata)), **arrays)
    old.seek(0)
    return old


def test_schema_1_files_load_with_the_times_a_new_run_records(result):
    loaded = Result.load(schema_1_file(result))
    np.testing.assert_array_equal(loaded._replicates.time, result._replicates.time)
    assert loaded.keep is None
    assert loaded._replicates.moments.shape == (len(result.index), 0)
    assert not len(loaded.abandoned)
    assert repr(loaded.summary()) == repr(result.summary())


def test_a_file_without_the_initial_width_loads_it_as_none(result):
    # Files written before the warning about a wide start hold neither field.
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    with np.load(buffer) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    del metadata["k_rule"]
    if metadata["tuning"] is not None:
        del metadata["tuning"]["initial_width"]
    arrays["metadata"] = np.array(json.dumps(metadata))
    old = io.BytesIO()
    np.savez(old, **arrays)
    old.seek(0)
    loaded = Result.load(old)
    assert loaded._k_rule is None
    assert result.tuning is None or loaded.tuning.initial_width is None


def test_a_file_of_an_unknown_family_raises_naming_it(result):
    buffer = io.BytesIO()
    result.save(buffer)
    buffer.seek(0)
    with np.load(buffer) as data:
        arrays = {name: data[name] for name in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["family"] = "multilevel"
    arrays["metadata"] = np.array(json.dumps(metadata))
    renamed = io.BytesIO()
    np.savez(renamed, **arrays)
    renamed.seek(0)
    with pytest.raises(ValueError, match="estimator family 'multilevel', which this telescopic"):
        Result.load(renamed)
