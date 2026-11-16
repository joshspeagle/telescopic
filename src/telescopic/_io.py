"""Saving and loading results and checkpoints as versioned ``.npz`` files, without pickle."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import IO, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence

from telescopic._types import FloatArray, IntArray
from telescopic.engine import COUNTS, Replicates, Settings, _empty, _seed_key, merge
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.results import Result
from telescopic.tuning import TuningReport
from telescopic.warnings import InputWarning, warn

# The file layout's version; a file of a later version raises on loading.
SCHEMA = 2

# The estimator family of this package's results, which a file names (spec section 7.1).
FAMILY = "coupled"

# The arrays of every family's replicates, saved as "replicates.<name>", and those of the
# coupled family, as "coupled.<name>".
COMMON_ARRAYS = (
    "index",
    "requested",
    "abandoned",
    "atoms",
    "weights",
    "owner",
    "time",
    "center",
    "moments",
)
COUPLED_ARRAYS = ("meeting_time", "distance_owner", "distance_time", "distance", "zero_density")
# Schema 1 saved these, all as "replicates.<name>".
SCHEMA_1_ARRAYS = (
    "index",
    "meeting_time",
    "atoms",
    "weights",
    "owner",
    "distance_owner",
    "distance_time",
    "distance",
    "zero_density",
    "requested",
)
TUNING_ARRAYS = ("mean", "cov", "acceptance", "rhat")

# The plan of a checkpoint directory (spec section 6.7, DECISIONS #269).
PLAN = "plan.npz"


class Record(NamedTuple):
    # What a file holds: replicates of a run, none or more, and what a result holds beside
    # them, the kernel and the initial distribution as their descriptions. A checkpoint's
    # records also hold the version of telescopic that wrote them and the digest of the
    # tuning report under which their replicates ran.
    replicates: Replicates
    names: list[str] | None
    kernel: Any  # an object, or its description as read
    initial: Any
    tuning: TuningReport | None
    pilot: FloatArray | None
    k_rule: int | None
    version: str | None = None
    tuned: str | None = None


def record_of(result: Result) -> Record:
    # A result's record, its kernel and initial distribution kept as they are.
    return Record(
        result._replicates,
        result.names,
        result.kernel,
        result.initial,
        result.tuning,
        result.pilot,
        result._k_rule,
    )


def save(result: Result, path: str | os.PathLike[str] | IO[bytes]) -> None:
    write(path, record_of(result))


def write(path: str | os.PathLike[str] | IO[bytes], record: Record) -> None:
    replicates, tuning = record.replicates, record.tuning
    arrays = {f"replicates.{name}": getattr(replicates, name) for name in COMMON_ARRAYS}
    arrays |= {f"{FAMILY}.{name}": getattr(replicates, name) for name in COUPLED_ARRAYS}
    if tuning is not None:
        arrays |= {f"tuning.{name}": getattr(tuning, name) for name in TUNING_ARRAYS}
    if record.pilot is not None:
        arrays["pilot"] = record.pilot
    metadata = {
        "schema": SCHEMA,
        "family": FAMILY,
        "settings": replicates.settings._asdict(),
        "seed": replicates.seed.state,
        "names": record.names,
        "kernel": None if record.kernel is None else str(record.kernel),
        "initial": None if record.initial is None else str(record.initial),
        "counts": {name: getattr(replicates, name) for name in (*COUNTS, "seconds")},
        "k_rule": record.k_rule,
        "tuning": None
        if tuning is None
        else {
            "n_steps": tuning.n_steps,
            "max_weight_ratio": tuning.max_weight_ratio,
            "initial_width": tuning.initial_width,
        },
        "version": record.version,
        "tuned": record.tuned,
    }
    # NumPy numbers and arrays in the metadata become Python ones, which JSON writes;
    # Python integers keep a seed's entropy of any size exactly.
    text = json.dumps(metadata, default=lambda value: value.tolist())
    np.savez(path, metadata=np.array(text), **arrays)


def load(path: str | os.PathLike[str] | IO[bytes], cls: type[Result]) -> Result:
    if isinstance(path, str | os.PathLike) and os.path.isdir(path):
        return load_checkpoint(Path(path), cls)
    # As np.savez does, a name without the suffix gets it.
    if isinstance(path, str | os.PathLike) and not os.fspath(path).endswith(".npz"):
        path = f"{os.fspath(path)}.npz"
    return result_of(read(path), cls)


def result_of(record: Record, cls: type[Result]) -> Result:
    result = cls(
        record.replicates,
        names=record.names,
        kernel=record.kernel,
        initial=record.initial,
        tuning=record.tuning,
        pilot=record.pilot,
    )
    result._k_rule = record.k_rule
    return result


def read(path: str | os.PathLike[str] | IO[bytes]) -> Record:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data.files:
            raise ValueError("the file is not a saved telescopic result: it has no metadata")
        metadata: dict[str, Any] = json.loads(str(data["metadata"]))
        if metadata["schema"] > SCHEMA:
            raise ValueError(
                f"the file has schema {metadata['schema']}, newer than this telescopic's "
                f"{SCHEMA}: upgrade telescopic to load it"
            )
        # Schema 1 holds no keep, no times, no moments and no abandoned replicates.
        settings = Settings(**metadata["settings"])
        if metadata["schema"] == 1:
            arrays = {name: data[f"replicates.{name}"] for name in SCHEMA_1_ARRAYS}
            arrays["time"] = record_times(
                settings, arrays["index"], arrays["meeting_time"], arrays["owner"]
            )
        else:
            family = metadata["family"]
            if family != FAMILY:
                raise ValueError(
                    f"the file holds a result of the estimator family {family!r}, which this "
                    "telescopic does not know: upgrade telescopic"
                )
            arrays = {name: data[f"replicates.{name}"] for name in COMMON_ARRAYS}
            arrays |= {name: data[f"{FAMILY}.{name}"] for name in COUPLED_ARRAYS}
        replicates = Replicates(
            seed=SeedSequence(**metadata["seed"]),
            settings=settings,
            **arrays,
            **metadata["counts"],
        )
        tuning = metadata["tuning"]
        if tuning is not None:
            tuning = TuningReport(
                **{name: data[f"tuning.{name}"] for name in TUNING_ARRAYS}, **tuning
            )
        return Record(
            replicates,
            metadata["names"],
            metadata["kernel"],
            metadata["initial"],
            tuning,
            data.get("pilot"),
            metadata.get("k_rule"),
            metadata.get("version"),
            metadata.get("tuned"),
        )


# Spec section 6.7. A checkpoint is a directory of records, each written to a temporary
# file and renamed, so that a killed process leaves whole files only: PLAN, the run's
# seed, settings, tuning and pilot, and the indices planned; request-<first>-<last>-<n>.npz,
# the indices a call requested; part-<first>-<last>.npz, finished replicates;
# abandoned-<first>-<last>.npz, the replicates an error abandoned when none finished. Only
# the plan holds the tuning and the pilot.
def digest(tuning: TuningReport | None) -> str | None:
    # The tuning report's mean and covariance, which fix a tuned kernel's numbers.
    if tuning is None:
        return None
    sha = hashlib.sha256(np.ascontiguousarray(tuning.mean).tobytes())
    sha.update(np.ascontiguousarray(tuning.cov).tobytes())
    return sha.hexdigest()[:16]


def version() -> str:
    # Imported here: telescopic imports this module before it sets its version.
    import telescopic

    return telescopic.__version__


def read_plan(directory: Path) -> Record | None:
    path = directory / PLAN
    return read(path) if path.is_file() else None


def write_plan(directory: Path, plan: Record) -> Record:
    # Creates the plan unless the directory holds one already, as when another process
    # made it first, and returns the plan the directory then holds.
    directory.mkdir(parents=True, exist_ok=True)
    temporary = write_temporary(directory, plan._replace(version=version()))
    try:
        os.link(temporary, directory / PLAN)
    except FileExistsError:
        pass
    else:
        sync(directory)
    finally:
        os.unlink(temporary)
    return cast(Record, read_plan(directory))


def write_request(directory: Path, plan: Record, indices: IntArray) -> None:
    # The record of the indices a call requested, written before it runs any.
    ndim = plan.replicates.atoms.shape[1]
    request = _empty(plan.replicates.seed, plan.replicates.settings, ndim, np.sort(indices))
    write_record(directory, Record(request, None, None, None, None, None, None))


def write_record(directory: Path, record: Record) -> None:
    # A part, an abandoned record or a request, named after its indices; nothing for a
    # record without any.
    replicates = record.replicates
    for kind, indices in (
        ("part", replicates.index),
        ("abandoned", replicates.abandoned),
        ("request", replicates.requested),
    ):
        if len(indices):
            name = f"{kind}-{indices[0]}-{indices[-1]}"
            if kind == "request":
                name += f"-{len(indices)}"
            os.replace(write_temporary(directory, record), directory / f"{name}.npz")
            sync(directory)
            return


def write_temporary(directory: Path, record: Record) -> str:
    # A synced file of a unique name in the directory, hidden from its records.
    handle, name = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    if os.name == "posix":
        # mkstemp makes the file private; a record gets the permissions of a new file.
        os.fchmod(handle, 0o666 & ~umask())
    with os.fdopen(handle, "wb") as file:
        write(file, record)
        file.flush()
        os.fsync(file.fileno())
    return name


@functools.cache
def umask() -> int:
    # The process's umask, which can only be read by setting it, once.
    mask = os.umask(0)
    os.umask(mask)
    return mask


def check_links(directory: Path) -> None:
    # The plan is created by a hard link, which some filesystems lack.
    directory.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    os.close(handle)
    link = f"{name}.link"
    try:
        os.link(name, link)
    except OSError as error:
        raise ValueError(
            f"checkpoint {directory} is on a filesystem without hard links ({error}), with "
            "which a plan is created by one process only: use a directory on another "
            "filesystem"
        ) from error
    else:
        os.unlink(link)
    finally:
        os.unlink(name)


def sync(directory: Path) -> None:
    # The directory's entries, so that a renamed record survives a crash of the machine;
    # a no-op where directories cannot be opened, as on Windows.
    if os.name == "posix":
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def stored(directory: Path, plan: Record, indices: IntArray) -> list[Replicates]:
    # The parts that hold replicates of these indices, each of the plan's run and holding
    # only them.
    parts = []
    for path in sorted(directory.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            index = data["replicates.index"]
        inside = np.isin(index, indices)
        if not inside.any():
            continue
        if not inside.all():
            raise ValueError(
                f"checkpoint {directory}: {path.name} holds replicates both inside and "
                f"outside the indices of this call, {indices[0]} to {indices[-1]}; run the "
                "ranges that wrote it"
            )
        part = read(path).replicates
        if not same_run(part, plan.replicates):
            raise ValueError(
                f"checkpoint {directory}: {path.name} holds replicates of another run, whose "
                "seed or settings differ from the plan's"
            )
        parts.append(part)
    return parts


def same_run(a: Replicates, b: Replicates) -> bool:
    return a.settings == b.settings and _seed_key(a.seed) == _seed_key(b.seed)


def load_checkpoint(directory: Path, cls: type[Result]) -> Result:
    plan = read_plan(directory)
    if plan is None:
        raise ValueError(f"the directory {directory} holds no {PLAN}, so it is no checkpoint")
    records = [read(path) for path in sorted(directory.glob("[!.]*.npz")) if path.name != PLAN]
    replicates = merge([plan.replicates, *(record.replicates for record in records)])
    if not len(replicates.index):
        raise ValueError(f"the checkpoint {directory} holds no finished replicate yet")
    result = result_of(plan._replace(replicates=replicates), cls)
    # Parts whose replicates ran under another tuning than the plan's (DECISIONS #267).
    tuned_apart = [
        record for record in records if len(record.replicates.index) and record.tuned != plan.tuned
    ]
    result._tuned_apart = len(tuned_apart)
    versions = [
        record
        for record in records
        if len(record.replicates.index) and record.version != plan.version
    ]
    result._versions_apart = len(versions)
    if versions:
        warn(
            f"{len(versions)} of the parts of checkpoint {directory} ran under another "
            f"version of telescopic than the plan's {plan.version}: their replicates can "
            "differ from those of the plan's version, so the run does not repeat bit for bit",
            InputWarning,
        )
    if tuned_apart:
        warn(
            f"{len(tuned_apart)} of the parts of checkpoint {directory} ran with a kernel tuned "
            "differently from the plan's, as on another type of CPU, BLAS or library "
            "version: each replicate stays unbiased, but the run does not repeat bit for bit",
            InputWarning,
        )
    return result


def record_times(
    settings: Settings, index: IntArray, meeting_time: npt.NDArray[np.float64], owner: IntArray
) -> npt.NDArray[np.int32]:
    # The times of the atoms of a schema-1 file, which the engine's record order
    # fixes (spec R-EST-6): for each replicate that met, in index order, t for each t in S
    # and t twice for each t in [L, tau) with c_t > 0, by t and the averaged atom first.
    # O(N), with no sort: each atom's position within its replicate is computed.
    k, m, lag, thin = settings.k, settings.m, settings.lag, settings.thin
    met = np.isfinite(meeting_time)
    tau = meeting_time[met].astype(np.int64)
    averaged = averaging_times(k, m, thin)
    span = np.arange(lag, max(lag, int(tau.max(initial=lag))))
    corrected = span[correction_coefficients(span, k, m, lag, thin) > 0]
    n = np.searchsorted(corrected, tau)  # each replicate's times with corrections
    sizes = len(averaged) + 2 * n
    if not np.array_equal(np.repeat(index[met], sizes), owner):
        raise ValueError(
            f"the file's atoms do not follow the record order of their settings: {len(owner)} "
            f"atoms where the meeting times give {sizes.sum()}"
        )
    first = np.cumsum(sizes) - sizes
    time = np.empty(len(owner), dtype=np.int32)
    # The averaged atom at S_j follows j averaged atoms and two for each earlier correction.
    before = np.minimum(np.searchsorted(corrected, averaged)[None, :], n[:, None])
    time[(first[:, None] + np.arange(len(averaged)) + 2 * before).ravel()] = np.tile(
        averaged, len(tau)
    )
    # The pair of correction j follows 2 j correction atoms and the averaged ones up to it.
    replicate = np.repeat(np.arange(len(tau)), n)
    j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    position = first[replicate] + 2 * j + np.searchsorted(averaged, corrected[j], side="right")
    time[position] = time[position + 1] = corrected[j]
    return time
