"""Saving and loading results and checkpoints as versioned ``.npz`` files, without pickle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence

from telescopic._types import FloatArray, IntArray
from telescopic.engine import COUNTS, Replicates, Settings, _pairs, merge
from telescopic.estimators import averaging_times, correction_coefficients
from telescopic.results import Result
from telescopic.tuning import TuningReport

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
    # them, the kernel and the initial distribution as their descriptions.
    replicates: Replicates
    names: list[str] | None
    kernel: str | None
    initial: str | None
    tuning: TuningReport | None
    pilot: FloatArray | None
    k_rule: int | None


def save(result: Result, path: str | os.PathLike[str] | IO[bytes]) -> None:
    write(
        path,
        Record(
            result._replicates,
            result.names,
            None if result.kernel is None else str(result.kernel),
            None if result.initial is None else str(result.initial),
            result.tuning,
            result.pilot,
            result._k_rule,
        ),
    )


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
        "kernel": record.kernel,
        "initial": record.initial,
        "counts": {name: getattr(replicates, name) for name in (*COUNTS, "seconds")},
        "k_rule": record.k_rule,
        "tuning": None
        if tuning is None
        else {
            "n_steps": tuning.n_steps,
            "max_weight_ratio": tuning.max_weight_ratio,
            "initial_width": tuning.initial_width,
        },
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
        )


# Spec section 6.7. A checkpoint is a directory of records, each written to a temporary
# file and renamed, so that a killed process leaves whole files only: PLAN, the run's
# seed, settings, tuning and pilot; request-<first>-<last>-<n>.npz, the indices a call
# requested; part-<first>-<last>.npz, finished replicates; abandoned-<first>-<last>.npz,
# the replicates an error abandoned when none finished. Only the plan holds the tuning
# and the pilot.
def empty(seed: SeedSequence, settings: Settings, ndim: int, requested: IntArray) -> Replicates:
    # A record of no replicate, with the shapes of the run's.
    width = 0 if settings.keep is None else ndim
    columns = 0 if settings.keep is None else ndim + len(_pairs(ndim)[0])
    integers = np.zeros(0, dtype=np.int64)
    return Replicates(
        seed=seed,
        settings=settings,
        index=integers,
        meeting_time=np.zeros(0),
        atoms=np.zeros((0, ndim)),
        weights=np.zeros(0),
        owner=integers,
        distance_owner=integers,
        distance_time=integers,
        distance=np.zeros(0),
        zero_density=integers,
        requested=np.asarray(requested, dtype=np.int64),
        center=np.zeros((0, width)),
        moments=np.zeros((0, columns)),
    )


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
            write_atomic(directory / f"{name}.npz", record)
            return


def write_atomic(path: Path, record: Record) -> None:
    # Through a temporary file beside it, synced, then renamed over the name.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as file:
        write(file, record)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def stored(directory: Path, indices: IntArray) -> list[Replicates]:
    # The parts that hold replicates of these indices, each of which must hold only them.
    parts = []
    for path in sorted(directory.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            index = data["replicates.index"]
        inside = np.isin(index, indices)
        if inside.all():
            parts.append(read(path).replicates)
        elif inside.any():
            raise ValueError(
                f"the checkpoint's {path.name} holds replicates both inside and outside the "
                f"indices of this call, {indices[0]} to {indices[-1]}: run the ranges that "
                "wrote it"
            )
    return parts


def load_checkpoint(directory: Path, cls: type[Result]) -> Result:
    if not (directory / PLAN).is_file():
        raise ValueError(f"the directory {directory} holds no {PLAN}, so it is no checkpoint")
    plan = read(directory / PLAN)
    records = [
        read(path).replicates for path in sorted(directory.glob("[!.]*.npz")) if path.name != PLAN
    ]
    replicates = merge([plan.replicates, *records])
    if not len(replicates.index):
        raise ValueError(f"the checkpoint {directory} holds no finished replicate yet")
    return result_of(plan._replace(replicates=replicates), cls)


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
