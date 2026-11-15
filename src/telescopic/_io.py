"""Saving and loading results as versioned ``.npz`` files, without pickle."""

from __future__ import annotations

import json
import os
from typing import IO, Any

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence

from telescopic._types import IntArray
from telescopic.engine import COUNTS, Replicates, Settings
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


def save(result: Result, path: str | os.PathLike[str] | IO[bytes]) -> None:
    replicates = result._replicates
    arrays = {f"replicates.{name}": getattr(replicates, name) for name in COMMON_ARRAYS}
    arrays |= {f"{FAMILY}.{name}": getattr(replicates, name) for name in COUPLED_ARRAYS}
    tuning = result.tuning
    if tuning is not None:
        arrays |= {f"tuning.{name}": getattr(tuning, name) for name in TUNING_ARRAYS}
    if result.pilot is not None:
        arrays["pilot"] = result.pilot
    metadata = {
        "schema": SCHEMA,
        "family": FAMILY,
        "settings": replicates.settings._asdict(),
        "seed": replicates.seed.state,
        "names": result.names,
        "kernel": None if result.kernel is None else str(result.kernel),
        "initial": None if result.initial is None else str(result.initial),
        "counts": {name: getattr(replicates, name) for name in (*COUNTS, "seconds")},
        "k_rule": result._k_rule,
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
    # As np.savez does, a name without the suffix gets it.
    if isinstance(path, str | os.PathLike) and not os.fspath(path).endswith(".npz"):
        path = f"{os.fspath(path)}.npz"
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
        result = cls(
            replicates,
            names=metadata["names"],
            kernel=metadata["kernel"],
            initial=metadata["initial"],
            tuning=tuning,
            pilot=data.get("pilot"),
        )
        result._k_rule = metadata.get("k_rule")
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
