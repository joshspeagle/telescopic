"""Saving and loading results as versioned ``.npz`` files, without pickle."""

from __future__ import annotations

import json
import os
from typing import IO, Any

import numpy as np
from numpy.random import SeedSequence

from telescopic.engine import Replicates
from telescopic.results import Result
from telescopic.tuning import TuningReport

# The file layout's version; a file of a later version raises on loading.
SCHEMA = 1

# The arrays of a run's replicates, saved as "replicates.<name>".
REPLICATE_ARRAYS = (
    "index",
    "meeting_time",
    "cost",
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
    arrays = {f"replicates.{name}": getattr(replicates, name) for name in REPLICATE_ARRAYS}
    tuning = result.tuning
    if tuning is not None:
        arrays |= {f"tuning.{name}": getattr(tuning, name) for name in TUNING_ARRAYS}
    if result.pilot is not None:
        arrays["pilot"] = result.pilot
    metadata = {
        "schema": SCHEMA,
        "settings": {
            name: getattr(result, name) for name in ("k", "m", "lag", "thin", "max_iter")
        },
        "names": result.names,
        "kernel": None if result.kernel is None else str(result.kernel),
        "initial": None if result.initial is None else str(result.initial),
        "seed": None if result.seed is None else result.seed.state,
        "counts": {
            "n_log_prob": replicates.n_log_prob,
            "n_grad": replicates.n_grad,
            "seconds": replicates.seconds,
        },
        "tuning": None
        if tuning is None
        else {"n_steps": tuning.n_steps, "max_weight_ratio": tuning.max_weight_ratio},
    }
    # NumPy numbers and arrays in the metadata become Python ones, which JSON writes;
    # Python integers keep a seed's entropy of any size exactly.
    text = json.dumps(metadata, default=lambda value: value.tolist())
    np.savez(path, metadata=np.array(text), **arrays)


def load(path: str | os.PathLike[str] | IO[bytes], cls: type[Result]) -> Result:
    with np.load(path, allow_pickle=False) as data:
        metadata: dict[str, Any] = json.loads(str(data["metadata"]))
        if metadata["schema"] > SCHEMA:
            raise ValueError(
                f"the file has schema {metadata['schema']}, newer than this telescopic's "
                f"{SCHEMA}: upgrade telescopic to load it"
            )
        replicates = Replicates(
            **{name: data[f"replicates.{name}"] for name in REPLICATE_ARRAYS},
            **metadata["counts"],
        )
        tuning = metadata["tuning"]
        if tuning is not None:
            tuning = TuningReport(
                **{name: data[f"tuning.{name}"] for name in TUNING_ARRAYS}, **tuning
            )
        seed = metadata["seed"]
        return cls(
            replicates,
            **metadata["settings"],
            names=metadata["names"],
            kernel=metadata["kernel"],
            initial=metadata["initial"],
            seed=None if seed is None else SeedSequence(**seed),
            tuning=tuning,
            pilot=data.get("pilot"),
        )
