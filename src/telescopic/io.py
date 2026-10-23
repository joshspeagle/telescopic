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
    """Save a result to a ``.npz`` file that :func:`load` reads (spec R-OUT-1).

    The kernel and the initial distribution are saved as descriptions, their ``repr``;
    the engine's paths are not saved.

    Parameters
    ----------
    result : Result
    path : str, path-like or binary file
        NumPy appends ``.npz`` to a name that lacks it.
    """
    replicates = result._replicates
    arrays = {f"replicates.{name}": getattr(replicates, name) for name in REPLICATE_ARRAYS}
    tuning = result.tuning
    if tuning is not None:
        arrays |= {f"tuning.{name}": getattr(tuning, name) for name in TUNING_ARRAYS}
    if result.pilot is not None:
        arrays["pilot"] = result.pilot
    seed = result.seed
    metadata = {
        "schema": SCHEMA,
        "settings": {
            name: getattr(result, name) for name in ("k", "m", "lag", "thin", "max_iter")
        },
        "names": result.names,
        "kernel": str(result.kernel),
        "initial": str(result.initial),
        "seed": None
        if seed is None
        else {
            "entropy": np.asarray(seed.entropy, dtype=object).tolist(),
            "spawn_key": [int(key) for key in seed.spawn_key],
            "pool_size": seed.pool_size,
        },
        "counts": {
            "n_log_prob": replicates.n_log_prob,
            "n_grad": replicates.n_grad,
            "seconds": replicates.seconds,
        },
        "tuning": None
        if tuning is None
        else {"n_steps": tuning.n_steps, "max_weight_ratio": tuning.max_weight_ratio},
    }
    # NumPy integers in the metadata become Python ones, which JSON writes.
    text = json.dumps(metadata, default=lambda value: value.item())
    np.savez(path, metadata=np.array(text), **arrays)


def load(path: str | os.PathLike[str] | IO[bytes]) -> Result:
    """Load a result saved by :func:`save`.

    The kernel and the initial distribution come back as their descriptions, strings.

    Parameters
    ----------
    path : str, path-like or binary file

    Returns
    -------
    Result
    """
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
        if seed is not None:
            seed = SeedSequence(
                seed["entropy"], spawn_key=tuple(seed["spawn_key"]), pool_size=seed["pool_size"]
            )
        return Result(
            replicates,
            **metadata["settings"],
            names=metadata["names"],
            kernel=metadata["kernel"],
            initial=metadata["initial"],
            seed=seed,
            tuning=tuning,
            pilot=data.get("pilot"),
        )
