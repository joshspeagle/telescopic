"""The setup of a measurement, as one line for the head of an experiment's output (issue #85).

CLAUDE.md asks every measurement to record its commit, machine, thread settings and
library versions. `setup()` returns them on one line: the commit of the package that was
imported (`git describe --always --dirty`, run in the package's folder), the processor
model and the number of logical cores, without a hostname, the thread variables of the
OpenMP and BLAS libraries that are set, and the versions of Python, NumPy and SciPy.

Use, in a script of this folder run as `python notes/experiments/<name>.py`:
  from i85_setup import setup
  print(setup())
"""

import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy

import telescopic

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def processor():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def setup():
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(telescopic.__file__).parent,
    ).stdout.strip()
    threads = [f"{name}={os.environ[name]}" for name in THREAD_VARIABLES if name in os.environ]
    return (
        f"setup: commit {commit} of the package; {processor()}, {os.cpu_count()} logical "
        f"cores; {', '.join(threads) or 'no thread variable set'}; Python "
        f"{platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}"
    )
