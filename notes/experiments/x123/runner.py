"""What every runner of experiment x123 prints first: the setup of the run.

The commit of the package it imported (``git describe --always --dirty`` in the
package's folder), the processor model and core count, the thread variables as set, and
the versions of Python, NumPy and SciPy.
"""

import os
import platform
import subprocess
import sys

import numpy as np
import scipy

import telescopic

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def processor():
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def header():
    folder = os.path.dirname(os.path.abspath(telescopic.__file__))
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=folder,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    threads = ", ".join(f"{name}={os.environ.get(name)}" for name in THREADS)
    print(
        f"commit {commit}; processor {processor()}, {os.cpu_count()} cores; {threads}; "
        f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}"
    )
    # file arguments by their names only: no machine paths in an output
    script = f"notes/experiments/x123/{os.path.basename(sys.argv[0])}"
    args = [script] + [os.path.basename(a) if os.sep in a else a for a in sys.argv[1:]]
    print("command: " + " ".join(["python", *args]), flush=True)


def smoke():
    # The smoke runs of phase 1 use the seen seeds of families.SMOKE.
    return "--smoke" in sys.argv
