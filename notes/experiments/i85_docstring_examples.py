"""Do the Examples of the docstrings print the same with other seeds, and how fast (#85)?

Each example fixes ``rng=0``. Its printed output is exact on one machine and NumPy
version, but another platform or version can draw other numbers (spec R-EXEC-2). As a
proxy, each example is rerun with ``rng`` set to each of 1,000 seeds, warnings as errors,
and its output is compared with the one written in the docstring; any difference or
warning would fail on a platform that draws those numbers. For each example the table
counts the seeds whose run warns, by the warning's class, and those that print another
output. The time is that of one run at the first seed of the range, after the imports;
at ``rng=0``, the seed of the docstrings, for the range that starts at 0. The seeds can
be run in pieces of a few minutes each, whose counts add up.

Result (``i85_docstring_examples.out``, run in 10 pieces of 100 seeds): at 1,000 seeds no
example warns, and each prints its documented output, except ``check_coupling``, whose
``passed()`` prints False at 3 seeds, within the false-failure rate of at most about 4 in
1,000 that its docstring states. The examples that call ``sample`` or ``run`` use fewer
than 100 replicates, below which the replicate-variance warning does not apply. Together
they take 2.3 s at ``rng=0``, the longest (HMC) 0.48 s.

Run: python notes/experiments/i85_docstring_examples.py [first last], for the seeds
from ``first`` to ``last - 1``; all 1,000 by default.
"""

import collections
import doctest
import importlib
import os
import pkgutil
import platform
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy

import telescopic

SEEDS = range(*map(int, sys.argv[1:3])) if len(sys.argv) > 1 else range(1000)

examples = [
    test
    for module in pkgutil.walk_packages(telescopic.__path__, "telescopic.")
    for test in doctest.DocTestFinder().find(importlib.import_module(module.name), globs={})
    if test.examples
]
commit = subprocess.run(
    ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
).stdout.strip()
print(f"commit {commit}; {platform.machine()}, {os.cpu_count()} cores")
print(
    f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}; "
    f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}; seeds {SEEDS.start} to {SEEDS.stop - 1}"
)
print(f"{'example':<46} {'seconds':>7} {'warns':>6} {'output':>6}  warnings")
total = 0.0
warnings.simplefilter("error")
for test in examples:
    warned, differs = collections.Counter(), 0
    for seed in SEEDS:
        seeded = doctest.DocTestParser().get_doctest(
            test.docstring.replace("rng=0", f"rng={seed}"), {}, test.name, None, 0
        )
        report = []
        start = time.perf_counter()
        failed = doctest.DocTestRunner().run(seeded, out=report.append).failed
        if seed == SEEDS.start:
            seconds = time.perf_counter() - start
        if failed:
            # An example that warns fails with the warning as its exception.
            raised = re.search(r"^\s*([\w.]+Warning): ", "".join(report), re.MULTILINE)
            if raised:
                warned[raised.group(1).rsplit(".", 1)[-1]] += 1
            else:
                differs += 1
    total += seconds
    names = ", ".join(f"{name} {count}" for name, count in warned.items())
    print(f"{test.name:<46} {seconds:7.2f} {warned.total():>6} {differs:>6}  {names}")
print(f"{'total':<46} {total:7.2f}")
