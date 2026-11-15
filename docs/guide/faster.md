---
file_format: mystnb
kernelspec:
  name: python3
---

# Make a run faster

A run's time is the time of its {term}`replicates <replicate>`, each a number of kernel
applications, plus that of the {term}`tuning pilot` and the {term}`meeting-time pilot`
before them. Each section below is one lever, with its effect and the page that holds
its facts; [Measured times](#measured-times) times the first three on one target.

## Where the time goes

A replicate applies the kernel to its first chain alone for the first $L$ steps, to both
chains until they meet, and to the met chain until $m$
([Theory](../theory.md#cost)):

```{diagram} cost
The kernel applications of one pair run by telescopic, with random-walk Metropolis on a
standard normal target, as cells along time: $X$ steps alone for the first $L$ steps,
the {term}`lag`; both chains step until the meeting time $\tau$, each step counting as two
applications; and the met chain steps alone from $\tau$ to $m$. The sum is the
replicate's entry of `result.costs`.
```

How many evaluations of the log-density and its gradient an application costs depends on
the kernel, and the pilots add their own
([A slow log-density](settings.md#a-slow-log-density)). A result records the
evaluations and the seconds of its replicates, `result.n_log_prob`, `result.n_grad` and
`result.wall_time`, without those of the pilots
([What a result holds](../reference/result-fields.md#what-a-result-holds)).

## Vectorize the log-density

With `vectorize=True`, one call of the log-density evaluates a batch of chains instead of
one point, which saves the time of the calls
([The log-density](log-density.md#the-log-density)); its effect on reproducibility is in
[Reproducibility](parallel.md#reproducibility).

## Run on a pool

`pool` runs the replicates on worker processes, which pays off for runs longer than the
start of the workers ([Pools](parallel.md#pools),
[Chunks and workers](parallel.md#chunks-and-workers)).

## Skip the pilots

A kernel whose parameters are all given skips tuning, and a given `k` skips the
meeting-time pilot ([A slow log-density](settings.md#a-slow-log-density)), so several
jobs on one target can tune once ([Tuning](parallel.md#tuning)).

## Choose the kernel by its inefficiency

Kernels differ in how fast their pairs meet and in what an application costs. The
{term}`inefficiency`, the cost of a replicate times the variance of its estimate, is the
figure to minimize at equal work, counted in evaluations or seconds when kernels differ in
the cost of an application ([Cost](settings.md#cost)).
[Choosing a kernel](kernels.md#choosing-a-kernel) says when each kernel suits, and
[Gradients](../tutorials/gradients.md) compares the random walk, MALA and HMC on one
target.

## Store less

`thin` and `keep` save memory, not time
([Memory and thinning](settings.md#memory-and-thinning),
[Keeping a subset](settings.md#keeping-a-subset)).

## Measured times

[`benchmarks/run_faster.py`](https://github.com/joshspeagle/telescopic/blob/main/benchmarks/run_faster.py)
times {func}`telescopic.sample` on {func}`telescopic.examples.gaussian` in 10 dimensions,
with a random walk whose covariance is the target's and a fixed `k`, so that no pilot
runs: with the log-density evaluated point by point and vectorized, in this process and
on pools of worker processes that each call starts. It then times the automatic
pipeline at the default settings against the same run with the kernel and `k` given. The
table and its setup are read from the script's output:

```{code-cell} python
:tags: [remove-input]

import json
from pathlib import Path

# The script's output, in the repository's benchmarks/results/.
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "benchmarks").is_dir())
output = root / "benchmarks" / "results" / "run-faster.json"
if not output.exists():
    print("Not yet measured: run python benchmarks/run_faster.py.")
else:
    report = json.loads(output.read_text())
    print(f"Seconds; {report['case']}.\n")
    print(f"{'':20}{'point by point':>16}{'vectorized':>12}")
    for row, seconds in report["seconds by pool and log-density"].items():
        print(f"{row:20}{seconds['point by point']:>16.2f}{seconds['vectorized']:>12.2f}")
    pilots = report["seconds with and without the pilots"]
    print(
        f"\nAutomatic pipeline: {pilots['automatic']:.2f} s, of which the replicates "
        f"{pilots['of which the replicates']:.2f} s (k = {pilots['automatic k']}); "
        f"kernel and k given: {pilots['kernel and k given']:.2f} s."
    )
    blas = ", ".join(f"{b['internal_api']} {b['version']}" for b in report["blas"])
    print(
        f"\nSetup: telescopic at commit {report['commit']}, Python {report['python']}, "
        f"NumPy {report['numpy']} and SciPy {report['scipy']}, {blas} on one thread, "
        f"on {report['machine']}, with {report['available CPUs']} CPUs available."
    )
```
