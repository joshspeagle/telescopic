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

With `vectorize=True`, the log-density takes a batch of points of shape `(n, d)` and
returns shape `(n,)`, so a step of many chains costs one call instead of one per point
([The log-density](log-density.md#the-log-density)). The engine advances a batch of
replicates together ([Chunks and workers](parallel.md#chunks-and-workers) gives its
size), so the saving grows with the share of a step's time that the calls of the log-density take. A
vectorized run is bitwise repeatable only under the same batching and BLAS threads,
with estimates that agree within their error bars otherwise
([Reproducibility](parallel.md#reproducibility)).

## Run on a pool

Replicates are independent, so `pool` runs them on worker processes
([Pools](parallel.md#pools)). Starting the workers takes time, so a pool pays off only for
runs longer than the start. A pool runs chunks of consecutive replicates, each worker's
chunk in lanes as above, and its workers run BLAS on one thread so that they do not
compete for the cores ([Chunks and workers](parallel.md#chunks-and-workers)). Each chunk
carries the target with its data, so keep the data that a log-density holds small
([Pickling and the main guard](parallel.md#pickling-and-the-main-guard)).

## Skip the pilots

Tuning runs in this process, before the replicates and whatever the pool, and the
meeting-time pilot runs before the first run. A kernel whose parameters are all given
skips tuning, and a given `k` skips the meeting-time pilot
([A slow log-density](settings.md#a-slow-log-density)). Several jobs on one target can
tune once and pass the tuned kernel and its `k` to each
([Tuning](parallel.md#tuning)).

## Choose the kernel by its inefficiency

Kernels differ in how fast their pairs meet and in what an application costs. The
{term}`inefficiency`, the cost of a replicate times the variance of its estimate, is the
figure to minimize at equal work, counted in evaluations or seconds when kernels differ in
the cost of an application ([Cost](settings.md#cost)).
[Choosing a kernel](kernels.md#choosing-a-kernel) says when each kernel suits, and
[Gradients](../tutorials/gradients.md) compares the random walk, MALA and HMC on one
target.

## Store less

`thin` and `keep` store fewer {term}`atoms <atom>` and save memory, but do not shorten a
run: a thinned run is as long ([Memory and thinning](settings.md#memory-and-thinning)), and
`keep` adds the accumulation of every replicate's moments
([Keeping a subset](settings.md#keeping-a-subset)). They help when memory, not time,
limits a run.

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
