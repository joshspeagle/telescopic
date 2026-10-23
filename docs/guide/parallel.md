---
file_format: mystnb
kernelspec:
  name: python3
---

# Parallel runs

Replicates are independent, so they run in parallel without communicating. The `pool`
argument of {func}`telescopic.sample` and {class}`telescopic.CoupledSampler` decides where
the meeting-time pilot and the replicates run; the kernel is tuned in this process. The
[parallel tutorial](../tutorials/parallel.md) works through an example.

## Pools

`pool` takes:

- `None`, the default: the replicates run in this process;
- an integer: that many worker processes (at most 61 on Windows), which telescopic
  starts with the spawn method for each call of `run` or `meeting_times`, and stops when
  the call ends. They start before the tuning, so that their start overlaps it, and
  serve the pilot and the run;
- an executor with `submit`, such as a {class}`concurrent.futures.ProcessPoolExecutor`;
- an object with `map`, such as a `multiprocessing.Pool` or the schwimmbad pools that
  emcee takes.

`log_prob` below is defined in this notebook and evaluated one point at a time. The same
run, in this process and on two workers:

```{code-cell} python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import telescopic


def log_prob(theta):
    return -0.5 * np.sum(theta**2)


init = telescopic.initial.Normal([0.0, 0.0], [4.0, 4.0])
here = telescopic.sample(log_prob, init, rng=3)
there = telescopic.sample(log_prob, init, rng=3, pool=2)
```

telescopic starts and stops only the workers of an integer pool. A pool of your own
serves as many runs as you like, without starting processes for each, and you close it.
This executor uses the spawn method, as telescopic's workers do: forking a process that
runs threads, such as a notebook's kernel or a multithreaded BLAS, can deadlock.

```{code-cell} python
spawn = multiprocessing.get_context("spawn")
with ProcessPoolExecutor(2, mp_context=spawn) as executor:
    runs = [telescopic.sample(log_prob, init, rng=seed, pool=executor) for seed in (3, 4)]
[run.n_replicates for run in runs]
```

## Chunks and workers

A pool runs chunks of consecutive replicate indices: one per worker at least, more where
a chunk would hold more than 64 replicates (the lanes that the engine advances together),
and four per worker at most. An executor receives every chunk at once, in index order; a
pool with only `map` receives them in waves of one chunk per worker. The workers are
counted as an integer pool's number, an executor's `_max_workers`, a `multiprocessing`
pool's `_processes` or a schwimmbad pool's `size`, and otherwise as the CPUs that this
process may use.

Each chunk carries the whole job: the target with its `args` and `kwargs`, the kernel, the
initial distribution, the root seed and the settings. The results are joined in index
order; `n_log_prob` and `n_grad` add up the workers' counts, and `wall_time` is this
process's.

A task in the main thread of another process runs with BLAS limited to one thread, so
that the workers do not compete for the cores, and ignores Ctrl-C, which this process
handles. A task in this process, such as one on a thread pool, leaves the process's
settings alone.

## Reproducibility

Each replicate draws from a random stream of its own, derived from the root seed and its
index, so its output does not depend on where it runs. With a log-density evaluated point
by point (`vectorize=False`, the default), the replicates are bitwise identical in any
pool, with any number of workers and chunks, on one machine with the same NumPy and SciPy
versions; generator algorithms, SciPy's samplers and BLAS builds can change between
versions and machines. A vectorised log-density can differ in the last bits between batch
layouts, so there the agreement is statistical. The two runs above:

```{code-cell} python
(
    np.array_equal(here.meeting_times, there.meeting_times),
    np.array_equal(here.measure.atoms, there.measure.atoms),
    np.array_equal(here.measure.weights, there.measure.weights),
)
```

## Pickling and the main guard

The workers receive the job pickled. With cloudpickle installed (the `parallel` extra),
telescopic pickles it with cloudpickle, which also sends lambdas and functions defined in
a notebook or a script, as `log_prob` above. Without cloudpickle, the standard pickle
needs functions defined at the top level of an importable module: a lambda raises
`TypeError`, and a function that a worker cannot import, such as one defined in a
notebook, raises `RuntimeError`, each with that advice.

Under the spawn and forkserver start methods, the worker processes import the main
script, and telescopic's own workers always use spawn. A script that runs on such a pool
must keep its work under the `if __name__ == "__main__":` guard; without it, each worker
runs the script again, and the run fails. A notebook needs no guard.

```python
import numpy as np

import telescopic


def log_prob(theta):
    return -0.5 * np.sum(theta**2)


if __name__ == "__main__":
    init = telescopic.initial.Normal([0.0, 0.0], [4.0, 4.0])
    result = telescopic.sample(log_prob, init, pool=4, rng=1)
    print(result.summary())
```

## Interrupting a run

The first Ctrl-C during the replicates of a run starts no new replicate and lets those in
progress finish. The run then returns a result flagged `interrupted`, with every replicate
that started, and telescopic prints that a second Ctrl-C stops at once. Its estimates are
consistent but not exactly unbiased, since the number of replicates that ran depends on
their costs, and its summary says so. The sampler's next `run` starts with the missing
replicates, so a run of as many as are missing completes the requested set, which
restores exact unbiasedness:

```python
sampler = telescopic.CoupledSampler(log_prob, init, rng=1)
result = sampler.run(1000)  # Ctrl-C once while it runs
result.interrupted  # True: some of the 1000 replicates never started
result = sampler.run(1000 - len(result.index))  # the missing replicates
result.interrupted  # False
```

A second Ctrl-C raises `KeyboardInterrupt` at once, without a result, and stops the
workers that telescopic started. Ctrl-C during the tuning or the meeting-time pilot raises
`KeyboardInterrupt`, since they have no result to return early. telescopic handles Ctrl-C
only in the main thread; a run in another thread leaves it alone.

On a pool, telescopic's workers start no new replicate after the first Ctrl-C, and a chunk
that starts afterwards runs nothing, unless it is the first, so that the result holds at
least one replicate. On an executor of your own, the chunks not yet started are cancelled
and the running ones finish; a pool with only `map` returns the waves it completed. How
the rest of a pool of your own reacts to Ctrl-C is up to the pool.

## Progress

`progress=True`, given to {func}`telescopic.sample`, to {class}`telescopic.CoupledSampler`
or to one call of `run`, shows the progress of the meeting-time pilot and of each run, in
replicates finished: a tqdm bar with the `progress` extra, and otherwise a line on stderr
every 10 seconds and when the run ends. On a pool, the count moves as chunks finish.

## Memory

The workers send the signed measures of their replicates back to this process, which
holds them all. [Settings](settings.md#memory-and-thinning) gives their size, the
{class}`telescopic.InputWarning <telescopic.warnings.InputWarning>` that the sampler emits
when they would take more than 1 GB, and `thin`, which stores fewer. As a run ends,
sorting the atoms by replicate briefly holds about three times their size (spec section
6.6, [Specification](../design/spec.md)).
