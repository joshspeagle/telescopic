---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  merge_streams: true
---

# Parallel runs

Replicates are independent, so they run in parallel without communicating. The `pool`
argument of {func}`telescopic.sample` and {class}`telescopic.CoupledSampler` decides where
the meeting-time pilot and the replicates run; the kernel is tuned in this process.

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

Starting worker processes takes time, so a pool pays off only for runs that take longer
than the start. telescopic starts and stops only the workers of an integer pool. The
workers of an executor or a pool of your own live as long as it does and serve every run
given it, and you close it. The executor below spawns its workers, as telescopic does:
forking a process that runs threads, such as a notebook's kernel or a multithreaded BLAS,
can deadlock. With `progress=True`, the sampler shows the progress of its pilot pairs and
of each run ([Progress](#progress)):

```{code-cell} python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import telescopic

gaussian = telescopic.examples.gaussian(2)  # evaluated one point at a time here
spawn = multiprocessing.get_context("spawn")
with ProcessPoolExecutor(2, mp_context=spawn) as executor:
    sampler = telescopic.CoupledSampler(
        gaussian.log_prob, gaussian.initial, pool=executor, progress=True, rng=3
    )
    first = sampler.run(100)
    both = sampler.run(100)
both.n_replicates
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert (first.n_replicates, both.n_replicates) == (100, 200)
assert (both.k, both.m, both.lag) == (first.k, first.m, first.lag)
```

The second run ran 100 more replicates on the same workers, with the settings of the
first, and returned the result of all the replicates so far. To split a run across
machines, run separate ranges of replicate indices with one seed and the same settings,
and join the results ([Merging runs](results.md#merging-runs)).

## Chunks and workers

A pool runs chunks of consecutive replicate indices: one per worker at least, more where
a chunk would hold more than 64 replicates (the lanes that the engine advances together),
and four per worker at most. An executor receives every chunk at once, in index order; a
pool with only `map` receives them in waves of one chunk per worker.

telescopic reads the number of workers of an integer pool, a `concurrent.futures`
executor, a `multiprocessing` pool and a schwimmbad pool. It counts any other pool as
having as many workers as this process may use CPUs, so the chunks of a larger pool, such
as a cluster's, can leave workers idle. To use them all, give such a pool an attribute
`size`, its number of workers, as schwimmbad's pools have.

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
layouts, so there the agreement is statistical. The first run above, on two workers, and
the same run in this process:

```{code-cell} python
here = telescopic.sample(gaussian.log_prob, gaussian.initial, rng=3)
(
    np.array_equal(here.meeting_times, first.meeting_times),
    np.array_equal(here.measure.atoms, first.measure.atoms),
    np.array_equal(here.measure.weights, first.measure.weights),
)
```

## Pickling and the main guard

The workers receive the job pickled. With cloudpickle installed (the `parallel` extra),
telescopic pickles it with cloudpickle, which also sends lambdas and functions defined in
a notebook or a script. Without cloudpickle, the standard pickle needs functions defined
at the top level of an importable module, as the log-density of
{func}`telescopic.examples.gaussian` above: a lambda raises `TypeError`, and a function
that a worker cannot import, such as one defined in a notebook, raises `RuntimeError`,
each with that advice.

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
sampler = telescopic.CoupledSampler(gaussian.log_prob, gaussian.initial, rng=1)
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
the rest of a pool of your own reacts to Ctrl-C is up to the pool. A process pool of your
own should ignore Ctrl-C in its workers, as telescopic's do:
`ProcessPoolExecutor(4, initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN))`.
Otherwise Ctrl-C kills its idle workers and breaks the pool: the run returns the chunks
that had finished, flagged `interrupted`, or raises `BrokenProcessPool` if none had, and
the pool cannot run again.

## Progress

`progress=True`, given to {func}`telescopic.sample`, to {class}`telescopic.CoupledSampler`
or to one call of `run`, shows the progress of the meeting-time pilot and of each run, in
replicates finished: a tqdm text bar on stderr with the `progress` extra, and otherwise a
line on stderr every 10 seconds and when the run ends. On a pool, the count moves as
chunks finish.

## Memory

The workers send the signed measures of their replicates back to this process, which
holds them all. [Settings](settings.md#memory-and-thinning) gives their size, the
{class}`telescopic.InputWarning <telescopic.warnings.InputWarning>` that the sampler emits
when they would take more than 1 GB, and `thin`, which stores fewer.
