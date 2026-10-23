---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  merge_streams: true
---

# Parallel runs

Replicates are independent, so telescopic can run them on worker processes. The
argument `pool` of {func}`telescopic.sample` and {class}`telescopic.CoupledSampler` says
where the pilot pairs and the replicates run; tuning runs in this process. The
[parallel guide](../guide/parallel.md) gives the rules of pools, reproducibility,
pickling and interrupts.

```{code-cell} python
:tags: [remove-cell]

import warnings

# Without ipywidgets, tqdm warns in a notebook, then draws its bars as text.
warnings.filterwarnings("ignore", message="IProgress not found")
```

## The same replicates on four workers

The line fit of [Getting started](../getting-started.md), whose log-density takes one
point at a time:

```{code-cell} python
import numpy as np
from scipy import optimize

import telescopic

rng = np.random.default_rng(0)
x = np.sort(10 * rng.random(50))
yerr = 0.1 + 0.5 * rng.random(50)
y = -0.9 * x + 4.3 + yerr * rng.normal(size=50)


def log_prob(theta, x, y, yerr):
    slope, intercept = theta
    return -0.5 * np.sum(((y - slope * x - intercept) / yerr) ** 2)


fit = optimize.minimize(lambda theta: -log_prob(theta, x, y, yerr), [0.0, 0.0])
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
```

`pool=4` starts four worker processes for the call. The run gives the same meeting times
and the same estimates as a run in this process:

```{code-cell} python
here = telescopic.sample(log_prob, init, args=(x, y, yerr), rng=42)
pooled = telescopic.sample(log_prob, init, args=(x, y, yerr), pool=4, rng=42)
(
    np.array_equal(here.meeting_times, pooled.meeting_times),
    np.array_equal(here.mean().value, pooled.mean().value),
)
```

A replicate's random numbers derive from the seed and its index alone, and with a
log-density evaluated point by point its arithmetic does not depend on the other
replicates in its batch, so no replicate depends on the pool. A vectorised log-density
can round differently in batches of other sizes, and the replicates then agree only in
distribution. Starting the workers takes time, so a pool is worth it only for runs that
take longer.

## An executor, with progress

`pool` also takes an executor, such as a {class}`concurrent.futures.ProcessPoolExecutor`,
or an object with `map`, such as a {class}`multiprocessing.pool.Pool` or a schwimmbad
pool. Its workers live as long as it does, so later runs reuse them, where `pool=4`
starts new ones for each call of `run` or `meeting_times`. The executor below spawns its
workers, as telescopic does: forking a process whose BLAS runs threads can deadlock.
`progress=True` shows the progress of the pilot pairs and of each run, with a tqdm bar
when tqdm is installed (the `progress` extra) and lines on stderr otherwise.

```{code-cell} python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as executor:
    sampler = telescopic.CoupledSampler(
        log_prob, init, args=(x, y, yerr), pool=executor, progress=True, rng=42
    )
    first = sampler.run(100)
    both = sampler.run(100)
np.array_equal(first.meeting_times, here.meeting_times), both.n_replicates
```

The first run has the replicates of the run in this process again. The second ran 100
more, with the settings of the first, and returned the result of all the replicates so
far.

## Scripts, notebooks and interrupts

- **Scripts.** Spawned workers import the script that started them. Code that starts a
  run must sit under `if __name__ == "__main__":`, or each worker would run it again on
  import, and the run would fail:

  ```python
  import telescopic


  def log_prob(theta):
      return -0.5 * theta @ theta


  if __name__ == "__main__":
      init = telescopic.initial.Normal([0.0, 0.0], [4.0, 4.0])
      result = telescopic.sample(log_prob, init, pool=4)
      print(result.summary())
  ```

- **Notebooks.** The workers receive the log-density, the kernel and the initial
  distribution by pickling. cloudpickle, installed with the `parallel` extra, also sends
  functions defined in a notebook, as on this page; without it, they must be defined at
  the top level of a module.
- **Ctrl-C.** During the replicates, the first Ctrl-C lets the replicates in progress
  finish and returns a result whose `interrupted` is true: its estimates are consistent
  but not exactly unbiased. The sampler's next run starts with the replicates that did
  not run, and with them restores unbiasedness. A second Ctrl-C stops at once, without a
  result, as does a Ctrl-C during tuning or the pilot.
- **Merging.** {meth}`telescopic.Result.merge` joins two results of the same seed and
  settings that have no replicate in common.

The [parallel guide](../guide/parallel.md) has the details.
