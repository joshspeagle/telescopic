---
file_format: mystnb
kernelspec:
  name: python3
---

# Gradients from JAX

A model written in JAX gets its gradient from `jax.grad`, and telescopic takes both the
log-density and the gradient as ordinary Python functions. This page runs the
Metropolis-adjusted Langevin algorithm ({class}`telescopic.kernels.MALA`) on a logistic
regression written in JAX, first point by point and then on batches of points, and
checks its estimates against a random-walk run that uses no gradient.

```{code-cell} python
:tags: [remove-cell]

from myst_nb import glue
```

## Float64 first

telescopic computes in float64, and JAX computes in float32 unless 64-bit mode is
turned on before any array is created:

```{code-cell} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
```

## The model

Two hundred binary outcomes $y_i$ depend on five predictors through
$P(y_i = 1) = 1/(1 + e^{-x_i^\top\beta})$, where $x_i$ starts with a 1 for the
intercept, and each of the six coefficients has the prior $\mathcal N(0, 5^2)$. The data
are simulated with NumPy, and the log-density of one point is written with `jax.numpy`:

```{code-cell} python
import numpy as np
from scipy import optimize, special

import telescopic

rng = np.random.default_rng(1)
X = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
y = (rng.random(200) < special.expit(X @ [0.5, 1.0, -0.5, 0.25, 0.0, -1.0])).astype(float)
names = ["intercept", "b1", "b2", "b3", "b4", "b5"]


def log_density(beta, X, y):
    eta = X @ beta
    return jnp.sum(y * eta - jnp.logaddexp(0.0, eta)) - jnp.sum(beta**2) / 50
```

## Point by point

`jax.jit` compiles the log-density, and `jax.grad` gives its gradient with respect to
the first argument. Both take one point of shape `(6,)`, as `log_prob` and `grad` do by
default ([The log-density](../guide/log-density.md#the-log-density)), and the data pass
through `args`:

```{code-cell} python
log_prob = jax.jit(log_density)
grad = jax.jit(jax.grad(log_density))
```

They return JAX arrays, which telescopic copies into NumPy float64 arrays as it checks
their shape and type, so no conversion is needed. Without 64-bit mode, JAX returns
float32 values, which round the target, and the sampler warns
([The log-density](../guide/log-density.md#the-log-density)).

The {term}`pairs <pair>` start from a Gaussian ten times wider than an optimizer's error bars
([Choose an initial distribution](../guide/initial-distribution.md#choosing-one)). The
optimizer takes the value and the gradient of the negative log-density from
`jax.value_and_grad`, with `jac=True`:

```{code-cell} python
negative = jax.jit(jax.value_and_grad(lambda beta, X, y: -log_density(beta, X, y)))
fit = optimize.minimize(negative, np.zeros(6), args=(X, y), jac=True)
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
```

A run of 300 {term}`replicates <replicate>` with MALA:

```{code-cell} python
point = telescopic.sample(
    log_prob, init, 300, args=(X, y), grad=grad, kernel="mala", names=names, rng=2
)
point.summary()
```

## On batches of points

`jax.vmap` maps a function of one point over the first axis of a batch, so
`jax.vmap(log_density, in_axes=(0, None, None))` takes points of shape `(n, 6)` and the
same data for every point. With `vectorize=True`, the kernels then call each function
once per step for all the chains they advance
([The log-density](../guide/log-density.md#the-log-density)). `jax.jit` compiles a
function anew for each shape of its input, and the number of chains in a batch changes
during a run, as pairs meet and new pairs start. A short run of 20 replicates shows how
many sizes the batches take:

```{code-cell} python
log_prob_batch = jax.jit(jax.vmap(log_density, in_axes=(0, None, None)))
grad_batch = jax.jit(jax.vmap(jax.grad(log_density), in_axes=(0, None, None)))

sizes = set()


def counted(beta, X, y):
    sizes.add(len(beta))
    return log_prob_batch(beta, X, y)


telescopic.sample(
    counted, init, 20, args=(X, y), grad=grad_batch, vectorize=True, kernel="mala", rng=2
)
len(sizes), max(sizes)
```

```{code-cell} python
:tags: [remove-cell]

glue("n_sizes", len(sizes), display=False)
```

Each of these {glue:text}`n_sizes:d` sizes costs a compilation, and a longer run meets
more of them. Padding each batch to
the next power of 2, with copies of its first point, and keeping the rows of the actual
points, leaves at most one compilation per power of 2:

```{code-cell} python
def padded(function):
    def call(beta, *args):
        n = len(beta)
        size = 1 << (n - 1).bit_length()
        filled = np.concatenate([beta, np.repeat(beta[:1], size - n, axis=0)])
        return function(filled, *args)[:n]

    return call


batch = telescopic.sample(
    padded(log_prob_batch),
    init,
    300,
    args=(X, y),
    grad=padded(grad_batch),
    vectorize=True,
    kernel="mala",
    names=names,
    rng=2,
)
print(f"point by point: {point.wall_time:.1f} s")
print(f"in padded batches: {batch.wall_time:.1f} s")
print(f"largest difference between the estimates: {np.max(np.abs(batch.mean().value - point.mean().value)):.1e}")
```

```{code-cell} python
:tags: [remove-cell]

# The claim of the next paragraph: the two runs batch differently, so they agree within
# their error bars, not bit for bit.
gap = np.abs(batch.mean().value - point.mean().value)
assert np.all(gap < 3 * np.hypot(batch.mean().standard_error, point.mean().standard_error))
```

The two runs have the same seed but evaluate the log-density in different batches, so
their estimates agree within their error bars, and bit for bit only when rounding never
changes an accept decision ([Reproducibility](../guide/parallel.md#reproducibility)).
How much time the batches save depends on the model, the number of chains and the
machine; the times above are those of the machine that built this page
([Make a run faster](../guide/faster.md)). A function that returns the value
and the gradient together, such as `jax.value_and_grad`, can be passed as
`log_prob_and_grad` instead of `grad`
([The value and the gradient in one call](../guide/log-density.md#the-value-and-the-gradient-in-one-call)).

## Against a run without gradients

A random-walk run uses only the log-density, so its estimates do not depend on the
gradient at all. Its 400 replicates give a reference with error bars of its own:

```{code-cell} python
reference = telescopic.sample(
    padded(log_prob_batch), init, 400, args=(X, y), vectorize=True, names=names, rng=3
)
```

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      For each of the six coefficients, two estimates of the posterior mean with
      error bars of two standard errors, from MALA with JAX gradients and from the
      random walk, side by side and overlapping.
  figure:
    caption: >-
      The posterior means of the six coefficients, with error bars of two standard
      errors, from MALA with JAX's gradients and from the random walk, which uses no
      gradient.
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.5))
position = np.arange(6)
for offset, label, run in ((-0.1, "MALA, JAX gradients", batch), (0.1, "random walk", reference)):
    estimate = run.mean()
    ax.errorbar(
        position + offset,
        estimate.value,
        2 * estimate.standard_error,
        fmt="o",
        markersize=4,
        label=label,
    )
ax.set_xticks(position, names)
ax.set(ylabel="posterior mean")
ax.legend();
```

```{code-cell} python
a, b = batch.mean(), reference.mean()
z = (a.value - b.value) / np.hypot(a.standard_error, b.standard_error)
dict(zip(names, z.round(2).tolist()))
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.all(np.abs(z) < 3)
glue("largest_z", float(np.max(np.abs(z))), display=False)
```

The two runs have different seeds, so their errors are independent, and each difference
of their estimates, in standard errors, should look like a draw from
$\mathcal N(0, 1)$. All six, printed above, are within 3 standard errors; the largest
in absolute value is {glue:text}`largest_z:.2f`.

## What the warnings say

```{code-cell} python
from telescopic import diagnostics

for label, run in (("point by point", point), ("batched", batch), ("random walk", reference)):
    print(f"{label}: {diagnostics.issues(run)}")
```

```{code-cell} python
:tags: [remove-cell]

assert all(diagnostics.issues(run) == [] for run in (point, batch, reference))
```

None of the three runs warns. An empty list means only that none of the conditions of
[Warnings and errors](../reference/warnings.md#warnings) holds, not that the run is right.
MALA stays exact with any gradient, so the agreement with the random walk checks the
runs, not the gradient: a wrong gradient would make the pairs slow to meet instead. The
sampler compares `grad` with differences of `log_prob` as it is built
([Checking the gradient](../guide/log-density.md#checking-the-gradient)), and that check
emitted no warning here.
