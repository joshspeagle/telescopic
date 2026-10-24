---
file_format: mystnb
kernelspec:
  name: python3
---

# Writing a kernel

A kernel is a subclass of {class}`telescopic.kernels.Kernel` with two methods: `step`
moves a batch of chains, and `coupled_step` moves a batch of pairs. The engine does the
rest.

## The interface

- `step(target, state, draws)` returns the new states, and each chain's probability of
  accepting its proposal, of shape `(n,)`: 1 for a kernel that always moves. Dual
  averaging tunes step sizes on its mean.
- `coupled_step(target, state_x, state_y, draws)` returns the new states of the first
  and the second chain of each pair.

`target` is the {class}`telescopic.Target`: `target.log_prob(x)` evaluates the
log-density at the rows of `x`, and `target.log_prob_and_grad(x)` also the gradient; both
raise on the values that [Special values](targets.md#special-values) rules out. A
{class}`telescopic.kernels.State` holds `n` chains, batch first: positions `x` of shape
`(n, d)`, log-densities `log_prob` of shape `(n,)`, and gradients `grad` of shape
`(n, d)` or `None`. A kernel returns new states with the fields of its input states, all
valid at the new positions: when the input carries gradients, so does the output. It
never modifies its input states.

A {class}`telescopic.kernels.Draws` holds the random numbers. Row `i` of every request
belongs to chain, or pair, `i`:

- `draws.normal(size)`: standard normals, shape `(n, size)`;
- `draws.uniform()`: uniforms on $[0, 1)$, shape `(n,)`;
- `draws.log_uniform()`: their logarithms, `-inf` for a draw of 0, for Metropolis tests
  in log space;
- `draws.generator(row)`: the `numpy.random.Generator` of a row, for draws that the
  others do not cover, such as the `rvs` of a SciPy distribution or a rejection sampler.

Other members have defaults that a kernel can override:

- `needs_grad`, false by default: whether the kernel evaluates the gradient. When it is
  true, the states carry gradients and a target without `grad` is refused.
- `needs_log_prob`, true by default: false for a kernel that never evaluates the
  log-density, such as Gibbs. The target may then have `log_prob=None`.
- `init_state(target, x)` evaluates the target at starting points, with gradients when
  `needs_grad` is true.
- `ndim`, `None` by default: the dimension that the kernel's parameters fix, which
  `init_state` checks.
- `is_tuned`, true by default, and `tune(target, initial, rng=None, *, pilot=None)`. A
  kernel with parameters to tune makes `is_tuned` false while they are unset, and
  overrides `tune` to return a tuned copy and a report.
  {func}`telescopic.tuning.adaptive_metropolis` runs the pilot chains, unless a mixture
  passes them as `pilot`; their `mean`, `cov` and `report()` are what the built-in
  kernels use.
- `__repr__` names the kernel in summaries.

## The rules

1. **Marginals.** Each chain of `coupled_step` moves with exactly the law of `step`.
   Otherwise the estimates are biased.
2. **Exact meetings.** The engine declares a meeting when the two positions of a pair
   are equal, compared with `==`. Where the coupling makes the two new positions equal,
   the second chain must get a copy of the first chain's position, log-density and
   gradient, never a recomputation, which can differ in the last bits.
3. **Rows are independent.** A row's arithmetic must not depend on the other rows of the
   batch. Use elementwise operations, and `np.einsum` without `optimize` for linear
   algebra on rows; `@` and `solve` on a batch can give a row different bits in batches
   of different sizes. A replicate's results then do not depend on how the replicates
   are batched ([Parallel runs](parallel.md#reproducibility)).
4. **Determinism.** A step is a function of its inputs and its draws alone: every random
   number comes from `draws`, and nothing is kept between calls.

A coupled step need not keep equal states equal: the engine never applies it to a pair
that has met. To run on a pool of processes, the kernel must be picklable
([Parallel runs](parallel.md#pickling-and-the-main-guard)).

## What the engine does

{func}`telescopic.engine.run`, which the sampler calls:

- draws each pair's starting points, runs the first chain alone for `lag` steps, then
  applies `coupled_step` to the pairs that have not met, many replicates at a time;
- compares the positions of each pair after every coupled step, exactly, and records the
  meeting time;
- after the meeting applies `step` to the first chain alone, so the chains stay together
  whatever the coupling does, and a pair that has met costs one chain;
- gives each replicate its own random streams, so that its numbers depend only on the
  seed, its index and its own requests;
- stores the signed measures, counts the cost, and censors the pairs that reach
  `max_iter`.

## Coupling primitives

{mod}`telescopic.couplings` holds the couplings that the built-in kernels use. Each
works on a batch, and on rows where the two draws coincide it returns a copy.

- {func}`telescopic.couplings.reflection_maximal` couples $\mathcal N(\mu_x, CC^\top)$
  and $\mathcal N(\mu_y, CC^\top)$, given $C$ and $C^{-1}$, with $d$ normals and 1
  uniform per row. It is maximal: the draws coincide with probability
  $2\Phi(-\lVert C^{-1}(\mu_x - \mu_y)\rVert/2)$.
- {func}`telescopic.couplings.maximal` is the rejection maximal coupling of two
  distributions per row, objects with `rvs(random_state=...)` and a normalised `logpdf`
  or `logpmf`. Its number of draws is random, so it draws from each row's generator.
- {func}`telescopic.couplings.discrete_maximal` couples two distributions on
  $\{0, \dots, K-1\}$, given as log-probabilities of shape `(n, K)`, with 3 uniforms per
  row.

## An example: AR(1)

The kernel $x' = \rho x + \sqrt{1 - \rho^2}\,\xi$, with $\xi \sim \mathcal N(0, 1)$,
leaves $\mathcal N(0, 1)$ invariant and always moves. From $x$ it draws from
$\mathcal N(\rho x, s^2)$ with $s^2 = 1 - \rho^2$, so its coupled step draws the two
moves from the reflection-maximal coupling of $\mathcal N(\rho x, s^2)$ and
$\mathcal N(\rho y, s^2)$:

```{code-cell} python
import numpy as np

import telescopic
from telescopic import couplings
from telescopic.kernels import Kernel, State


class AR1(Kernel):
    def __init__(self, rho):
        self.rho, self.scale = rho, np.sqrt(1 - rho**2)
        # The Cholesky factor of the moves' variance, and its inverse.
        self.chol, self.inverse = np.array([[self.scale]]), np.array([[1 / self.scale]])

    def step(self, target, state, draws):
        x = self.rho * state.x + self.scale * draws.normal(1)
        return State(x, target.log_prob(x)), np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        x, y, identical = couplings.reflection_maximal(
            self.rho * state_x.x, self.rho * state_y.x, self.chol, self.inverse, draws
        )
        log_prob_x = target.log_prob(x)
        log_prob_y = log_prob_x.copy()  # copied where the moves coincide
        log_prob_y[~identical] = target.log_prob(y[~identical])
        return State(x, log_prob_x), State(y, log_prob_y)

    def __repr__(self):
        return f"AR1(rho={self.rho})"


def standard_normal(x):
    return -0.5 * np.sum(x**2, axis=1)


target = telescopic.Target(standard_normal, vectorize=True)
```

Its states carry no gradient. In a mixture with MALA or HMC they would, and the kernel
would then have to evaluate `target.log_prob_and_grad` at its new positions.

## Testing

{func}`telescopic.testing.check_coupling` starts `n` pairs, `(x, y)` or `(y, x)` in
random rows, and compares one coupled step with one step of the kernel alone from the
same points, among the chains that start at each point: how often each chain stays put
(Fisher's exact test), and where it moves (Kolmogorov-Smirnov tests along random
directions and of the distance moved). The swapped pairs catch a coupling that reads
another row's state, the more surely the more the log-densities at `x` and `y` differ.
It also reports the fraction of pairs that met; the states whose log-density or gradient
differs from a fresh evaluation; and the pairs that nearly met, within 64 units in the
last place, the mark of a recomputed proposal. `passed()` requires every p-value to be
at least $10^{-3}$, some pairs to meet, and no stale state or near miss. `x` and `y`
should be close enough to meet in one step sometimes.

```{code-cell} python
from telescopic.testing import check_coupling, fixed_draws

report = check_coupling(AR1(0.9), target, x=[0.5], y=[0.8], rng=1)
report, report.passed()
```

{func}`telescopic.testing.fixed_draws` replays given random numbers, to compare a step
with a computation by hand; a request beyond them raises `IndexError`.
{func}`telescopic.testing.random_draws` gives fresh ones for any number of rows. A step
from 2.0 with the normal 0.5, then coupled steps with the same numbers from 2.0 and 2.1,
and from 2.0 and 3.0:

```{code-cell} python
kernel = AR1(0.9)
starts = kernel.init_state(target, np.array([[2.0], [2.1], [3.0]]))
x, close, far = (starts.take([i]) for i in range(3))
moved, _ = kernel.step(target, x, fixed_draws(normals=[[0.5]]))
print("step:", moved.x[0, 0], "by hand:", 0.9 * 2.0 + np.sqrt(1 - 0.9**2) * 0.5)
pairs = {}
for name, y in [("close", close), ("far", far)]:
    draws = fixed_draws(normals=[[0.5]], uniforms=[[0.5]])
    pairs[name] = kernel.coupled_step(target, x, y, draws)
    print(name, pairs[name][0].x[0, 0], pairs[name][1].x[0, 0])
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
scale = np.sqrt(1 - 0.9**2)
assert moved.x[0, 0] == 0.9 * 2.0 + scale * 0.5
assert pairs["close"][0].x[0, 0] == pairs["close"][1].x[0, 0]
assert pairs["far"][1].x[0, 0] == 0.9 * 3.0 - scale * 0.5
```

The step equals the computation by hand. The close pair meets, and the distant one moves
by reflected normals: in one dimension, the second chain moves by $-z$ where the first
moves by $z$.

Last, a run on a target with known expectations, here $E[x] = 0$ and $E[x^2] = 1$,
checks the kernel as the sampler uses it:

```{code-cell} python
init = telescopic.initial.Normal([3.0], [0.25])
result = telescopic.sample(
    standard_normal, init, 1000, vectorize=True, kernel=AR1(0.9), rng=2
)
moments = result.estimate(lambda x: np.concatenate([x, x**2], axis=1))
moments
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.all(np.abs(moments.value - [0.0, 1.0]) < 3 * moments.standard_error)
```

Both estimates lie within three standard errors of the exact values.
