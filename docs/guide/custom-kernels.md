---
file_format: mystnb
kernelspec:
  name: python3
---

# Write a kernel

A kernel is a subclass of {class}`telescopic.kernels.Kernel` with two methods: `step`
moves a batch of chains, and `coupled_step` moves a batch of {term}`pairs <pair>`. The engine does the
rest.

## The interface

- `step(target, state, draws)` returns the new states, and each chain's probability of
  accepting its proposal, of shape `(n,)`: 1 for a kernel that always moves. Dual
  averaging tunes step sizes on its mean.
- `coupled_step(target, state_x, state_y, draws)` returns the new states of the first
  and the second chain of each pair.

`target` is the {class}`telescopic.Target`: `target.log_prob(x)` evaluates the
log-density at the rows of `x`, and `target.log_prob_and_grad(x)` also the gradient, by
the rules of [Special values](log-density.md#special-values). A
{class}`telescopic.kernels.State` holds `n` chains, batch first: positions `x` of shape
`(n, d)`, log-densities `log_prob` of shape `(n,)`, and gradients `grad` of shape
`(n, d)` or `None`. A kernel returns new states with the fields of its input states, all
valid at the new positions: when the input carries gradients, so does the output, and a
run or a mixture raises `ValueError` when it lacks them. It never modifies its input
states.

A {class}`telescopic.kernels.Draws` holds the random numbers. Row `i` of every request
belongs to chain, or pair, `i`:

- `draws.normal(size)`: standard normals, shape `(n, size)`;
- `draws.uniform()`: uniforms on $[0, 1)$, shape `(n,)`;
- `draws.log_uniform()`: their logarithms, `-inf` for a draw of 0, for Metropolis tests
  in log space;
- `draws.generator(row)`: the `numpy.random.Generator` of a row, for draws that the
  others do not cover, such as the `rvs` of a SciPy distribution or a rejection sampler;
- `draws.child(row)`: a new child of that generator, a stream of its own that does not
  move the row's, for two draws that read the same numbers, as the Gibbs kernel's
  {term}`common random numbers` do.

Other members have defaults that a kernel can override:

- `needs_grad`, false by default: whether the kernel evaluates the gradient. When it is
  true, the states carry gradients and a target without a gradient, given as `grad` or
  `log_prob_and_grad`, is refused.
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
- `markov_fields` and `init_extra(state)`, for a kernel with fields of its own
  ([Fields of one kernel](#fields-of-one-kernel)).
- `__repr__` names the kernel in summaries.

## The rules

1. **Marginals.** Each chain of `coupled_step` moves with exactly the law of `step`.
   Otherwise the estimates are biased.
2. **Exact meetings.** The engine declares a meeting when the two Markov states of a
   pair are equal, compared with `==`: the positions, and the fields that
   `markov_fields` names. Where the {term}`coupling` makes the two new positions equal,
   the second chain must get a copy of the first chain's position, log-density, gradient
   and Markov fields, never a recomputation, which can differ in the last bits.
3. **Rows are independent.** For a target evaluated point by point, a row's arithmetic
   must not depend on the other rows of the batch. Use elementwise operations, and
   `np.einsum` without `optimize` for linear algebra on rows; `@` and `solve` on a batch
   can give a row different bits in batches of different sizes. A {term}`replicate`'s results
   then do not depend on how the replicates are batched
   ([Parallel runs](parallel.md#reproducibility)). Where `target.vectorize` is true,
   the built-in kernels use `@` for its speed and give up that invariance, which a
   vectorized target whose rows are computed independently, as the built-in examples'
   are, would otherwise keep. A coupled step that uses `@`
   computes each chain's products in a call of its own, of the same shape: two equal
   rows of one call can get different bits, which would separate chains that met.
4. **Determinism.** A step is a function of its inputs and its draws alone: every random
   number comes from `draws`, and nothing is kept in the kernel between calls. State
   kept from one step to the next lives in the states' `extra`.

A coupled step need not keep equal states equal: the engine never applies it to a pair
that has met. To run on a pool of processes, the kernel must be picklable
([Parallel runs](parallel.md#pickling-and-the-main-guard)).

## What the engine does

{func}`telescopic.engine.run`, which the sampler calls:

- draws each pair's starting points, runs the first chain alone for `lag` steps, then
  applies `coupled_step` to the pairs that have not met, many replicates at a time;
- compares the Markov states of each pair after every coupled step, exactly, and
  records the {term}`meeting time`;
- after the meeting applies `step` to the first chain alone, so the chains stay together
  whatever the coupling does, and a pair that has met costs one chain;
- gives each replicate its own random streams, so that its numbers depend only on the
  seed, its index and its own requests;
- stores the {term}`signed measures <signed measure>`, counts the cost, and censors the
  pairs that reach `max_iter`.

## Fields of one kernel

A state's `extra` maps names to arrays of the kernel's own, with the chains along the
first axis, such as a momentum that persists between steps or a cached value. The kernel
gives their initial values in `init_extra(state)`, which `init_state` attaches, and names
in `markov_fields` those that are part of its Markov state: a pair meets only when these
are equal too, and the engine raises `ValueError` when a step returns states without
them. Every other field is a cache, which the kernel checks before use and recomputes
where it is missing or stale, since a state can reach it without its caches or after
another kernel moved the chain.

A kernel carries unchanged the fields it does not know: it acts as the identity on
them, and returns copies of them, as the AR(1) example below does, so that a returned
state shares no array with its input. `State.take` copies every field, and `State.put`
writes the fields that both states have and leaves the others. A kernel that builds its
new states without its input's fields drops them; a run keeps their old values, so the
caches of other kernels in a mixture stay correct only because each checks its own.

A {class}`~telescopic.kernels.Mixture` names the field `name` of component `i` as
`f"{i}/{name}"`, so two components never share a field, and gives each component its own
fields under their own names. It refuses a component with Markov fields: a mixture
leaves a joint law $\pi(x)\psi(v)$ invariant only when $\psi$ does not depend on $x$,
which it cannot check. A state carries its kernel's caches, so before handing a state to
another kernel by hand, drop its `extra`.

No kernel of the package has Markov fields. {func}`telescopic.testing.check_coupling`
counts meetings on the Markov state, and steps each returned state with and without its
caches to catch a cache that does not match its chain.

## Coupling primitives

{mod}`telescopic.couplings` holds the couplings of the built-in kernels, and
`discrete_maximal` for kernels on finite state spaces. Each works on a batch, and on
rows where the two draws coincide it returns a copy. Its coupling objects,
{class}`~telescopic.couplings.Reflection`, {class}`~telescopic.couplings.Maximal` and
{class}`~telescopic.couplings.Shared`, couple only the proposals of
{class}`telescopic.kernels.MetropolisHastings` that the package ships (Proposals and
coupling objects, in [Choose a kernel](kernels.md)); the AR(1) example below, whose
proposal mean $\rho x$ is not one of theirs, is a kernel of its own.

- {func}`telescopic.couplings.reflection_maximal` couples $\mathcal N(\mu_x, CC^\top)$
  and $\mathcal N(\mu_y, CC^\top)$, given $C$ and $C^{-1}$, with $d$ normals and 1
  uniform per row. It is maximal: the draws coincide with probability
  $2\Phi(-\lVert C^{-1}(\mu_x - \mu_y)\rVert/2)$. With `blas=True`, for a vectorized
  target, it computes its products with BLAS.
- {func}`telescopic.couplings.maximal` is the rejection {term}`maximal coupling` of two
  distributions per row, objects with `rvs(random_state=...)` and a normalized `logpdf`
  or `logpmf`. Its number of draws is random, so it draws from each row's generator.
- {func}`telescopic.couplings.discrete_maximal` couples two distributions on
  $\{0, \dots, K-1\}$, given as log-probabilities of shape `(n, K)`, with 3 uniforms per
  row.

## A template

Every member a kernel needs, to copy and fill in: `step`, `coupled_step` and `__repr__`,
then the members whose defaults the kernel may override, each returning its default:

```{code-cell} python
from telescopic.kernels import Kernel, State


class MyKernel(Kernel):
    def __init__(self, scale):
        self.scale = scale  # parameters only: nothing is kept between steps

    def step(self, target, state, draws):
        # Move n chains, row i of every request of draws for chain i. Return the new
        # states, with the fields of `state` valid at the new positions, and each chain's
        # probability of accepting its proposal, shape (n,).
        raise NotImplementedError

    def coupled_step(self, target, state_x, state_y, draws):
        # Move n pairs, each chain with exactly the law of step. Where the coupling makes
        # the two positions equal, give the second chain a copy of the first chain's
        # position, log-density, gradient and Markov fields.
        raise NotImplementedError

    def __repr__(self):
        return f"MyKernel(scale={self.scale})"

    # The members below have defaults; keep only those the kernel changes.

    @property
    def needs_grad(self):
        return False  # True when the kernel evaluates the gradient

    @property
    def needs_log_prob(self):
        return True  # False when it never evaluates the log-density

    @property
    def ndim(self):
        return None  # the dimension that the kernel's parameters fix

    @property
    def is_tuned(self):
        return True  # False while a parameter is unset

    def tune(self, target, initial, rng=None, *, pilot=None):
        # A tuned copy and its report; the kernel itself and None when nothing is unset.
        return self, None

    @property
    def markov_fields(self):
        return ()  # the fields of `extra` that are part of the Markov state

    def init_extra(self, state):
        return {}  # the initial values of the kernel's own fields, each of shape (n, ...)
```

```{code-cell} python
:tags: [remove-cell]

# The template overrides only members of Kernel, the two abstract methods among them, and
# returns their defaults.
members = {name for name in vars(MyKernel) if not name.startswith("_")}
assert members <= set(dir(Kernel)) and Kernel.__abstractmethods__ <= members


class Bare(Kernel):
    step = coupled_step = None


template, bare = MyKernel(1.0), Bare()
for name in ("needs_grad", "needs_log_prob", "ndim", "is_tuned", "markov_fields"):
    assert getattr(template, name) == getattr(bare, name), name
assert template.init_extra(None) == bare.init_extra(None)
assert template.tune(None, None) == (template, None) and bare.tune(None, None) == (bare, None)
```

## An example: AR(1)

The kernel $x' = \rho x + \sqrt{1 - \rho^2}\,\xi$, with $\xi \sim \mathcal N(0, 1)$,
leaves $\mathcal N(0, 1)$ invariant and always moves. From $x$ it draws from
$\mathcal N(\rho x, s^2)$ with $s^2 = 1 - \rho^2$, so its coupled step draws the two
moves from the {term}`reflection-maximal coupling` of $\mathcal N(\rho x, s^2)$ and
$\mathcal N(\rho y, s^2)$:

```{code-cell} python
import copy

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
        # Other kernels' fields, carried unchanged as copies.
        moved = State(x, target.log_prob(x), extra=copy.deepcopy(state.extra))
        return moved, np.ones(len(x))

    def coupled_step(self, target, state_x, state_y, draws):
        x, y, identical = couplings.reflection_maximal(
            self.rho * state_x.x, self.rho * state_y.x, self.chol, self.inverse, draws
        )
        log_prob_x = target.log_prob(x)
        log_prob_y = log_prob_x.copy()  # copied where the moves coincide
        log_prob_y[~identical] = target.log_prob(y[~identical])
        return (
            State(x, log_prob_x, extra=copy.deepcopy(state_x.extra)),
            State(y, log_prob_y, extra=copy.deepcopy(state_y.extra)),
        )

    def __repr__(self):
        return f"AR1(rho={self.rho})"


def standard_normal(x):
    return -0.5 * np.sum(x**2, axis=1)


target = telescopic.Target(standard_normal, vectorize=True)
```

Its states carry no gradient. In a mixture with MALA or HMC they would, and the kernel
would then have to evaluate `target.log_prob_and_grad` at its new positions.

## Testing

{func}`telescopic.testing.check_coupling` checks a kernel's coupling as the engine uses
it: one coupled step against steps of the kernel alone from the same points, the fraction
of pairs that meet, stale cached log-densities and caches, and pairs that nearly
meet. A kernel of
your own should pass it before use. Its report's `passed()` says whether the coupled and
the uncoupled steps agree, as the API reference details. It catches a
coupling that reads another row's state the more surely the more the log-densities at
`x` and `y` differ, and `x` and `y` should be close enough to meet in one step
sometimes.

```{code-cell} python
from telescopic.testing import check_coupling, fixed_draws

report = check_coupling(AR1(0.9), target, x=[0.5], y=[0.8], rng=1)
report, report.passed()
```

The check compares moves in distribution. Drawn by hand for the same kernel and points,
the quantiles of each chain's coupled moves against those of the kernel alone lie on the
diagonal:

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      Two quantile-quantile plots side by side, for chain X from 0.5 in magenta and
      chain Y from 0.8 in teal: the quantiles of the coupled moves against those of
      the kernel alone fall along the gray diagonal.
  figure:
    caption: >-
      The quantiles of 10,000 coupled steps of AR(1) from $x = 0.5$ and $y = 0.8$,
      chain $X$ in magenta and chain $Y$ in teal, against the quantiles of 10,000
      steps of the kernel alone from the same points. Each chain's points fall on
      the diagonal, so each chain of the coupling moves with the law of the
      kernel, the first rule above.
---
import matplotlib.pyplot as plt
from scipy import stats

from telescopic import plotting
from telescopic.testing import random_draws

n = 10_000
kernel = AR1(0.9)
points = kernel.init_state(target, np.array([[0.5], [0.8]]))
start_x, start_y = points.take(np.zeros(n, int)), points.take(np.ones(n, int))
coupled = kernel.coupled_step(target, start_x, start_y, random_draws(n, rng=3))
levels = np.linspace(0.005, 0.995, 199)
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
qq = {}
for ax, chain, start, moved, color, seed in zip(
    axes, "XY", (start_x, start_y), coupled, (plotting.CHAIN_X, plotting.CHAIN_Y), (4, 5)
):
    alone = kernel.step(target, start, random_draws(n, rng=seed))[0]
    qq[chain] = moved.x[:, 0], alone.x[:, 0]
    ax.axline((0, 0), slope=1, color="gray", linewidth=0.8)
    ax.plot(
        np.quantile(alone.x[:, 0], levels),
        np.quantile(moved.x[:, 0], levels),
        ".",
        color=color,
        markersize=3,
    )
    ax.set(title=f"chain ${chain}$ from {start.x[0, 0]:g}", xlabel="kernel alone")
axes[0].set(ylabel="coupled step");
```

```{code-cell} python
:tags: [remove-cell]

# The claim of the caption: the coupled and uncoupled moves agree in distribution.
assert all(stats.ks_2samp(*qq[chain]).pvalue > 1e-3 for chain in "XY")
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
by_hand = 0.9 * 3.0 - scale * 0.5
assert abs(pairs["far"][1].x[0, 0] - by_hand) <= 2 * np.spacing(by_hand)
```

The step equals the computation by hand. The close pair meets, and the distant one moves
by reflected normals: in one dimension, the second chain moves by $-z$ where the first
moves by $z$, to rounding.

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
