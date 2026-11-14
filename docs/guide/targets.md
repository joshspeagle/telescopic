---
file_format: mystnb
kernelspec:
  name: python3
---

# Targets and initial distributions

A run needs two inputs: the log-density of the target, and the initial distribution from
which each pair of chains starts.

## The log-density

`log_prob(x, *args, **kwargs)` takes one point `x` of shape `(d,)` and returns its
log-density, up to an additive constant, as a float. The arguments `args` and `kwargs`
of {func}`telescopic.sample` are passed on to it: `args` is a tuple or a list, such as
`(data,)`, and any other `args`, such as an array, raises `TypeError`. With
`vectorize=True` it takes a batch of shape `(n, d)` and returns shape `(n,)`, so that a
step of many chains costs one call instead of one per point. An output whose shape
differs from the expected one only by axes of length 1, such as the shape `()` that
SciPy's `multivariate_normal.logpdf` returns for a batch of one point, is reshaped; a
function called point by point may return a float at some points and an array of one
element at others. Integers, booleans and real numbers held as objects, such as
`Fraction`s or `Decimal`s, convert to float64. Any other shape raises `ValueError`, as
does an output that is not numbers, such as `None` or emcee's `(log_prob, blobs)` tuple:
quantities derived from the parameters are computed after the run, with
{meth}`telescopic.Result.estimate`. A complex output whose imaginary part is exactly 0,
as complex-step derivatives give, counts as its real part; any other complex output
raises. telescopic computes in float64: a function that returns float32 values, as JAX
does unless `jax.config.update("jax_enable_x64", True)` is called first, rounds the
target, so it emits an `InputWarning`, once per target and naming the dtype. On a pool,
each worker process evaluates its own copy of the target and warns there, so the warning
appears in the workers' output rather than in this process, unless this process also
evaluates the target, as tuning does.

```{code-cell} python
import numpy as np
from scipy import stats

import telescopic


def log_prob(x, mean, sd):
    return -0.5 * np.sum(((x - mean) / sd) ** 2)


def log_prob_batch(x, mean, sd):
    return -0.5 * np.sum(((x - mean) / sd) ** 2, axis=1)


mean, sd = np.array([1.0, -2.0]), np.array([0.5, 2.0])
```

`grad` is the gradient of the log-density, with the same signature, returning shape
`(d,)`, or `(n, d)` with `vectorize=True`. MALA and HMC need it
([Kernels and couplings](kernels.md)). Both functions receive read-only arrays, so one
that writes to its input raises instead of corrupting the chains, and what they return is
copied, so one may reuse its output array from call to call.

{class}`telescopic.Target` holds these functions. {func}`telescopic.sample` and
{class}`telescopic.CoupledSampler` build one from their arguments; it is needed directly
only with the building blocks, such as kernels, the engine and
{func}`telescopic.testing.check_coupling`. Its `log_prob` takes a batch of shape
`(n, d)`, whether or not the function is vectorized, and counts the evaluations:

```{code-cell} python
points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
one = telescopic.Target(log_prob, args=(mean, sd))
batch = telescopic.Target(
    log_prob_batch, kwargs={"mean": mean, "sd": sd}, vectorize=True
)
one.log_prob(points), batch.log_prob(points), one.n_log_prob
```

With `vectorize=True`, a replicate's results can depend in the last bits on how the
replicates are batched ([Parallel runs](parallel.md#reproducibility)).

## Special values

- `-inf` means zero density. Every Metropolis test rejects a proposal there, so the
  log-density should return it outside the support.
- NaN and `+inf` raise `ValueError` with the point.

SciPy returns NaN for a negative scale, so a model of 20 observations whose scale is not
restricted fails at the first chain that proposes one:

```{code-cell} python
y = np.random.default_rng(2).normal(1.0, 2.0, size=20)


def log_prob_unrestricted(theta, y):
    location, scale = theta
    return np.sum(stats.norm.logpdf(y, location, scale))


unrestricted = telescopic.Target(log_prob_unrestricted, args=(y,))
try:
    unrestricted.log_prob(np.array([[0.0, -1.0]]))
except ValueError as error:
    print(error)
```

Returning `-inf` outside the support fixes it:

```{code-cell} python
def log_prob_restricted(theta, y):
    location, scale = theta
    if scale <= 0:
        return -np.inf
    return np.sum(stats.norm.logpdf(y, location, scale))


restricted = telescopic.Target(log_prob_restricted, args=(y,))
restricted.log_prob(np.array([[0.0, 1.0], [0.0, -1.0]]))
```

For the gradient:

- Where the log-density is `-inf`, the gradient is not evaluated, so it may fail there.
  States there carry a gradient of 0, so MALA and HMC propose from them without drift.
- Where the log-density is finite, a gradient that is not finite raises `ValueError`
  with the point at the states that chains start from: a MALA chain could not leave such
  a start, since a move back is accepted only if the reverse proposal, which carries the
  full drift, could return. Tuning's step-size search starts its chains from the
  pilot's last states, not from the initial distribution, and takes it as 0 there, since
  only the step size depends on those chains. Elsewhere it is taken as 0 too, which
  keeps MALA and HMC exact but takes their drift away there. The pilot pairs and the
  replicates of a run that meet one warn, in this process as soon as they do and on a
  pool when the run ends, and `Result.n_grad_zeroed` counts the replicates' ones. Tuning
  does not warn: its search for a step size reaches far into the tails.
- Such a gradient costs nothing where the chains do not go, as with a mixture's
  responsibilities computed from densities, which underflow far in the tails. Where they
  go, it slows them, and HMC pairs that lose their drift over much of the target may not
  meet. A gradient computed in log space avoids both, such as responsibilities from
  `scipy.special.logsumexp`. In this process, `warnings.simplefilter("error",
  telescopic.InputWarning)` stops a run at the first such gradient after tuning.

The log-density must be deterministic: the same point must always give the same value.
Pairs meet when their states are equal, and a log-density estimated by simulation breaks
the couplings.

## Checking the gradient

MALA and HMC stay exact with a wrong gradient: their Metropolis tests correct any drift.
A wrong gradient makes them slow instead, and their pairs can take hundreds of times
longer to meet. So when {class}`telescopic.CoupledSampler` or {func}`telescopic.sample`
is built with `grad` and a kernel that evaluates it, MALA, HMC or a mixture with one of
them, it compares `grad` with differences of `log_prob` before tuning. At each of 8 draws
$x$ of the initial distribution, along the direction $u$ between two further draws, it
compares the change of $\log\pi$ that `grad` predicts, $\nabla\log\pi(x)\cdot u$, with
central differences of `log_prob` at steps $hu$, where $h = \epsilon^{1/3}$ for the
epsilon $\epsilon$ of the values `log_prob` returns. A line disagrees when the gap
between the two exceeds 10 times the estimated error of the comparison: the truncation
of the differences, from a second difference at steps $2hu$, and the rounding of
`log_prob` and of `grad`. It costs 40 evaluations of `log_prob` and 8 of `grad` at any
dimension, and 8 more of each when `grad` returns float32 values, and again when
`log_prob` does; it draws from a random stream of its own, and it changes
no result: a run is the same with and without it.

A gradient with its sign wrong warns before any pair runs, with the size of the
disagreement:

```{code-cell} python
:tags: [wrap-output]

import warnings


def log_prob_gaussian(x):
    return -0.5 * np.sum((x - 1.0) ** 2, axis=-1)


def grad_wrong_sign(x):
    return x - 1.0


init = telescopic.initial.Normal(np.zeros(5), 4 * np.eye(5))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sampler = telescopic.CoupledSampler(
        log_prob_gaussian, init, grad=grad_wrong_sign, vectorize=True, kernel="mala", rng=0
    )
print(caught[0].message)
```

The warning is an `InputWarning`. The sizes are in nats along $u$, which spans the
spread of the initial distribution. The sampler keeps what the check compared, as a
{class}`telescopic.diagnostics.GradientCheck`, in `sampler.gradient_check`: the gap,
the tolerance and the predicted change of each line, the draw it starts from, and the
ratio of gap to tolerance. The check warns when a gap exceeds its tolerance, and here
each of the 8 lines does:

```{code-cell} python
check = sampler.gradient_check
print(f"{len(check.gap)} lines, the smallest ratio {check.ratio.min():.3g}")
```

`check_grad=False` turns the check off, and `gradient_check` is then `None`. A check
that could compare no line, because every draw has zero density or a gradient that is not
finite, or every line reaches zero density, as from a start on a boundary of the support,
warns too, since it leaves `grad` unchecked. An error raised inside the check, by
`log_prob` or `grad` or about what they return, is raised as it is, with a note that it
was raised in the gradient check, which evaluates them at draws of the initial
distribution when the sampler is built.

A warning means that `grad` is not the derivative of `log_prob` as computed, along the
spread of the initial distribution, by more than the check can attribute to rounding:
about $2\times10^{-8}\,|\log\pi|$ nats with float64 values, and
$1.5\times10^{-2}\,|\log\pi|$ with float32 ones. It does not mean that the gradient
slows the chains. These figures come from a test whose criteria were fixed before it
ran, with MALA pairs measuring how much a wrong gradient slows meeting, and from an
independent attempt to refute it:

- **Exact gradients of smooth code stay quiet.** On 109 targets whose `grad` is the exact
  derivative of `log_prob`, 300 draws of the check each, it never warned: Gaussians
  with scales from $10^{-8}$ to $10^{8}$ and conditioning to $10^{12}$, heavy tails,
  kinks such as a Laplace density in 10,000 dimensions, tables, and float32 values. The
  largest ratio was 0.39.
- **Wrong gradients warn.** Of 49 wrong gradients that slowed the meeting of MALA pairs
  by a factor of 2 or more, it warned in every check on 46: wrong signs, factors of 2,
  terms left out, and one wrong coordinate among 10,000. The other 3 started much
  tighter than the target in the wrong coordinate (below).
- **Gradients that are approximate by design warn too**, although they slowed nothing
  measurably: the exact gradient of a log-density computed by an iterative solver at a
  relative tolerance of $10^{-3}$ or $10^{-5}$, a gradient 1.005 times too large, a
  logistic regression's gradient without its prior (300 of 300 each); a gradient
  computed in float32 and returned as float64 (195 to 200 of 200 on six of nine
  targets); forward differences as `grad` (194 to 200 of 200); a log-density from an
  adaptive ODE solver at a tolerance of $10^{-6}$ with the gradient of the exact
  solution (100 of 100). Such a gradient warns once per sampler; pass `check_grad=False`
  once its accuracy is known to be enough.
- **A correct gradient can warn on a strongly periodic or finely tabulated
  log-density** from a wide start: with $10^5$ periods across the initial distribution,
  48 of 200 checks warned, with $10^6$, all 200; a piecewise-linear table with a spacing
  of $10^{-5}$ of the initial spread warned 143 times in 200.
- **Silence is weak evidence.** The check compares at 8 draws where the chains start. It
  saw nothing in 200 checks each of a wrong-sign gradient from a start at the mode, of a
  gradient wrong only away from a small ball of starts, of a `log_prob` offset by
  $-10^{10}$, and, with float32 values of `log_prob` and one coordinate's sign wrong, it
  warned in 196 of 200 checks in 200 dimensions, 37 of 200 in 1,000 and none of 100 in
  10,000. A start much tighter than the target in the wrong coordinate also weakens it:
  100 times tighter in 10,000 dimensions, it warned in 518 of 1,000 checks.

## Constrained parameters

A parameter with a bound, such as a scale $\sigma > 0$, can be sampled through an
unconstrained transform, such as $s = \log\sigma$. The log-density of $s$ adds the
logarithm of the Jacobian $d\sigma/ds = \sigma$, which is $s$. The chains then never
reach a boundary, and every point of the initial distribution lies inside the support.
We sample the model above with flat priors on the location and the scale:

```{code-cell} python
def log_prob_log_scale(theta, y):
    location, log_scale = theta
    log_jacobian = log_scale
    return np.sum(stats.norm.logpdf(y, location, np.exp(log_scale))) + log_jacobian


init = telescopic.initial.Normal([0.0, 0.0], [1.0, 1.0])
result = telescopic.sample(
    log_prob_log_scale, init, args=(y,), names=["location", "log_scale"], rng=0
)
```

An expectation of $\sigma$ is an expectation of a function of $s$, so the same run gives
it:

```{code-cell} python
result.estimate(lambda t: np.exp(t[:, 1]))
```

[Results](results.md#quantiles-and-credible-intervals) gives the quantiles of the
sampled coordinates and of derived quantities. To sample $\sigma$ itself instead, return
`-inf` where $\sigma \le 0$, as above, and draw the initial distribution inside the
support ([Choosing one](#choosing-one)).

## Initial distributions

Each replicate draws the starting points $X_0$ and $Y_0$ of its pair independently from
the initial distribution $\pi_0$, with its own random stream. `init` can be:

- an object with a method `rvs(random_state=generator)` that returns one point:
  {class}`telescopic.initial.Normal` (a mean, and a covariance matrix or the variances
  of a diagonal one), {class}`telescopic.initial.Uniform` (the box `[low, high)`),
  {class}`telescopic.initial.Empirical` (the uniform distribution over a set of points),
  or a frozen SciPy distribution such as `scipy.stats.multivariate_normal(mean, cov)`;
- a callable `init(rng)` that returns one point, drawn with the `numpy.random.Generator`
  it receives, so that the run stays reproducible from its `rng`;
- an array of points of shape `(n, d)`, used as `Empirical`, such as the last state of
  an ensemble sampler like emcee. A single point of shape `(d,)` is refused;
  `Empirical([point])` starts every chain there.

The samplers pass `init` to {func}`telescopic.initial.as_initial`, which turns any of
these forms into an object with `rvs`. The dimension `d` comes from a draw. Draws are converted to shape `(d,)`: SciPy returns
a scalar for a normal in one dimension, and shape `(1, d)` for a Dirichlet. A draw of
another shape, or with a non-finite coordinate, raises `ValueError`. We can run each form on the Gaussian above:

```{code-cell} python
points = stats.multivariate_normal(mean, np.diag(sd**2)).rvs(size=50, random_state=1)


def init_t(rng):
    return mean + 3 * sd * rng.standard_t(5, size=2)


inits = {
    "Normal": telescopic.initial.Normal(mean, cov=(3 * sd) ** 2),
    "Uniform": telescopic.initial.Uniform(mean - 5 * sd, mean + 5 * sd),
    "SciPy": stats.multivariate_normal(mean, np.diag((3 * sd) ** 2)),
    "callable": init_t,
    "array": points,
}
for name, init in inits.items():
    result = telescopic.sample(
        log_prob_batch, init, args=(mean, sd), vectorize=True, rng=1
    )
    print(f"{name:>8}: {result.mean()}")
```

## Choosing one

The estimates are unbiased for every initial distribution from which the chain
converges ([Theory](../theory.md#guarantees-and-their-limits)), but the choice sets their
variance and the cost, and decides whether the chains find every mode.

- **Cover every mode.** If every pair starts near one mode and the kernel rarely crosses
  to another, every pair meets in that mode, and the estimates and their error bars
  describe that mode alone. Nothing in the run shows the missing mode at a moderate
  number of replicates:
  [the theory's list of what is not guaranteed](../theory.md#guarantees-and-their-limits)
  cites a published case, and
  [When unbiased is not enough](../tutorials/failure-modes.md#a-start-that-misses-a-mode)
  runs one. The tuning chains are a first check, in one direction only: when their means
  spread more widely than the chains themselves, telescopic warns that the target may
  have several modes
  ([Diagnostics and warnings](diagnostics.md#disagreeing-tuning-chains-tuning)), but
  chains that agree can all sit in one mode. From a start that covers every mode, pairs
  that start in different modes take longer to meet, which the meeting times show,
  unless the kernel was tuned to one mode and meets quickly inside it
  ([A tuned proposal that covers one mode](../tutorials/failure-modes.md#a-tuned-proposal-that-covers-one-mode)).
- **Wider than the target, but not by orders of magnitude.** The automatic settings take
  `k` at the 99% quantile of the pilot's meeting times and the lag equal to `k`, so
  about one pair in a hundred meets while its chain is still on its way to the target.
  That replicate averages states far from the target without correction, and only rarer
  replicates whose other chain is late balance it, in expectation. From a start orders
  of magnitude wider than the target, these replicates dominate the estimates of
  variances, which can come out negative
  ([When unbiased is not enough](../tutorials/failure-modes.md#a-start-far-wider-than-the-target)).
  A start a few to ten times wider than the target avoids this.
- **For a target with one mode,** a Gaussian centered on an optimizer's fit, with its
  inverse Hessian scaled by 100, is ten times wider than the fit's error bars
  ([Getting started](../getting-started.md#a-first-run)).
- **Inside the support.** A replicate that starts where the target density is zero emits
  an {class}`~telescopic.warnings.InputWarning`, as soon as it starts in this process and
  when the run ends on a pool, and its chains stay there until they accept a proposal of
  finite density. A start many proposal scales outside the support may never get one,
  and without `max_iter` the run then waits forever.
- **The last state of an ensemble** keeps the estimates unbiased when the chain
  converges from it, but covers only the modes that the ensemble found.
