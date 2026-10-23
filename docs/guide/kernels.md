---
file_format: mystnb
kernelspec:
  name: python3
---

# Kernels and couplings

A kernel moves one chain and leaves the target invariant. Its coupling moves a pair of
chains so that each chain, on its own, moves by the kernel, and so that the two can
become equal ([Theory](../theory.md#coupled-chains)). Each kernel of
{mod}`telescopic.kernels` comes with its coupling. The argument `kernel` of
{func}`telescopic.sample` and {class}`telescopic.CoupledSampler` takes a kernel, or the
string that names it:

| Kernel | String | Needs | Coupling | `tune` sets |
|---|---|---|---|---|
| `RandomWalk` | `"auto"`, `"rwmh"` | log-density | reflection-maximal proposals, common uniform | `cov` |
| `MALA` | `"mala"` | log-density, gradient | reflection-maximal proposals, common uniform | `cov`, `step_size` |
| `HMC` | `"hmc"` | log-density, gradient | common momentum and uniform; reflection-maximal meeting step | `cov`, `step_size` |
| `Independence` | `"independence"` | log-density | common proposal and uniform | `proposal` |
| `Gibbs` | none | full conditionals | per block: maximal, common random numbers, or both | nothing |
| `Mixture` | none | what its components need | common component | its components' parameters |

`"auto"`, the default, is the random walk even when `grad` is given.

## Choosing a kernel

- **The random walk**, the default, needs only the log-density and has no step size to
  tune.
- **MALA and HMC** need the gradient. A step evaluates the log-density and the gradient
  once (MALA) or `n_steps` times (HMC). Their automatic step sizes can fail to couple
  ([When the automatic step fails](#when-the-automatic-step-fails)).
- **The independence kernel** suits a target that a Student-$t$ approximates well, such
  as a posterior close to Gaussian.
- **Gibbs** suits models whose full conditionals are known, such as conjugate
  hierarchical models.

Coupling does not make a kernel mix faster
([Theory](../theory.md#guarantees-and-their-limits)): a kernel that mixes slowly meets
slowly. To compare kernels on a target, compare their inefficiency, the variance of a
replicate's estimate times the cost of a replicate ([Settings](settings.md#cost)).
[Gradients](../tutorials/gradients.md) compares the random walk, MALA and HMC on a
Gaussian.

## Tuning

Parameters left as `None` are tuned from four adaptive Metropolis chains, the tuning
pilot ([Settings](settings.md)); parameters given are kept.
{meth}`telescopic.kernels.Kernel.tune` returns a tuned copy and the pilot's report. The
sampler calls it once, before anything else runs, and keeps the tuned kernel:

```{code-cell} python
import numpy as np
from scipy import optimize, special, stats

import telescopic
from telescopic import examples, kernels

gaussian = examples.gaussian(10)  # N(0, S), S_ij = 0.9^|i - j|, from N(0, 4I)
target = telescopic.Target(gaussian.log_prob, grad=gaussian.grad, vectorize=True)
tuned, report = kernels.HMC().tune(target, gaussian.initial, rng=0)
tuned
```

## Random walk

{class}`telescopic.kernels.RandomWalk` proposes $x' = x + \lambda Cz$, with $CC^\top$ =
`cov`, $\lambda$ = `scale` and $z \sim \mathcal N(0, I)$, and accepts when
$\log u < \log\pi(x') - \log\pi(x)$.

- `coupling="reflection"`, the default, draws the two proposals from the
  reflection-maximal coupling of their Gaussians and decides both acceptances with one
  uniform, as in Jacob, O'Leary & Atchadé (2020) and O'Leary (2021). A coupled step
  evaluates the log-density at most twice per pair, once when the proposals coincide.
- `coupling="maximal"` uses the rejection maximal coupling with independent residuals,
  whose meeting times grow exponentially with the dimension (Jacob, O'Leary & Atchadé
  2020, §4.2); it is there for comparison.
- `tune` sets `cov` to the covariance of the pilot chains. `scale` defaults to
  $2.38/\sqrt d$, the optimal scale of Roberts, Gelman & Gilks (1997) for a Gaussian
  target of covariance `cov`, and is not tuned.

## MALA

{class}`telescopic.kernels.MALA` needs `grad`. It proposes
$x' = x + \tfrac{\epsilon^2}{2}\Sigma\nabla\log\pi(x) + \epsilon Cz$, with
$\Sigma = CC^\top$ = `cov` and $\epsilon$ = `step_size`, and accepts by the
Metropolis-Hastings ratio, which includes the proposal densities.

- The coupled step draws the two proposals from the reflection-maximal coupling of their
  distributions, which share the covariance $\epsilon^2\Sigma$, and shares the uniform,
  as in Jacob, O'Leary & Atchadé (2020, §4.4). It is MALA's only coupling.
- `tune` sets `cov` to the covariance of the pilot chains, then `step_size` by dual
  averaging toward the acceptance rate of 0.574 that Roberts & Rosenthal (1998) found
  optimal, on four chains started from the pilot's last states.

## HMC

{class}`telescopic.kernels.HMC` needs `grad`. Each step applies one of two components,
chosen by a uniform:

- with probability $1 - \gamma$, $\gamma$ = `meeting_weight`, a leapfrog trajectory of
  `n_steps` steps of size $\epsilon$ = `step_size`, with `cov` as the inverse mass
  matrix; a trajectory that reaches zero density or a non-finite gradient stops there
  and is rejected;
- with probability $\gamma$, the meeting step: a random-walk step with proposal
  covariance $\sigma^2\Sigma$, $\sigma$ = `meeting_scale`.

Coupled chains take the same component. Their trajectories share the momentum and the
uniform, which can bring the chains together but never makes them equal. The meeting
step couples them by the reflection-maximal coupling, which can make them equal once
they are within about `meeting_scale` of each other in the coordinates that `cov`
whitens. This is the construction of Heng & Jacob (2019), whose meeting step is not
whitened and uses the rejection maximal coupling.

`tune` sets `cov` to the covariance of the pilot chains, then `step_size` by dual
averaging of the trajectories alone toward an acceptance rate of 0.8. `n_steps` defaults
to $\lceil 1/\epsilon\rceil$, a trajectory of length about 1 in whitened coordinates,
and at most 1,000.

## When the automatic step fails

Without `step_size`, MALA and HMC tune their step toward the acceptance rate that suits
the kernel alone, on chains near the bulk of the target, and Heng & Jacob (2019, §5)
show that settings that suit HMC alone can fail to bring coupled chains together. On
targets that are not log-concave, such as the banana of
{func}`telescopic.examples.banana` from its default initial distribution, the automatic
steps of MALA and HMC leave pilot pairs apart. Log-concave targets can fail too, where
the log-density curves more sharply than where the step was tuned. Where the largest
eigenvalue of $\Sigma H$, with $\Sigma$ = `cov` and $H$ the Hessian of $-\log\pi$,
exceeds $4/\epsilon^2$, MALA's proposals overshoot: on a Gaussian of that Hessian, the
proposal mean lands farther from the mode than the chain. HMC's leapfrog steps are
unstable beyond the same bound. A chain that starts in such a region can reject every
proposal. On the logistic regression below, whose predictors nearly separate the
outcomes, the automatic MALA leaves pilot pairs unmet. The symptom is pilot pairs still
unmet at the cap, so that `sample` raises `ValueError` once the pilot has run, or very
long meeting times. The remedy is an explicit `step_size`, checked as in
[Checking a step size](#checking-a-step-size). On the banana, Heng & Jacob (2019, §5.2)
report a mean meeting time of 158 with a step of 1/500 and 500 leapfrog steps,
`HMC(1/500, 500, cov=np.eye(2))`.

## Checking a step size

Pilot pairs check a step size before a run. With every parameter of the kernel given,
{meth}`telescopic.CoupledSampler.meeting_times` runs no tuning, only pairs with lag 1
until they meet or reach `max_iter`. Over a few step sizes, compare the mean cost of a
pair: the evaluations per step times the mean meeting time, infinite when a pair is
still unmet at the cap. Keep the step with the lowest cost.

The example is the logistic regression of
[Gradients](../tutorials/gradients.md#a-logistic-regression), with coefficients that
nearly separate the outcomes:

```{code-cell} python
rng = np.random.default_rng(1)
X = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
coefficients = [-0.5, 1.0, -1.0, 0.5, 0.0, 2.0]
y = (rng.random(200) < special.expit(X @ coefficients)).astype(float)


def log_prob(beta, X, y):
    eta = beta @ X.T
    return np.sum(y * eta - np.logaddexp(0, eta), axis=-1) - np.sum(beta**2, axis=-1) / 50


def grad(beta, X, y):
    return (y - special.expit(beta @ X.T)) @ X - beta / 25


fit = optimize.minimize(
    lambda beta: -log_prob(beta, X, y), np.zeros(6), jac=lambda beta: -grad(beta, X, y)
)
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
```

The scan starts from the automatic step, with the covariance that tuning chose. The
samplers have the same seed, so their pilot pairs start from the same points:

```{code-cell} python
automatic = telescopic.CoupledSampler(
    log_prob, init, args=(X, y), grad=grad, vectorize=True, kernel="mala", rng=2
)
automatic.tune()
cov = automatic.kernel.cov
per_step = 2  # a log-density and a gradient per chain and step
costs = {}
for step in [automatic.kernel.step_size, 1.0, 0.7, 0.5, 0.35]:
    sampler = telescopic.CoupledSampler(
        log_prob,
        init,
        args=(X, y),
        grad=grad,
        vectorize=True,
        kernel=kernels.MALA(step, cov=cov),
        rng=2,
    )
    tau = sampler.meeting_times(100, max_iter=3000)
    costs[step] = per_step * tau.mean()
    print(f"step {step:.3f}: {np.isinf(tau).sum():3} unmet, cost {costs[step]:.0f}")
best = min(costs, key=costs.get)
```

```{code-cell} python
:tags: [remove-cell]

# The claims of the next paragraph.
assert np.isinf(costs[automatic.kernel.step_size]) and np.isinf(costs[1.0])
assert np.isfinite(costs[best]) and best < 1.0
```

The automatic step leaves pairs unmet at the cap, and so does the step of 1.0. The step
with the lowest cost runs:

```{code-cell} python
result = telescopic.sample(
    log_prob,
    init,
    args=(X, y),
    grad=grad,
    vectorize=True,
    kernel=kernels.MALA(best, cov=cov),
    rng=3,
)
result.summary()
```

## Independence

{class}`telescopic.kernels.Independence` proposes $x' \sim q$ whatever the current
state, and accepts when
$\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]$.

- The coupled step gives both chains the same proposal and the same uniform, so a pair
  meets when both accept. It is a maximal coupling of the kernel (O'Leary, Wang & Jacob
  2021).
- `proposal` is a {class}`telescopic.kernels.MultivariateT`, a Student-$t$ with location
  `mean`, scale matrix `cov` and `df` degrees of freedom, a positive integer (`df=None`
  gives a Gaussian).
- `tune` sets a Student-$t$ with 5 degrees of freedom, located at the pilot's mean, with
  the pilot's covariance as scale matrix. The report's `max_weight_ratio` is the largest
  weight $\pi/q$ among the pilot's states, relative to their median; a large value
  announces heavy-tailed meeting times.

The meeting times have geometric tails only when the weights $\pi/q$ are bounded, so the
proposal needs tails at least as heavy as the target's.

## Gibbs

{class}`telescopic.kernels.Gibbs` updates blocks of coordinates in turn, each drawn from
its full conditional. `conditional(x, b)` receives a read-only point `x` of shape `(d,)`
and a block number `b`, and returns the distribution of block `b` given the other
coordinates of `x`: an object with `rvs(random_state=...)` and a normalised `logpdf`, or
`logpmf`, such as a frozen SciPy distribution. `blocks` lists index arrays that
partition the coordinates `0, ..., d - 1`. The coupled step couples each block's two
conditionals:

- `coupling="maximal"`, the default: the rejection maximal coupling. With `eta` below 1
  the meeting probability is capped at `eta`, and the number of draws it takes has the
  bounded variance of Atchadé & Jacob (2024, Algorithm 2).
- `coupling="crn"`: common random numbers, one uniform through `ppf` for a block of one
  coordinate, or common standard normals for a multivariate normal conditional (an
  object with `mean` and `cov`). They can bring the chains together, but make them equal
  only by rounding; `"two-scale"` adds the maximal coupling.
- `coupling="two-scale"`: common random numbers for pairs farther apart than `threshold`
  at the start of a sweep, and the maximal coupling for the others, as in Biswas,
  Bhattacharya, Jacob & Johndrow (2022).

Where the two chains agree on every coordinate outside a block, both get the same draw.
A Gibbs kernel needs no log-density, so `log_prob` can be `None`, and has nothing to
tune:

```{code-cell} python
rho = 0.8


def conditional(x, block):
    # One coordinate given the other: a standard bivariate normal, correlation rho.
    return stats.norm(rho * x[1 - block], np.sqrt(1 - rho**2))


gibbs = kernels.Gibbs(conditional, blocks=[[0], [1]])
init = telescopic.initial.Normal([0.0, 0.0], [4.0, 4.0])
sampler = telescopic.CoupledSampler(None, init, kernel=gibbs, rng=0)
np.quantile(sampler.meeting_times(), [0.5, 0.99])  # of 100 pilot pairs' meeting times
```

A sweep calls `conditional` once per chain and block, in Python, so building the
distributions can dominate its cost. A small class with only the methods used is faster
than a frozen SciPy distribution, as in {func}`telescopic.examples.pumps`. Worked
example: [Gibbs](../tutorials/gibbs.md).

## Mixtures

{class}`telescopic.kernels.Mixture` applies at each step one of its `kernels`, chosen by
a uniform with probabilities `weights`. Coupled chains take the same component and its
coupling. The choice never depends on the states, which keeps each chain's moves those
of the mixture (Atchadé & Jacob 2024, §4.2). `tune` tunes the untuned components on one
pilot. HMC is a mixture of its trajectories and its meeting step.
