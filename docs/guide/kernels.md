---
file_format: mystnb
kernelspec:
  name: python3
---

# Kernels and couplings

A kernel moves one chain and leaves the target invariant. Its coupling moves a pair of
chains so that each chain, on its own, moves by the kernel, and so that the two can
become equal ([Theory](../theory.md)). Each kernel of {mod}`telescopic.kernels` comes
with its coupling.

The argument `kernel` of {func}`telescopic.sample` and
{class}`telescopic.CoupledSampler` takes a kernel or one of these strings:

| String | Kernel |
|---|---|
| `"auto"`, `"rwmh"` | `RandomWalk()` |
| `"mala"` | `MALA()` |
| `"hmc"` | `HMC()` |
| `"independence"` | `Independence()` |

`"auto"`, the default, is the random walk, with or without a gradient: gradient kernels
are chosen by name.

## Tuning

Parameters left as `None` are tuned from four adaptive Metropolis chains, the tuning
pilot ([Settings](settings.md)); parameters given are kept.
{meth}`telescopic.kernels.Kernel.tune` returns a tuned copy and the pilot's report. The
sampler calls it once, before anything else runs, and keeps the tuned kernel:

```{code-cell} python
import numpy as np
from scipy import stats

import telescopic
from telescopic import examples, kernels

gaussian = examples.gaussian(10)  # N(0, S), S_ij = 0.9^|i - j|, from N(0, 4I)
target = telescopic.Target(gaussian.log_prob, grad=gaussian.grad, vectorize=True)
tuned, report = kernels.HMC().tune(target, gaussian.initial, rng=0)
tuned
```

| Kernel | Needs | Coupling | `tune` sets |
|---|---|---|---|
| `RandomWalk` | log-density | reflection-maximal proposals, common uniform | `cov` |
| `MALA` | log-density, gradient | reflection-maximal proposals, common uniform | `cov`, `step_size` |
| `HMC` | log-density, gradient | common momentum and uniform; reflection-maximal meeting step | `cov`, `step_size` |
| `Independence` | log-density | common proposal and uniform | `proposal` |
| `Gibbs` | full conditionals | per block: maximal, common random numbers, or both | nothing |
| `Mixture` | what its components need | common component | its components' parameters |

## Random walk

{class}`telescopic.kernels.RandomWalk` proposes $x' = x + \lambda Cz$, with $CC^\top$ =
`cov`, $\lambda$ = `scale` and $z \sim \mathcal N(0, I)$, and accepts when
$\log u < \log\pi(x') - \log\pi(x)$.

- `coupling="reflection"`, the default, draws the two proposals from the
  reflection-maximal coupling of their Gaussians and decides both acceptances with one
  uniform (Jacob, O'Leary & Atchadé 2020; O'Leary 2021). A coupled step evaluates the
  log-density at most twice per pair, once when the proposals coincide.
- `coupling="maximal"` uses the rejection maximal coupling with independent residuals.
  Its meeting times grow exponentially with the dimension (Jacob, O'Leary & Atchadé
  2020, §4.2); it is there for comparison.
- `tune` sets `cov` to the covariance of the pilot chains. `scale` defaults to
  $2.38/\sqrt d$, the optimal scale for a Gaussian target of covariance `cov` (Roberts,
  Gelman & Gilks 1997), and is not tuned.

## MALA

{class}`telescopic.kernels.MALA` needs `grad`. It proposes
$x' = x + \tfrac{\epsilon^2}{2}\Sigma\nabla\log\pi(x) + \epsilon Cz$, with
$\Sigma = CC^\top$ = `cov` and $\epsilon$ = `step_size`, and accepts by the
Metropolis-Hastings ratio, which includes the proposal densities.

- The coupled step draws the two proposals from the reflection-maximal coupling of their
  distributions, which share the covariance $\epsilon^2\Sigma$, and shares the uniform
  (Jacob, O'Leary & Atchadé 2020, §4.4). It is MALA's only coupling.
- `tune` sets `cov` to the covariance of the pilot chains, then `step_size` by dual
  averaging toward an acceptance rate of 0.574 (Roberts & Rosenthal 1998), on four
  chains started from the pilot's last states.

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

Settings that suit HMC alone can fail to contract when coupled (Heng & Jacob 2019, §5),
so check the meeting times before a long run. The automatic step targets the acceptance
rate of HMC alone. On targets that are not log-concave it can leave the coupled chains
apart: on the banana of {func}`telescopic.examples.banana`, from its default initial
distribution, the automatic steps of HMC and of MALA leave so many pilot pairs unmet
that `k` cannot be chosen, and the run raises. On such targets, give an explicit,
smaller `step_size`. On the banana, Heng & Jacob (2019, §5.2) report a mean meeting time
of 158 with a step of 1/500 and 500 leapfrog steps, `HMC(1/500, 500, cov=np.eye(2))`.
Worked examples: [Gradients](../tutorials/gradients.md).

## Independence

{class}`telescopic.kernels.Independence` proposes $x' \sim q$ whatever the current
state, and accepts when
$\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]$.

- The coupled step gives both chains the same proposal and the same uniform, so a pair
  meets when both accept (a maximal coupling of the kernel; O'Leary, Wang & Jacob 2021).
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
  the meeting probability is capped at `eta` and the number of draws it takes has a
  bounded variance (Atchadé & Jacob 2024, Algorithm 2).
- `coupling="crn"`: common random numbers, one uniform through `ppf` for a block of one
  coordinate, or common standard normals for a multivariate normal conditional (an
  object with `mean` and `cov`). They can bring the chains together, but make them equal
  only by rounding; `"two-scale"` adds the maximal coupling.
- `coupling="two-scale"`: common random numbers for pairs farther apart than `threshold`
  at the start of a sweep, the maximal coupling for the others (Biswas, Bhattacharya,
  Jacob & Johndrow 2022).

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
coupling. The choice never depends on the states, so each chain moves by the mixture
(Atchadé & Jacob 2024, §4.2). `tune` tunes the untuned components on one pilot. HMC is a
mixture of its trajectories and its meeting step.

## Choosing a kernel

- **The random walk**, the default, needs only the log-density and has no step size to
  tune.
- **MALA and HMC** need the gradient. A step evaluates the log-density and the gradient
  once (MALA) or `n_steps` times (HMC). Check their meeting times: the automatic steps
  can fail to couple.
- **The independence kernel** suits a target that a Student-$t$ approximates well, such
  as a posterior close to Gaussian.
- **Gibbs** suits models whose full conditionals are known, such as conjugate
  hierarchical models.

Coupling does not make a kernel mix faster ([Theory](../theory.md)): a kernel that mixes
slowly meets slowly. To compare kernels on a target, compare the cost of a given
precision: the variance of a replicate's estimate times the cost of a replicate, counted
here in evaluations of the log-density and the gradient. A run of each gives both:

```{code-cell} python
for name in ["rwmh", "mala", "hmc", "independence"]:
    result = telescopic.sample(
        gaussian.log_prob, gaussian.initial, grad=gaussian.grad, vectorize=True,
        kernel=name, rng=1,
    )
    evaluations = (result.n_log_prob + result.n_grad) / result.n_replicates
    variance = result.n_replicates * result.mean().standard_error[0] ** 2
    print(
        f"{name:>12}: k = {result.k:3}, {evaluations:5.0f} evaluations and variance "
        f"{variance:.4f} per replicate, product {evaluations * variance:5.1f}"
    )
```

On this Gaussian the random walk takes the most evaluations for a given precision of the
mean of $x_0$. On other targets the order can change, and the variance of other
estimates can rank the kernels differently.
