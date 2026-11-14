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
string that names it. In the coupling column, a **maximal** coupling makes two draws
equal with the largest probability that their two distributions allow. A
**reflection-maximal** coupling does so for two Gaussians with one covariance, and when
the draws differ, the offset of the second from its mean mirrors that of the first. A
**common** uniform, momentum or random number is one draw that both chains use:

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
  ([When the automatic step fails](#when-the-automatic-step-fails)). A wrong gradient
  leaves them exact but slow, so a sampler built with one compares `grad` with
  differences of `log_prob` first
  ([Checking the gradient](targets.md#checking-the-gradient)).
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

The four chains share one proposal covariance, and on a target with well-separated modes
they can all settle in one mode during their descent, before they reach it: on the
two-mode target of
[A tuned proposal that covers one mode](../tutorials/failure-modes.md#a-tuned-proposal-that-covers-one-mode),
started from $\mathcal N(0, 100I)$, in 15 of 20 runs in 10 dimensions and 18 of 20 in 32.
The tuned parameters then describe that mode alone, and the chains' $\hat R$ stays near
1, so the warning about disagreeing tuning chains
([Diagnostics and warnings](diagnostics.md#disagreeing-tuning-chains-tuning)) is silent.

Setup of these runs: telescopic at commit d014ac1, Python 3.13.18, NumPy 2.5.4 and SciPy
1.18.1, `OMP_NUM_THREADS=1`, on an Intel Xeon processor at 2.10 GHz with 4 logical cores.

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
  matrix;
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
pair: its evaluations per step times the mean meeting time, infinite when a pair is
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
per_step = 4  # a pair's step: a log-density and a gradient for each chain
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
$\log u < [\log\pi(x') - \log q(x')] - [\log\pi(x) - \log q(x)]$. The weight $\pi/q$
is 0 wherever $\pi$ is 0, even where $q$ is 0 too, so a chain at zero density accepts
any proposal of finite density, as the random walk does. A proposal where $q$ is 0 is
rejected, even where $\pi$ is not: `sample` drew a point that `log_density` gives no
mass, such as a draw clipped onto the edge of an open support.

- The coupled step gives both chains the same proposal and the same uniform, so a pair
  meets when both accept. It is a maximal coupling of the kernel (O'Leary, Wang & Jacob
  2021).
- `proposal` is a {class}`telescopic.kernels.MultivariateT`, a Student-$t$ with location
  `mean`, scale matrix `cov` and `df` degrees of freedom, a positive integer (`df=None`
  gives a Gaussian). A proposal of your own has `sample(draws)` and `log_density(x)`,
  the normalized density of what `sample` draws, positive wherever `sample` can land; a
  `log_density` that returns NaN or `+inf` raises `ValueError` with the point.
- `tune` sets a Student-$t$ with 5 degrees of freedom, located at the pilot's mean, with
  the pilot's covariance as scale matrix. The report's `max_weight_ratio` is the largest
  weight $\pi/q$ among the pilot's states, relative to their median; a large value
  announces heavy-tailed meeting times.
- A tuned proposal covers only the modes that the tuning chains reached. On a target with
  well-separated modes, both chains of a pair then accept proposals in that mode and
  meet within a few tens of steps, and the estimates can be confidently wrong, in some
  runs without any warning: fast meeting says nothing about a region the proposal does
  not cover
  ([A tuned proposal that covers one mode](../tutorials/failure-modes.md#a-tuned-proposal-that-covers-one-mode)).
  When the modes are known, give `proposal` explicitly so that it covers them all.

The proposal must be positive wherever the target is. The kernel never moves a chain to
where $q$ is 0, so, used alone, a proposal that misses part of the target's support
never reaches that part, and the estimates are biased from any start, without a
warning. On $\mathcal N(0, 1)$ with
a proposal uniform on $(-2, 2)$, 500 replicates started in $(-1, 1)$ estimate
$E[x^2] = 1$ as 0.778 with a standard error of 0.0093, which agrees within its error
with the value 0.774 of $\mathcal N(0, 1)$ restricted to $(-2, 2)$. A chain started
where the target is positive and the proposal is 0 rejects every proposal: started at
3, the same run reports $E[x] = 3$ with a standard error of 0. Inside a
{class}`telescopic.kernels.Mixture` with a kernel that reaches the rest of the support,
chains do arrive at such points: the independence component rejects every proposal from
there, which is the rule that keeps the target invariant, and the other kernel moves
them on.

The meeting times have geometric tails only when the weights $\pi/q$ are bounded, so the
proposal needs tails at least as heavy as the target's. With unbounded weights the
sampler is not geometrically ergodic (Mengersen & Tweedie 1996): a chain that reaches a
point of weight $w$, for $\pi$ and $q$ normalized, stays there for $w$ steps or more on
average. The replicates that carry the bias correction for the region where $\pi/q$ is
large are then too rare to occur in a run, and the variance of the estimates can be
infinite. A run usually gives that region too little mass, with error bars that are too
small and often without a warning.

On the target Exp(1), whose mean is 1, with a normal proposal $\mathcal N(1, 1)$
floored at 0, a chain stays for 51 steps on average at $x = 5$ and for 1,671 at
$x = 6$. Of 200 runs of 1,000 replicates started in $(0, 2)$, the median estimate of
the mean was 0.970, 72.5% of the estimates were more than two standard errors below 1
and none was that far above, and 142 runs raised no warning, 135 of them with an
estimate that low. With a Student-$t$ proposal with 3 degrees of freedom in place of the
normal, whose weights are bounded, the median was 0.999 and 1.5% of the estimates were
that low.

Setup of these runs: telescopic at commit d014ac1, Python 3.13.18, NumPy 2.5.4 and SciPy
1.18.1, `OMP_NUM_THREADS=1`, on an Intel Xeon processor at 2.10 GHz with 4 logical cores.

## Gibbs

{class}`telescopic.kernels.Gibbs` updates blocks of coordinates in turn, each drawn from
its full conditional. `conditional(x, b)` receives a read-only point `x` of shape `(d,)`
and a block number `b`, and returns the distribution of block `b` given the other
coordinates of `x`: an object with `rvs(random_state=...)` and a normalized `logpdf`, or
`logpmf`, such as a frozen SciPy distribution. `blocks` lists index arrays that
partition the coordinates `0, ..., d - 1`. The coupled step couples each block's two
conditionals:

- `coupling="maximal"`, the default: the rejection maximal coupling. With `eta` below 1
  the meeting probability is capped at `eta`, and the number of draws it takes has the
  bounded variance of Atchadé & Jacob (2024, Algorithm 2).
- `coupling="crn"`: common random numbers. For a block of one coordinate whose two
  conditionals have `ppf`, one uniform goes through both. For any other block, each
  chain draws from its own conditional with `rvs`, from generators in the same state, so
  each keeps its conditional's law whatever its type. Common random numbers can bring
  the chains together, but make them equal only by rounding; `"two-scale"` adds the
  maximal coupling.
- `coupling="two-scale"`: common random numbers for pairs farther apart than `threshold`
  at the start of a sweep, and the maximal coupling for the others, as in Biswas,
  Bhattacharya, Jacob & Johndrow (2022).

Where the two chains agree on every coordinate outside a block, both get the same draw.

Under `"crn"`, the two draws of a block drawn with `rvs` are close when `rvs` reads a
fixed count of numbers from the generator and maps them smoothly in the conditional's
parameters. A normal block drawn as
`mean + np.linalg.cholesky(cov) @ random_state.standard_normal(k)`, or with
`random_state.multivariate_normal(mean, cov, method="cholesky")`, does: from
generators in the same state, draws of two normals of 3 coordinates whose covariances
differ in shape by 1% are 0.0056 apart on average, against 2.2 for independent draws.
NumPy's default method, a singular value decomposition of the covariance, which SciPy's
`multivariate_normal` calls, does not: its signs and order can change between nearby
covariances, so the same draws are 0.063 apart on average, and for covariances near a
multiple of the identity, 2.42 apart, as far as independent ones. In a robust
regression whose block of 3 coefficients has a normal conditional, pairs meet after a
median of 147.1 sweeps, averaged over 5 seeds, with NumPy's Cholesky method, as many as
with the Cholesky factor written out, and 205.4 with SciPy's `multivariate_normal`.

Setup of these runs: telescopic at commit 9be9024, Python 3.13.18, NumPy 2.5.4 and SciPy
1.18.1, `OMP_NUM_THREADS=1`, on an Intel Xeon processor at 2.10 GHz with 4 logical cores.

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
pilot, and the start of the chains checks that every component works in their
dimension. HMC is a mixture of its trajectories and its meeting step.
