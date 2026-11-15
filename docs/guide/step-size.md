---
file_format: mystnb
kernelspec:
  name: python3
---

# Check and set a step size

```{code-cell} python
:tags: [remove-cell]

import numpy as np
from scipy import optimize, special

import telescopic
from telescopic import kernels
```

## When the automatic step fails

Without `step_size`, MALA and HMC tune their step toward the acceptance rate that suits
the kernel alone, on chains near the bulk of the target, and Heng & Jacob (2019, §5)
show that settings that suit HMC alone can fail to bring coupled chains together. On
targets that are not log-concave, such as the banana of
{func}`telescopic.examples.banana` from its default initial distribution, the automatic
steps of MALA and HMC leave pilot {term}`pairs <pair>` apart. Log-concave targets can fail too, where
the log-density curves more sharply than where the step was tuned. Where the largest
eigenvalue of $\Sigma H$, with $\Sigma$ = `cov` and $H$ the Hessian of $-\log\pi$,
exceeds $4/\epsilon^2$, MALA's proposals overshoot: on a Gaussian of that Hessian, the
proposal mean lands farther from the mode than the chain. HMC's leapfrog steps are
unstable beyond the same bound for the largest step a trajectory can take,
$(1 + j)\epsilon$ with $j$ = `jitter`: from an eigenvalue of $4/((1 + j)\epsilon)^2$,
about $1.78/\epsilon^2$ at the default $j = 0.5$, some trajectories are unstable. A
chain that starts in such a region can reject every proposal. On the logistic
regression below, whose predictors nearly separate the outcomes, the automatic MALA
leaves pilot pairs unmet. The symptom is pilot pairs still
unmet at the cap, so that `sample` raises `ValueError` once the pilot has run, or very
long {term}`meeting times <meeting time>`. The remedy is an explicit `step_size`, checked as in
[Checking a step size](#checking-a-step-size). On the banana, Heng & Jacob (2019, §5.2)
report a mean meeting time of 158 with a fixed step of 1/500 and 500 leapfrog steps,
`HMC(1/500, 500, cov=np.eye(2), jitter=0)`.

HMC's automatic step could also fail on a Gaussian, through a resonance of the
leapfrog, which the default jitter of the step removes. With a shared momentum, an
accepted trajectory of $n$ = `n_steps` steps of size $\epsilon$ multiplies the separation
of the two chains along an eigenvector of $\Sigma H$ with eigenvalue
$w^2 < 4/\epsilon^2$ by $\cos(n\theta)$, with $\theta = 2\arcsin(\epsilon w/2)$. Where
$n\theta$ is close to a multiple of $\pi$ the trajectory brings the chains no closer along
that direction, and the tuning, which looks only at the acceptance rate, can settle on
such a step. `HMC` therefore draws the step of each trajectory uniformly between
$(1 - j)\epsilon$ and $(1 + j)\epsilon$, with $j$ = `jitter`, 0.5 by default, the same for
both chains of a pair, and keeps `n_steps` fixed. The phase $n\theta$ then changes from
one trajectory to the next, so that no direction stays on a resonance.
With `jitter=0`, the step is fixed, and two settings expose the resonance:

- **A given `n_steps` with an automatic step.** On {func}`telescopic.examples.gaussian`
  in 10 dimensions with `HMC(n_steps=3)` and a fixed step, over 10 seeds, `k` could not
  be chosen at 2 of them and was 12,377 to 52,065 at the others, against 83 to 113 with
  the fixed step and the automatic `n_steps`.
- **A given `cov` that does not whiten the target**, which leaves eigenvalues that differ.
  On a Gaussian in 10 dimensions whose scales run geometrically from 1 to 1/4, started
  from $\mathcal N(0, 4I)$, with `HMC(cov=np.eye(10))` and a fixed step, over 10 seeds,
  `k` could not be chosen at 3 of them and was 1,566 to 10,134 at the others, against
  50 to 65 with the fixed step and the automatic `cov`.

With the default jitter, `k` was chosen at each of 30 seeds of each of these settings,
and of `HMC(n_steps=10)` on the same Gaussian in 10 dimensions, where the fixed step
could not choose it at 2 of 10 seeds. The jitter costs little where there is no
resonance: on {func}`telescopic.examples.gaussian` in 2 and 10 dimensions, at each
`n_steps` where the fixed step does not resonate, the pilot's median meeting time times
the gradients per iteration, a geometric mean over 30 seeds, was at most 1.17 times that
of the fixed step. The {term}`inefficiency`, the variance of a {term}`replicate`'s estimate times its
cost in gradients, a ratio of medians over seeds, was at most 1.31 times that of the
fixed step, on {func}`telescopic.examples.mixture` with `n_steps=2`, and 1.13 times on
the logistic regression below, over the seeds where the fixed step could choose `k`.

The jitter does not cure every failure: on the banana from its default initial
distribution, pilot pairs are still unmet at the cap. All of these runs ran the pilot
with its default cap.

Setup of these runs: telescopic at commits 7c6b0d1 (the first example with a fixed
step), ce89da8 (the second) and 3bf5f62 (the comparison with the jitter), Python
3.13.18, NumPy 2.5.4 (2.5.3 at 3bf5f62) and SciPy 1.18.1, `OMP_NUM_THREADS=1`, on an
Intel Xeon processor at 2.10 GHz with 4 logical cores.

## Checking a step size

Pilot pairs check a step size before a run. With every parameter of the kernel given,
{meth}`telescopic.CoupledSampler.meeting_times` runs no tuning, only pairs with {term}`lag` 1
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
