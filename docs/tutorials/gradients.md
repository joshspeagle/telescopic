---
file_format: mystnb
kernelspec:
  name: python3
---

# Gradients: MALA and HMC

With the gradient of the log-density, telescopic can run the Metropolis-adjusted Langevin
algorithm ({class}`telescopic.kernels.MALA`) and Hamiltonian Monte Carlo
({class}`telescopic.kernels.HMC`), whose proposals follow the gradient. On targets like
the Gaussian below, their pairs of chains meet sooner than those of random-walk
Metropolis; the last section shows where their automatic settings fail. The
[kernels guide](../guide/kernels.md) describes each kernel and its coupling.

## A Gaussian in 50 dimensions

{func}`telescopic.examples.gaussian` gives a Gaussian whose neighbouring coordinates have
correlation 0.9, with its log-density, its gradient and the initial distribution
$\mathcal N(0, 4I)$. Its functions take batches of points, hence `vectorize=True`. The
three kernels run with the same seed, so their pairs start from the same points. The
random walk's runs are long, so it averages every tenth state (`thin=10`), which stores
about a tenth of the atoms; thinning does not shorten a run ([Theory](../theory.md)).

```{code-cell} python
import numpy as np
from myst_nb import glue

import telescopic
from telescopic import examples

gaussian = examples.gaussian(50)
results = {}
for kernel in ("rwmh", "mala", "hmc"):
    results[kernel] = telescopic.sample(
        gaussian.log_prob,
        gaussian.initial,
        grad=gaussian.grad,
        vectorize=True,
        kernel=kernel,
        thin=10 if kernel == "rwmh" else None,
        rng=1,
    )
```

For each kernel: the median meeting time of the 100 pilot pairs, which run with lag 1
(`Result.pilot`); the burn-in `k`, their 99% quantile minus 1; the inefficiency; and the
evaluations of the log-density and of its gradient per replicate in the main run.

```{code-cell} python
print("kernel  median tau     k  inefficiency  log_prob   grad")
inefficiency = []
for kernel, result in results.items():
    median = float(np.median(result.pilot))
    inefficiency.append(result.diagnostics()["inefficiency"])
    glue(f"median_{kernel}", median, display=False)
    print(
        f"{kernel:6}  {median:10.0f}  {result.k:4}  {inefficiency[-1]:12.3f}  "
        f"{result.n_log_prob / result.n_replicates:8.0f}  "
        f"{result.n_grad / result.n_replicates:5.0f}"
    )
glue("inefficiency_low", min(inefficiency), display=False)
glue("inefficiency_high", max(inefficiency), display=False)
```

With gradients the pairs meet in a median of {glue:text}`median_mala:.0f` (MALA) and
{glue:text}`median_hmc:.0f` (HMC) steps, against {glue:text}`median_rwmh:.0f` for the
random walk. `k`, the lag and the run length `m` shrink with the meeting times, since the
lag equals `k` and `m = 10k`. The inefficiency is between
{glue:text}`inefficiency_low:.2f` and {glue:text}`inefficiency_high:.2f` for the three
kernels: it compares the cost of a replicate with that of the MCMC average of the same
kernel, and with `m = 10k` it stays near 10/9 to 11/9 ([Theory](../theory.md)). The
kernels differ in the steps a replicate takes and in what a step evaluates. MALA
evaluates the log-density and its gradient once per chain and step, HMC at each leapfrog
step of its trajectories, `n_steps` of them:

```{code-cell} python
results["hmc"].kernel
```

The replicates also differ in precision, which depends on how many states a replicate
averages and how correlated they are. A standard error falls as $1/\sqrt R$, so the
variance of one replicate's estimate of a mean, averaged over the 50 coordinates, gives
the replicates, and so the evaluations, that standard errors of 0.01 would take:

```{code-cell} python
print("kernel  replicates  log_prob")
evaluations = {}
for kernel, result in results.items():
    variance = result.n_replicates * np.mean(result.mean().standard_error**2)
    replicates = variance / 0.01**2
    evaluations[kernel] = float(replicates * result.n_log_prob / result.n_replicates)
    print(f"{kernel:6}  {replicates:10.0f}  {evaluations[kernel]:9,.0f}")
for kernel in ("mala", "hmc"):
    glue(f"fewer_{kernel}", evaluations["rwmh"] / evaluations[kernel], display=False)
```

For the same error bars, MALA and HMC evaluate the log-density
{glue:text}`fewer_mala:.0f` and {glue:text}`fewer_hmc:.0f` times less often than the
random walk, and the gradient as often as the log-density. Whether that saves time
depends on what a gradient costs.

## A logistic regression

Two hundred binary outcomes $y_i$ depend on five predictors through
$P(y_i = 1) = 1/(1 + e^{-x_i^\top\beta})$, where $x_i$ starts with a 1 for the intercept.
Each of the six coefficients has the prior $\mathcal N(0, 5^2)$. The log-density and its
gradient take a batch of points $\beta$ of shape `(n, 6)`, or one point of shape `(6,)`:

```{code-cell} python
from scipy import optimize, special

rng = np.random.default_rng(1)
X = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
y = (rng.random(200) < special.expit(X @ [0.5, 1.0, -0.5, 0.25, 0.0, -1.0])).astype(float)


def log_prob(beta, X, y):
    eta = beta @ X.T
    return np.sum(y * eta - np.logaddexp(0, eta), axis=-1) - np.sum(beta**2, axis=-1) / 50


def grad(beta, X, y):
    return (y - special.expit(beta @ X.T)) @ X - beta / 25
```

As in [Getting started](../getting-started.md), the pairs start from a Gaussian ten times
wider than an optimiser's error bars:

```{code-cell} python
fit = optimize.minimize(
    lambda beta: -log_prob(beta, X, y), np.zeros(6), jac=lambda beta: -grad(beta, X, y)
)
init = telescopic.initial.Normal(mean=fit.x, cov=100 * fit.hess_inv)
names = ["intercept", "b1", "b2", "b3", "b4", "b5"]
mala = telescopic.sample(
    log_prob, init, args=(X, y), grad=grad, vectorize=True, kernel="mala", names=names, rng=2
)
mala.summary()
```

```{code-cell} python
hmc = telescopic.sample(
    log_prob, init, args=(X, y), grad=grad, vectorize=True, kernel="hmc", names=names, rng=3
)
hmc.summary()
```

The two runs have different seeds, so their errors are independent, and the difference
of their estimates of each mean, divided by its standard error, should look like a draw
from $\mathcal N(0, 1)$:

```{code-cell} python
a, b = mala.mean(), hmc.mean()
z = (a.value - b.value) / np.hypot(a.standard_error, b.standard_error)
glue("largest_z", float(np.max(np.abs(z))), display=False)
dict(zip(names, z.round(2).tolist()))
```

The largest of the six in magnitude is {glue:text}`largest_z:.2f`.

## When the automatic step fails

Without `step_size`, MALA and HMC tune it by dual averaging toward an acceptance rate
that suits each kernel alone: 0.574 for MALA, 0.8 for HMC's trajectories. Settings that
suit a kernel alone can fail to bring coupled chains together (Heng & Jacob 2019, §5).
On {func}`telescopic.examples.banana`, whose log-density is not concave, the automatic
MALA and HMC leave more than 1% of the pilot pairs apart after $10^5$ iterations, so
`sample` cannot choose `k`: it raises `ValueError`, after a minute or more. A
log-concave target can fail too, where its log-density curves more sharply than near
the mode, around which the step was tuned: in a logistic regression whose predictors
nearly separate the two outcomes, MALA chains that start in such a region can reject
proposal after proposal. Such targets need a step size given explicitly, as in
`MALA(step_size=...)` or `HMC(step_size=...)`, checked against the meeting times of
pilot pairs from {meth}`telescopic.CoupledSampler.meeting_times`. The
[kernels guide](../guide/kernels.md) says more.
