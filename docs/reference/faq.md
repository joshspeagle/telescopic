# FAQ

Each answer is a paragraph that links to the page holding the fact.

## Results

### Can I plot the atoms with corner or ArviZ?

Not as samples. The {term}`atoms <atom>` carry weights, some of them negative, so a
plot without the weights counts them wrongly, and corner and ArviZ compute weighted
quantiles that assume non-negative weights
([What the atoms cannot give](../concepts/signed-measures.md#what-the-atoms-cannot-give)).
{meth}`~telescopic.Result.plot_marginals` and {meth}`~telescopic.Result.plot_corner`
draw the signed histograms instead ([Plot a result](../guide/plots.md)), and
{meth}`~telescopic.Result.chains` returns the ordinary MCMC chains of the kept
{term}`replicates <replicate>`, without the bias correction, for tools that need draws.

### Why is my variance negative, or a probability outside [0, 1]?

An estimate is a weighted sum over the atoms, and some weights of the
{term}`signed measures <signed measure>` are negative, so an {term}`unbiased` estimate
can fall outside the range of the function averaged
([What the atoms give](../concepts/signed-measures.md#what-the-atoms-give)).
{meth}`~telescopic.Result.var` and {meth}`~telescopic.Result.cov` are unbiased, and a
start far wider than the target can make a variance negative;
{meth}`~telescopic.Result.std` returns $\sqrt{\max(0, \widehat{\mathrm{Var}})}$, which
is {term}`consistent` but not unbiased
([Covariances and variances](../guide/estimates.md#covariances-and-variances)).

### Do quantiles and credible intervals have error bars?

No. They come from the atoms of all replicates pooled, and are consistent as the number
of replicates grows, but not unbiased and without a standard error
([Quantiles and credible intervals](../guide/estimates.md#quantiles-and-credible-intervals)).

## Running

### How many replicates do I need?

The standard errors fall as $1/\sqrt R$; to reach a target error, size a second run
from the standard error of a first one, and fix the number before looking at the results
([How many replicates](../guide/settings.md#how-many-replicates)).

### Can I add replicates to a run, or continue one that stopped?

Yes: a second `sampler.run(n)` on the same sampler adds `n` replicates;
{meth}`telescopic.CoupledSampler.resume` makes a new sampler continue a returned or
loaded result with its seed, settings, pilot and tuning, first running the replicates
that an interrupted run never started
([Continuing a run](../guide/save-load-merge.md#continuing-a-run)), and
{meth}`~telescopic.Result.merge` combines runs made apart
([Merging runs](../guide/save-load-merge.md#merging-runs)). Extending a run until its
error bar is small enough leaves the estimates consistent, not exactly unbiased
([Theory](../theory.md#guarantees-and-their-limits)).

### My run uses too much memory. What can I store less of?

`thin` stores about one in `thin` of the averaged states, with the same chains and
unbiased estimates ([Memory and thinning](../guide/settings.md#memory-and-thinning)).
`keep` stores the signed measures of the replicates of index below it only, and the
moments of every replicate, so the means, the variances and, up to 100 dimensions, the
covariance still come from every replicate
([Keeping a subset](../guide/settings.md#keeping-a-subset)).

### Can I start from the walkers of an emcee run?

Yes: an array of points of shape `(n, d)`, such as the last state of an ensemble, is
taken as the uniform distribution over those points
([Initial distributions](../guide/initial-distribution.md#initial-distributions)). The
estimates stay unbiased, but the {term}`pairs <pair>` then cover only the modes that the
walkers found
([Choosing one](../guide/initial-distribution.md#choosing-one)).

### Is it slower than emcee?

Each replicate is a full MCMC run with its own {term}`burn-in`, and the automatic
settings first run the {term}`tuning pilot` and the {term}`meeting-time pilot`.
[Coming from emcee](../start/coming-from-emcee.md#side-by-side) compares the error bars
of the two at the same number of log-density evaluations on one target, and
[A slow log-density](../guide/settings.md#a-slow-log-density) shows how to skip the
pilots.

### Does it work with JAX?

Yes, through a log-density, a gradient or `log_prob_and_grad` that JAX computes, such as
`jax.value_and_grad`, with `jax_enable_x64` set, since telescopic computes in float64
([The value and the gradient in one call](../guide/log-density.md#the-value-and-the-gradient-in-one-call)).

### Why does HMC vary its step from one trajectory to the next?

So that no step stays on a resonance of the leapfrog, where coupled trajectories with a
fixed step do not bring the chains closer. The argument `jitter` sets how much the step
varies, and [HMC](../guide/kernels.md#hmc) gives the rule and the default;
[When the automatic step fails](../guide/step-size.md#when-the-automatic-step-fails)
describes the failure it prevents.

### Is a run reproducible from its seed?

Each replicate draws from a stream of its own, derived from the root seed and its index.
With a log-density evaluated point by point, the replicates are bitwise identical in any
pool; with a vectorized one, only with the same pool, workers, BLAS build and threads
([Reproducibility](../guide/parallel.md#reproducibility)).

## When something goes wrong

### Why does my run stop with a `ValueError` before any replicate?

When its message says that `k` cannot be chosen, too many pilot pairs never met;
[k cannot be chosen](../guide/troubleshooting.md#k-cannot-be-chosen) gives the causes,
a check and the fixes.

### My run warns. Are the estimates wrong?

Each warning names a condition under which the error bars may be unreliable, and its
remedy ([Warnings and errors](warnings.md#warnings)). A run without warnings is not
shown to be right: pairs that never leave one mode can meet quickly there
([A run that never left one mode](warnings.md#a-run-that-never-left-one-mode)).

### Do I still need R-hat and the effective sample size?

Not for the error bars, which are the standard error of independent replicates, nor for
the bias, which the burn-in no longer decides
([Error bars from replicates](../concepts/error-bars.md)).
