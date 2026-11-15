# Is telescopic right for my problem?

telescopic suits some problems better than others. Find yours in the table: each row
says whether telescopic is a good or a poor fit, why, and which page holds the fact. A
short list of decisions follows.

| Your problem | Fit, and why |
|---|---|
| A posterior with one mode, and a log-density that is cheap or takes a batch of points | **Good.** Each replicate is a full MCMC run with its burn-in, so a run evaluates the log-density many times; with `vectorize=True` a step of many chains costs one call ([The log-density](../guide/log-density.md#the-log-density)). |
| Many processor cores, or several machines | **Good.** The replicates are independent and run in parallel without communicating ([Run in parallel](../guide/parallel.md)), and runs made apart can be merged ([Merging runs](../guide/save-load-merge.md#merging-runs)). |
| Error bars to report without judging convergence | **Good.** The error bar is the standard error of independent, unbiased replicates, with no autocorrelation time and no R-hat behind it ([Error bars from replicates](../concepts/error-bars.md)). |
| A gradient of the log-density | **Good.** MALA and HMC use it, and on a Gaussian in 50 dimensions they reach the same error bars with fewer evaluations than the random walk ([Gradients](../tutorials/gradients.md)). |
| Full conditionals that can be drawn from, as in conjugate hierarchical models | **Good.** Gibbs kernels come with three couplings ([Gibbs couplings](../tutorials/gibbs.md)). |
| Means and covariances in many dimensions, from many replicates | **Good.** With `keep`, a run stores the points of a subset of the replicates and the moments of all of them, so the means and variances, and up to 100 dimensions the covariances, still come from every replicate ([Keeping a subset](../guide/settings.md#keeping-a-subset)). |
| How long the burn-in of your own sampler should be | **Good, if its kernel can be coupled.** The meeting times bound the distance between the chain and the target at each time ([Meeting times](../concepts/meeting-times.md)); a kernel of your own needs a coupled step ([Write a kernel](../guide/custom-kernels.md)). |
| Several separated modes | **Poor, unless the initial distribution covers every mode and the kernel crosses between them.** Pairs that start in one mode meet there, and nothing in the run shows the mode they missed ([Choose an initial distribution](../guide/initial-distribution.md#choosing-one), [When unbiased is not enough](../tutorials/failure-modes.md#two-modes)). |
| A kernel that mixes slowly on your target | **Poor.** Coupling does not make chains mix faster, so pairs meet late and replicates cost more ([Theory](../theory.md#guarantees-and-their-limits)). |
| A slow log-density | **Poor with the automatic settings.** Tuning and the meeting-time pilot evaluate the log-density thousands of times before the replicates; giving the kernel's parameters and `k` skips both ([A slow log-density](../guide/settings.md#a-slow-log-density)). |
| Many dimensions, with the random walk alone | **Poor.** On a Gaussian in 50 dimensions, the random walk's pairs meet later than those of MALA and HMC, and it needs more evaluations for the same error bars ([Gradients](../tutorials/gradients.md#a-gaussian-in-50-dimensions)). |
| A target that is not log-concave, with MALA or HMC | **Poor with the automatic step size.** It can leave pilot pairs apart, and the run then stops with an error, until a step size is given. HMC's default jitter of its step prevents a failure on Gaussians, but not this one ([When the automatic step fails](../guide/step-size.md#when-the-automatic-step-fails)). |
| Unweighted draws to pass to corner, ArviZ or a resampler | **Poor.** The points come with weights, some negative, and are not draws ([Signed measures, not samples](../concepts/signed-measures.md#what-the-atoms-cannot-give)). {meth}`~telescopic.Result.chains` returns ordinary MCMC chains after the burn-in, which carry its bias. |

## Deciding

- **Try telescopic** when you need an expectation with an error bar that you can report
  without judging convergence, and either the target has one mode or your initial
  distribution covers every mode. Start with the [Quickstart](quickstart.md), or with
  [Coming from emcee](coming-from-emcee.md) to move an emcee model.
- **Give the settings yourself** when the log-density is slow: a tuned kernel and `k`
  from an earlier run skip the pilots
  ([A slow log-density](../guide/settings.md#a-slow-log-density)).
- **Use a gradient** when you have one and the dimension is large, and check the step
  size ([Check and set a step size](../guide/step-size.md)).
- **Fix the number of replicates before looking at the results.** A run can be extended
  or continued ([Continuing a run](../guide/save-load-merge.md#continuing-a-run)), but
  extending it until the error bar is small enough leaves the estimates consistent, not
  exactly unbiased ([Theory](../theory.md#guarantees-and-their-limits)).
- **Keep another sampler** when you need unweighted draws, or when no initial
  distribution you can build covers every mode of the target.
