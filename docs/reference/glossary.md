# Glossary

Each entry defines a term in a few sentences and links to the page that explains it.
The symbols are those of [Theory](../theory.md), and [Notation](notation.md) maps each
to its argument or attribute in the code.

```{glossary}
atom
  A point of a {term}`signed measure`, with its weight. The atoms of a replicate are
  the states of $X$ at the averaged times, with positive weights, and the states of both
  chains in the bias correction, those of $Y$ with negative weights. A result holds
  them in `result.measure.atoms` and their weights in `result.measure.weights`
  ([Signed measures, not samples](../concepts/signed-measures.md)).

burn-in
  The first averaged time $k$, the argument `k`: an estimate averages the states of $X$
  from time $k$ to $m$, and adds a bias correction for the pairs that meet after
  $k + L$. In telescopic the burn-in changes the cost and the variance of the
  estimates, not their expected value
  ([Why burn-in biases MCMC](../concepts/burn-in-bias.md),
  [The settings](../guide/settings.md#the-settings)).

censored pair
  A pair still unmet when $t$ reaches the cap `max_iter`. Its meeting time is `inf`,
  and it is left out of the estimates, of $R$ and of the signed measure, which biases
  the estimates. Without a cap, the default, no pair of a run is censored, but pairs of
  the {term}`meeting-time pilot` can be
  ([Censored pairs](warnings.md#censored-pairs-censored)).

common random numbers
  A coupling that gives the two chains the same random numbers, such as one uniform
  passed through both conditionals' quantile functions. It can bring the chains closer,
  but makes them equal only by rounding. A Gibbs kernel uses it with `coupling="crn"`,
  and HMC's trajectories share their step, momentum and uniform in the same way
  ([Gibbs](../guide/kernels.md#gibbs), [HMC](../guide/kernels.md#hmc)).

consistent
  Of an estimate: it converges to the quantity it estimates as the number of
  replicates $R$ grows, though its expected value at a finite $R$ can differ from it.
  The quantiles, the credible intervals and the standard deviations of telescopic are
  consistent, not {term}`unbiased`
  ([Theory](../theory.md#estimates-from-many-replicates)).

coupling
  A way to move two chains together, $\bar P$ in Theory: it draws the random numbers of
  their two moves jointly, while each chain on its own moves exactly by the kernel $P$.
  A coupling can make the two chains equal, which independent chains on a continuous
  space never are ([Two chains that meet](../concepts/two-chains.md#a-coupling)).

inefficiency
  $\mathbb E[C]\,\mathbb V[H_{k:m}]$, the mean cost of a replicate times the variance of
  its estimate: the figure to minimize when comparing kernels or settings, computed from
  `result.costs` and `result.estimate(h).replicates`
  ([Theory](../theory.md#cost), [Cost](../guide/settings.md#cost)).

lag
  $L$, the argument `lag`, at least 1: how many steps the second chain of a pair runs
  behind the first, so that $X_t$ is compared with $Y_{t-L}$
  ([The lag](../concepts/two-chains.md#the-lag)).

late pair
  A pair that meets after $k + L$: only such pairs can carry a bias correction.
  `result.diagnostics()["late_fraction"]` is the fraction of pairs that are late or
  censored, since a {term}`censored pair` is left out of the estimates and carries no
  correction, and the `late` warning fires when it is high
  ([Late pairs](warnings.md#late-pairs-late)).

maximal coupling
  A coupling of two distributions under which the two draws are equal with the largest
  probability that any coupling allows, one minus their total variation distance. The
  rejection maximal coupling of {func}`telescopic.couplings.maximal` is one, for two
  distributions given with a normalized density
  ([Coupling primitives](../guide/custom-kernels.md#coupling-primitives)).

meeting time
  $\tau$, the first time $t \ge L$ at which the two chains of a pair are equal,
  $X_t = Y_{t-L}$, compared with `==`. From then on the chains stay equal. A result
  holds one per pair in `result.meeting_times`
  ([Two chains that meet](../concepts/two-chains.md#equal-and-then-together),
  [Meeting times and distance to stationarity](../concepts/meeting-times.md)).

meeting-time pilot
  The pairs with lag 1 that the automatic settings run before the replicates, until
  they meet or reach a cap; the burn-in $k$ is chosen from their meeting times.
  `sampler.pilot` holds the meeting times of every pilot pair run so far
  ([The automatic pipeline](../guide/settings.md#the-automatic-pipeline)).

pair
  The two chains $X$ and $Y$ of a replicate. They start from independent draws of the
  initial distribution, move by the same kernel under a {term}`coupling`, and run
  with $Y$ the {term}`lag` $L$ behind $X$
  ([Two chains that meet](../concepts/two-chains.md)).

reflection-maximal coupling
  The {term}`maximal coupling` of two Gaussians with one covariance: it gives both
  chains the same proposal as often as two such draws can coincide, with probability
  $2\Phi(-\delta/2)$ for centers a whitened distance $\delta$ apart, and otherwise gives
  $Y$ the mirror image of $X$'s proposal. Random-walk Metropolis and MALA use it, and
  HMC in its meeting step
  ([Two chains that meet](../concepts/two-chains.md#a-coupling)).

relative cost
  $\mathbb E[C]/(m - k + 1)$, the mean cost of a replicate per averaged state, which
  `result.diagnostics()["relative_cost"]` holds
  ([Theory](../theory.md#cost), [Cost](../guide/settings.md#cost)).

replicate
  One {term}`pair` of coupled chains, run with random numbers of its own, and what it
  returns: its signed measure, its meeting time and its cost. Replicates are
  independent, and the error bar of an estimate is the standard error of the mean of
  their estimates. `result.n_replicates` counts the replicates whose pairs met, $R$
  ([Error bars from replicates](../concepts/error-bars.md#where-the-error-bar-comes-from)).

run length
  $m$, the argument `m`: the last averaged time. A replicate runs until $\max(\tau, m)$
  ([The settings](../guide/settings.md#the-settings)).

signed measure
  A set of {term}`atoms <atom>` with weights that sum to 1, some of which can be
  negative: a pair that meets within $k + L$ steps gives no negative weight. Each
  replicate returns one, $\hat\pi_r$, whose weighted sum of any $h$ at its atoms is the
  replicate's estimate of $\pi(h)$. The atoms are not draws from the target
  ([Signed measures, not samples](../concepts/signed-measures.md)).

tail index
  The power $\alpha$ at which the probability that $\tau - L + 1$ exceeds $t$ falls as
  $t^{-\alpha}$, estimated by Hill's estimator from the largest meeting times and held
  in `result.diagnostics()["tail_index"]`. Below 2 the meeting times have an infinite
  variance; the `tail` warning gives when the index is computed and when the warning
  fires ([Heavy-tailed meeting times](warnings.md#heavy-tailed-meeting-times-tail)).

tuning pilot
  The adaptive chains that tune the kernel's parameters left as `None` before any pair
  runs, followed for MALA and HMC by a tuning of `step_size`
  ([Choose a kernel](../guide/kernels.md)). The `tuning` warning checks their $\hat R$,
  the `wide` warning their spread against that of the initial distribution's draws,
  and `result.tuning` holds their report
  ([The automatic pipeline](../guide/settings.md#the-automatic-pipeline)).

unbiased
  Of an estimate: its expected value equals the quantity it estimates. The estimates of
  expectations, and of histograms with bins fixed before the run, are unbiased for every
  burn-in, under
  the conditions of [Theory](../theory.md#guarantees-and-their-limits), when the number
  of replicates is fixed before any result is seen
  ([The telescoping correction](../concepts/telescoping-correction.md#unbiased-for-any-burn-in)).
```
