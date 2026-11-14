# Open questions for Josh

Each item has the default that work proceeds on until Josh answers.

- [ ] **Budget mode (spec §3.7).** It is a should-level feature, and an unbiased per-lane
  estimator needs per-replicate weights throughout `Result`. Default: deferred beyond
  v0.1 (DECISIONS #22); an interrupted run and its completion cover stopping early.
  Josh ruled on issue #79 (2026-11-13): prototype it as an experiment, then decide.
- [ ] **The automatic steps of HMC and MALA.** Dual averaging toward
  acceptance 0.8 suits ordinary HMC, but its coupled chains then rarely meet on the
  banana from U[-5, 5]^2: 88 of 100 pilot pairs were unmet after 10^5 iterations, so
  `sample(kernel="hmc")` cannot choose k there. A target acceptance of 0.95, or a meeting
  scale of half the step, left 59 to 94 of 100 unmet at 20,000; steps near 0.02 couple
  (mean meeting time 160), and on the Gaussian every variant meets within a median of 32
  (`notes/experiments/m9_hmc_meeting.out`). Choosing the step from pilot meeting times, as
  Heng & Jacob (2019) advise, would be a new tuning procedure (DECISIONS #1). Default:
  keep the defaults, which couple on log-concave targets, and document that other targets
  need an explicit step; the error of `choose_k` already says to give a better-tuned
  kernel. The automatic MALA fails the same way on the banana: more than 1% of its pilot
  pairs were unmet at 10^5 (`benchmarks/results/acb12f9.json`). It can fail on a
  log-concave target too: on a logistic regression with 200 outcomes, started ten times
  wider than the fit, 19 of 100 pilot pairs were unmet at 3,000 and `sample` raised. Of
  200 chains at the tuned step, 21 never moved; they started where the curvature was
  higher (the largest eigenvalue of ΣH a median of 3.5, against 2.1 for the others and 1.2
  at the mode). A step of 0.7 met every pair, and the automatic HMC met them all
  (`notes/experiments/m10_mala_logistic.out`).
  Josh ruled on issue #89 (2026-11-13): an opt-in tuning that puts contraction first
  (issue #91), which becomes the default if its pre-registered test passes.
  A given `n_steps`, or a given covariance that does not whiten the target, can fail on
  a Gaussian too: see the item on issue #137 below.
- [ ] **Which remedy for the leapfrog resonance, if any, enters the package (issue
  #137).** With an automatic step, `HMC(n_steps=n)` can tune its step onto a leapfrog
  resonance, where one trajectory with a shared momentum barely shrinks the distance
  between the chains along a mode: at a cap of 10,000, `k` could not be chosen at 12,
  29 and 11 of 30 seeds of `examples.gaussian(2)` with `n_steps=2` and
  `examples.gaussian(10)` with `n_steps=3` and 10, and the predicted factor
  |cos(n theta)| of a failing seed was at least 0.9967
  (`notes/experiments/i137_baseline.out`). The default `n_steps` fails too when a given
  covariance does not whiten the target: on a Gaussian of scales from 1 to 1/4 in 10
  dimensions with `HMC(cov=np.eye(10))`, `k` could not be chosen at 3 of 10 seeds at the
  package's cap of 100,000 (`i137_reproduce_unequal.out`). Four remedies were tested
  against tests that, by the author's account, were written before the runs with pilot
  pairs (`i137_predictions.md`, which also scores its predictions); two remain
  candidates. A guard that moves the tuned step, on a grid of steps from e^-2 to e^0.1
  times the tuned one, to the nearest at which the predicted factor is at most 0.9,
  costs no evaluation and passed (`i137_remedies.out`, `i137_nongaussian.out`). It
  lowers the step near a resonance; it raises it where the trajectory is too short to
  contract (1 of 5 seeds of the banana with `n_steps=2`); and where no step of its grid
  meets its rule, it falls back to the step of smallest predicted factor (another of
  those 5 seeds). On the Gaussian with unequal scales no step of its grid meets the
  rule, at any `n_steps`: a step meeting it exists only for frequencies spanning a ratio
  of at most 2.05 to 3.50, depending on `n_steps`, and this target spans 4
  (`i137_unequal.out`). With the default `n_steps` its fallback chose among steps whose
  predicted factor is 0.9993 to 0.9998, and `k` could not be chosen at 3 of 10 seeds;
  the pilot's frequencies, within 4.5 to 12% of the exact ones, are not the cause. A
  step jittered by a uniform that both chains share gave `k` at every seed of every
  configuration (`i137_remedies.out`, `i137_unequal.out`), and failed the test of cost
  where there is no resonance on 2 of 3 cells: a median meeting time of 83.1 against
  70.4 on the banana with `n_steps=2`, and an inefficiency of 57,218 against 44,339 on
  the mixture (`i137_nongaussian.out`).
  Default: no change to the package's kernels or tuning until Josh rules on issue #137;
  meanwhile the kernels guide and the advice of the error raised when `k` cannot be
  chosen name this case (an automatic step with a given `n_steps`, or with a given
  covariance that does not whiten the target). Recommended: the guard as tested is not
  adopted, since it also lowers steps where the package already gives `k` (4 of the 12
  Gaussian configurations, `i137_remedies.out`) and has no step to return where the
  frequencies span too wide a ratio; a guard that acts only near a resonance and the
  jitter both enter the pre-registered test of issue #91, whose predictions are posted on
  that issue before it runs.
