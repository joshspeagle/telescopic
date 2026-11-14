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
  A given `n_steps` can fail on a Gaussian too (issue #137): the tuned step can land on
  a leapfrog resonance, where one trajectory with a shared momentum multiplies the
  distance between the chains along a mode by a factor within 1e-4 of 1 in absolute
  value. With a given covariance that does not whiten the target, the default `n_steps`
  fails as well: on a Gaussian of scales from 1 to 1/4 in 10 dimensions with
  `HMC(cov=np.eye(10))`, `k` could not be chosen at 3 of 10 seeds at the package's cap,
  reproduced by an agent that did not find it (issue #137). Of four remedies tested
  with pre-registered criteria (`notes/experiments/i137_remedies.out`,
  `i137_nongaussian.out`, `i137_unequal.out`), a guard that moves the tuned step off a
  resonance passed and costs no evaluation, but does not fix the case of the given
  covariance; a step jittered by a uniform that both chains share fixed every case and
  cost up to 29% more per effective replicate where there is no resonance. Default: no
  change to the package; the guard and the jitter wait for Josh's ruling on issue #137,
  and the jitter enters the pre-registered test of issue #91.
