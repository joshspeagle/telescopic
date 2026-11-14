# x137 predictions, written after part 1's tuning-only data and before any part-2 run

Written 2026-11-14, after the 1,200 tuning-only records of i137_baseline.py and before
the baseline's pilot pairs finished; no remedy had been run with pilot pairs except the
check_coupling runs (which measure marginals only) and one tuning smoke test at seed 0.

## Parameter set from part 1 (before part 2)
- Remedy a, BAND: the largest |w - 1| of a tuning's modes (w the exact whitened
  frequencies of the tuned covariance) has a 99% quantile of 0.093 at d = 2 and 0.254 at
  d = 10 (max 0.286), over 600 tunings each. BAND = 0.3: the guard takes every frequency
  within a factor 1.3 of the predicted one, which covers about 99% of the d = 10 tunings.
  BOUND = 0.9 (unchanged from the design). A wider band forces short trajectories for
  large n (for n = 10 the guard can only reach the window n theta < about 2.7 / 1.3).

## Derivation the predictions rest on
On a whitened Gaussian mode of frequency w, h = eps w, theta = 2 arcsin(h/2), the gap of
two chains with common momentum after an accepted trajectory is multiplied by
cos(n theta). At resonance (n theta = j pi) the endpoint does not depend on the momentum:
the uncoupled chain does not mix either. A shared jitter u of width j gives the
geometric-mean factor exp E log|cos(n theta(eps u w))|, about 1/2 once the phases sweep a
period, i.e. n theta j >= pi/2 approximately.

## Predictions (part 2)
1. Gaussian table (30 seeds, cap 1e4):
   - base: failures at d=2 n=2 (about 25-40% of seeds at cap 1e4: 38 of 100 tunings have
     factor > 0.999), d=10 n=3 (about 60-70%), d=10 n=10 (about 30%); none elsewhere
     except possibly d=10 n=5 (max factor 0.998).
   - a: 0 failures at d=2; at most 2 per configuration at d=10. Median meeting times at
     d=2 n=2 of tens to about 100 iterations. Cost: no extra evaluation. Weak point: at
     d=10 n=10 the guard shortens the trajectory a lot (eps about 0.2).
   - b: 0 failures anywhere; picks n theta near pi/2, so the lowest median meeting times
     at n >= 3; tuning gradients up by about 64% (2,560 n on top of about 4,000 n + 8,000).
   - c: 0 failures anywhere; median meeting times between a/b and the default's.
   - d: identical to c for n <= 3 at d = 2 (width capped at 0.5) except where the rule
     binds; at n = 5, 10 narrower jitter, 0 failures, similar meeting times to c.
2. Non-Gaussian (banana from exact draws, mixture; n_steps 2 and 5; 5 seeds):
   - a: identical to base (guard does not move eps where n theta is far from resonance).
   - b: within the spread on banana; uncertain on the mixture, where the whitened
     curvature inside a mode is about 4 times the pilot's prediction.
   - c, d: median meeting time within the base's seed spread; inefficiency within it.
3. Unequal scales (d=10, cov = I, w from 1 to 4; n default, 3, 10; 10 seeds):
   - base: failures at n=10 at some seeds; default n passes.
   - a: the guard finds no step passing for n=10 (fallback to the least-bad step) and may
     fail there; predicted from the pilot frequencies, it is otherwise like base or better.
   - b, c, d: 0-1 failures per configuration.
4. Recommendation expected: c (or d), with a package change limited to a given n_steps,
   unless b is clearly cheaper per pilot pair at matched evaluations.

## Falsification tests (fixed now)
- Gaussian table: a remedy fails if k cannot be chosen at more than 2 of 30 seeds in any
  configuration (the default HMC() gives k at every seed on both targets).
- Non-Gaussian: a remedy fails on a target and n_steps if the mean over 5 seeds of the
  seed's median pilot meeting time, or of the inefficiency of h = x0 (SE^2 x replicate
  gradients), exceeds the base mean by more than the standard deviation of the base's
  5 per-seed values.

## Notes added after the runs (2026-11-14, after the review of the pull request)

The text above is unchanged. These notes say what in it was wrong or unclear, and how the
outcomes compare with each prediction.

### What the text above leaves unclear or gets wrong
- The heading's "x137" is the label of this experiment on issue #137. Part 1 is the 1,200
  tuning-only records of `i137_baseline.py`; part 2 is every run with pilot pairs (the
  baseline's pilots, `i137_remedies.py`, `i137_nongaussian.py` and `i137_unequal.py`).
- The order of writing cannot be checked: this file, the scripts and their outputs
  arrived in one commit, so that the predictions were written before the runs with pilot
  pairs rests on the author's statement. A later test (the one of issue #91) can post
  its predictions on its issue before it runs.
- BAND: the guard multiplies each predicted frequency by factors from 1/1.3 to 1.3, so
  its band reaches 1 - 1/1.3 = 0.231 below a frequency and 0.3 above it, and the 99%
  quantile of 0.254 does not measure its coverage. As coded, every mode of a tuning lies
  inside the band at 558 of the 600 tunings at d = 10 (93%), not about 99%, and at 600
  of 600 at d = 2 (`i137_baseline.out`). No outcome depends on it: at the steps the
  guard returned, the exact largest factor is at most 0.835 in every configuration of
  the Gaussian table (`i137_remedies.out`), inside the bound 0.9.
- Prediction 1 for b: the package's tuning costs 4,000 n + 4 gradients, not
  4,000 n + 8,000 (4,004 at n_steps = 1, 8,004 at 2, 40,004 at 10, `i137_baseline.out`),
  and b adds 2,560 n + 256 to it (2,816 at n_steps = 1 and 25,856 at 10, for totals of
  6,820 and 65,860, `i137_remedies.out`).
- Prediction 2 for b: "about 4 times the pilot's prediction" on the mixture has no
  committed measurement; it is withdrawn.
- Part 3, the Gaussian with unequal scales, had predictions but no falsification test.
  The rule of the Gaussian table (k not chosen at more than 2 seeds) was applied to it
  after the runs, in the outputs' reading and on the issue.

### Outcome against each prediction
1. Gaussian table (`i137_baseline.out`, `i137_remedies.out`):
   - base: failures at d = 2, n = 2 at 12 of 30 seeds (predicted 25 to 40%: held, at
     40%); at d = 10, n = 3 at 29 of 30 (predicted 60 to 70%: failed, more); at d = 10, n = 10
     at 11 of 30 (predicted about 30%: held); none elsewhere, d = 10, n = 5 included
     (held).
   - a: no failure anywhere (held); median meeting time 27.0 at d = 2, n = 2 (held); no
     extra evaluation (held); at d = 10, n = 10 the step is 0.21 (held).
   - b: no failure anywhere (held); the lowest median meeting time of the five kernels
     at every n_steps >= 3 (held); tuning gradients up by 2,560 n + 256, 65% at n = 10
     (held).
   - c: no failure anywhere (held); its median meeting times lie between b's and the
     package's where the package fails (held); where the package does not fail, they are
     above the package's at the default n_steps and n_steps 1 and 2, and below it at the
     other n_steps.
   - d: equal to c at d = 2 only for n_steps <= 2: at n_steps = 3 its width rule gave
     less than 0.5 at every seed (failed); narrower at n_steps 5 and 10, with no failure
     and meeting times like c's (held).
2. Targets without resonance (`i137_nongaussian.out`):
   - a: identical to the package on the mixture and on the banana with n_steps = 5; on
     the banana with n_steps = 2 the guard raised the step at 2 of 5 seeds (failed).
   - b: on the banana it fails the test with n_steps = 2 (median meeting time; failed)
     and is faster with n_steps = 5; on the mixture, where no outcome was predicted, it
     fails (inefficiency 336,336 against the package's 44,339).
   - c, d: within the package's spread on the banana with n_steps = 5 only; they fail on
     the banana's median meeting time with n_steps = 2 (83.1 against 70.4) and on the
     mixture's inefficiency (57,218 against 44,339) (failed).
3. Unequal scales (`i137_unequal.out`):
   - base: no failure with n_steps = 10, and failures at 3 of 10 seeds with the default
     n_steps (both failed).
   - a: no step of the guard's grid met the rule at any n_steps, not only at n_steps = 10;
     it passes with n_steps = 3 and 10 and fails at 3 of 10 seeds of the default (failed).
   - b, c, d: c and d have no failure (held); b fails at 2 of 10 seeds of the default
     (failed).
4. The recommendation expected c (or d). The recommendation made on issue #137 was the
   guard (a) for the case the issue states, and the jitter (c) for the test of issue #91
   (failed). After the review it was revised: no remedy enters the package now, and a
   guard that acts only near a resonance and the jitter both go to the test of issue #91.
