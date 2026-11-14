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
