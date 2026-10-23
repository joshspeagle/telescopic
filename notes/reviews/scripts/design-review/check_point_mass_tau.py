"""Point-mass pi0 (X0 = Y0 = x0), lag 1, coupled RWMH (reflection-maximal, common uniform).

AJ24's convention (spec R-EST-2) gives tau = 1 whenever X_1 = Y_0, i.e. when the first
lag step rejects. An engine that only trusts the kernel flag 'met = identical & both accept'
records the first joint acceptance instead. Target N(0, I_d), step 2.38/sqrt(d), x0 = 3 * 1.
"""
import numpy as np

rng = np.random.default_rng(81)
R = 200_000
for d in (1, 5):
    s = 2.38 / np.sqrt(d)
    x0 = np.full(d, 3.0)
    lp0 = -0.5 * x0 @ x0
    # lag step for X
    prop = x0 + s * rng.standard_normal((R, d))
    acc1 = np.log(rng.random(R)) < -0.5 * np.sum(prop**2, 1) - lp0
    equal_at_L = ~acc1  # X_1 = x0 = Y_0
    # flag-only engine: from equal states the pair stays equal; flag needs a joint acceptance.
    # from x0 (equal states) the first joint acceptance time is geometric with the acceptance rate at x0
    a0 = np.mean(np.minimum(1, np.exp(-0.5 * np.sum((x0 + s * rng.standard_normal((R, d))) ** 2, 1) - lp0)))
    extra = rng.geometric(a0, size=R)  # coupled steps until the first joint acceptance
    print(f"d={d}: P(X_1 = Y_0) = {equal_at_L.mean():.3f}; for those pairs AJ24 tau = 1, flag-only "
          f"engine records tau = 1 + G with E[G] = {extra.mean():.2f} (acceptance at x0 {a0:.3f}); "
          f"each recorded step is counted as a coupled step (2 kernel applications)")
