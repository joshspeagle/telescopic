# The adversarial reviews of the x91 registration

`REVIEW.md` is the review of the first version of the registration
(`../PREREGISTRATION.v1.md`); `REVIEW2.md` is the review of the revised one. The probes
`p1_guard.py` to `p4_mala.py` and their outputs belong to the first review.

`p1_guard.out` and `p2_claims.out` record the review's comparison with the guard of the
first version, whose fallback took the grid step furthest from every resonance: the
review's own implementation of the guard from the text, `mine` in `p1_guard.py`, agreed
with it on all 990 tunings. The revision changed the fallback to keep the tuned step
(section 14, I2), so the scripts run against the current `../guard.py` print 30
differences, all on the fallback cases of the Gaussian with unequal scales and the
default `n_steps`. The outputs are kept as the record of what the review saw.

`p4_mala.py` times the funnel's gradient; its output's rates and seconds vary between
runs.
