"""Review of x123: what the detection formula predicts when log_prob is float32.

Question: the review measured that (N) warns on 196 of 200, 37 of 200 and 0 of 100
checks when log_prob and grad are float32 and one coordinate of N(0, I_d) has the wrong
sign, at d = 200, 1000 and 10000 from a start of sd 2 (miss_detect.out). Does the
registered formula, with the constants of float32, predict these rates?

Design: the formula of PREREGISTRATION.md, section 5, for a Gaussian target, where the
third derivative is 0: a line warns iff |e.u| > C_N L, with C_N = S U eps / h and
h = eps^(1/3) for the epsilon of float32, e.u = 2 x_0 u_0 and L = |x|^2 / 2. A Monte
Carlo of the inequality over x ~ N(0, 4 I_d) and u ~ N(0, 8 I_d), 100,000 lines (4,000
at d = 10000), from generator 500900; a check warns with probability 1 - (1 - p)^8.

Run: python notes/experiments/x123/review/f32pred.py > f32pred.out

Result: see f32pred.out and REVIEW.md, objection 5. The formula predicts check rates of
0.979, 0.191 and 0 at d = 200, 1000 and 10000, against 0.98, 0.185 and 0 measured.
"""

import numpy as np

rng = np.random.default_rng(500900)
e32 = np.finfo(np.float32).eps
h = e32 ** (1 / 3)
C = 10 * 64 * e32 / h
print("C_N32", C, "h32", h)
for d in (5, 50, 200, 1000, 10000):
    M = 20000 if d > 1000 else 100000
    hit = 0
    n = 0
    for _ in range(max(1, M // 20000)):
        m = 20000 if d <= 1000 else 4000
        x0 = 2 * rng.standard_normal(m)
        u0 = 2 * np.sqrt(2) * rng.standard_normal(m)
        L = 0.5 * (x0**2 + 4 * rng.chisquare(d - 1, m))
        hit += np.sum(2 * np.abs(x0 * u0) > C * L)
        n += m
    p = hit / n
    print(f"d={d}: formula line rate {p:.4f}, check rate {1 - (1 - p) ** 8:.4f}")
