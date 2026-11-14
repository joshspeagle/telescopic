# x123, second experiment: two gradient checks, registered before any run (#123)

This file fixes, before any run on the registered seeds, two designs of the gradient
check, the families they are tested on, the seeds, the predictions of each design's
formula, the measure of harm and the criteria. It is not changed after the commit that
adds it. A reader needs no other file to judge the result; the scripts beside it are
the registered code, and their hashes are in section 10.

## 1. What was seen before this registration

- **The first experiment** (seeds 0 to 41,004) and **its adversarial review** (seeds
  60,000 to 189,999), summarized on #123: a directional check with a 1% floor passed its
  criterion, and the review showed that the pass depended on which wrong families were
  listed (starts tight in the wrong coordinate, starts not matched to the target's
  scales, starts far wider than the target, dimensions above 1,000). The first
  experiment's scripts are not in the repository; its result and its review are on issue
  #123.
- **The design document of this round**, `design/DESIGN.md` beside this file (written on
  2026-11-14, summarized in a comment of that date on issue #123, committed with this
  registration together with the scripts and outputs it cites), derived the two designs
  below and explored them on seeds 200,000 to 219,999. It is cited here as "the design
  document" with its section number; every value and formula that this registration
  needs is stated here.
  Its prototype (`prototype.py` in this folder) changed twice on those seeds:
  absolute pooling of the rounding estimate was added after a ratio of 0.84 on a correct
  gradient (rotated, condition 1e12, d = 3), and the probe of a narrow gradient was added
  after 77 warnings in 100 on a float32 gradient of a rotated Gaussian of condition 1e8.
  None of its rates is a test.
- **Phase 1 of this experiment** used seeds 220,000 to 229,999 (the rewrite against the
  prototype, the smoke runs). Two registered generators were read in phase 1 by
  mistake, before this file existed: generator 381000, for a first run of the proxy's
  marginal check (it passed; the registered marginal check uses 381500), and generator
  380000, the proxy's step bisection, whose tuned steps at d = 5 and 50 (1.290, 0.717,
  0.862) were printed. A tuned step is a setting of the sampler, not an outcome, and it
  does not depend on the seed of any meeting time. The prediction model ran on its
  registered generators 395000 to 395101; that is how its numbers in section 5 were made.

## 2. The designs

Both designs are the function `check` of `gradcheck.py`; they differ only in the floor.

**The stream.** From the generator `replicate_generator(root_seed(rng), 3, 0)` (the
package's replicate streams; namespace 3, which no run of the sampler reads), draw
`3 P` points of the initial distribution, in the order `x_1, a_1, b_1, x_2, ...`, then
`2 P d` random signs with `choice([-1, 1])`: the first `P d` for the fallback directions,
the last `P d` for the probe of a narrow gradient. `P = 8`.

**The lines.** The line at `x_p` has direction `u_p = a_p - b_p`. Where `u_pi = 0` it is
a random sign times the range of the `3 P` draws in coordinate `i`, else `|x_pi|`, else
1; where `|u_pi| < sqrt(eps_64) |x_pi|` it is raised to that size. The step vector is
`v_p = h u_p`, `h = eps_f^(1/3)`, with `eps_f` the epsilon of the dtype that `log_prob`
returns (`6.06e-6` for float64, `4.9e-3` for float32). The points of the line are
`x_p + t v_p`, `t` in `{1, -1, 2, -2}`. Where `log_prob` returns a narrower dtype than
float64, `x_p` and the points of the line are rounded to it before use, and `x_p` is
evaluated again after rounding. The dtypes are those of the first point; if a later
point returns a narrower one, the sweep restarts with it (the decisions are those of
the prototype, which took the narrowest over all points; the counts can differ).

**The comparison**, per unit of `t`, with the points as rounded: `m_t` the rounded
displacement, `w_1 = (m_1 - m_-1).v / |v|^2` (about 2), `w_2` likewise (about 4),
`f_t` the values of `log_prob`, `g = grad(x_p)`:

    D  = (f_1 - f_-1) / w_1,      G  = g . (m_1 - m_-1) / w_1,
    D' = (f_2 - f_-2) / w_2,      G' = g . (m_2 - m_-2) / w_2,
    gap = |G - D|,                T = |(D - G) - (D' - G')|,
    F4 = f_-2 - 4 f_-1 + 6 f_0 - 4 f_1 + f_2,    L = max |f| on the line,
    R_f = (2 / w_1) max(r L, s, |F4| / sqrt(70), U eps_f L),
          r^2 = mean over the usable lines of (F4 / L)^2 / 70,   s^2 = mean of F4^2 / 70,
    R_g = sum_i max(U eps_g |g_i|, q_i) |m_1 - m_-1|_i / w_1,

with `eps_g` the epsilon of the dtype of `grad`, and `q_i = |grad_i(x_p (1 + 4 eps_g s))
- grad_i(x_p)|` (`s` the probe's signs) where `grad` returns a narrower dtype than
float64, 0 otherwise. A line warns when

- **(N), noise-limited:** `gap > S (T + R_f + R_g)`;
- **(H), harm floor:** `gap > max(S (T + R_f + R_g), h PHI)`;

and a check warns when any usable line warns. The inequalities are strict, so `0 > 0`
is silence. A line is unusable when a value on it is `-inf`, `w_1 <= 0` or `w_2 <= w_1`;
a point is skipped when its density is zero or its gradient is not finite (the package's
`Target` takes such a gradient as 0 and counts it).

**Constants and their sources.**

| Constant | Value | Source |
|---|---|---|
| `P` | 8 points, one line each | the design document, section 3: the review measured that more points with one direction each raise the power, more directions per point lower it |
| `S` | 10 | the first version of the check; upheld by its review and by the first experiment's |
| `U` | 64 units in the last place | the first version; upheld by both reviews; inherited, not derived |
| `h` | `eps_f^(1/3)` | the step of a central difference whose truncation and rounding errors balance |
| `delta*` | 1/8 | the design document, section 1: a drift error of an eighth of a proposal standard deviation changes the proposal with probability 0.05 and the tuned step by under 1% |
| `PHI` | `2 sqrt(2) delta* = 0.354` nats | the design document, section 9: `e.u ~ N(0, 2 |C^T e|^2)` when the start's covariance is the sampler's, with the step `eps = 1` |
| namespace | 3 | the first version; runs read 0, 1 and 2 |

**What the check does besides.** It builds a `Target` of its own (the sampler's
counters and once-only precision warning are untouched); it holds one line at a time,
drawing the `3 P` points twice (once for the range of the fallback, once for the lines;
no evaluation in the first pass); it costs `5 P = 40` evaluations of `log_prob` and
`P = 8` of `grad` at any dimension, `P` more of each where `log_prob` is narrower than
float64 and `P` more where `grad` is; an error raised by `log_prob` or `grad` inside it
is raised again as `GradientCheckError`, which says that it came from the gradient
check and that `check_grad=False` skips it, chained to the original. Each line keeps its
gap, its tolerance `S (T + R_f + R_g)`, `h`, and the norm `|g o (m_1 - m_-1)| / w_1`, so
that one call gives the decisions of (N), of (H) and of the relative floors of section 8.

## 3. The families

`families.py` defines them; this section lists every one with its index and seeds.
Group A: correct gradients, a warning is a false alarm; group B: gradients that are not
the derivative of `log_prob` as computed and that are expected to do little or no harm
(reported, no bar); group C: wrong gradients; F: wrong gradients whose predicted rate is
checked on 1,000 seeds (the formula checks). "Start sd s" is a start of
`N(0, s^2 I)`; a vector of spreads is a diagonal start. Where the design document left a
detail open, this registration chose it; section 9 lists those choices.

**Group A, correct gradients, 300 seeds each.**

| A | Family | Seeds |
|---|---|---|
| 0 | N(0, 1e-08^2 I) d=5, matched | 300000 to 300299 |
| 1 | N(0, 0.0001^2 I) d=5, matched | 300300 to 300599 |
| 2 | N(0, 1^2 I) d=5, matched | 300600 to 300899 |
| 3 | N(0, 10000^2 I) d=5, matched | 300900 to 301199 |
| 4 | N(0, 1e+08^2 I) d=5, matched | 301200 to 301499 |
| 5 | N(0, 1e-08^2 I) d=5, wide 100x | 301500 to 301799 |
| 6 | N(0, 0.0001^2 I) d=5, wide 100x | 301800 to 302099 |
| 7 | N(0, 1^2 I) d=5, wide 100x | 302100 to 302399 |
| 8 | N(0, 10000^2 I) d=5, wide 100x | 302400 to 302699 |
| 9 | N(0, 1e+08^2 I) d=5, wide 100x | 302700 to 302999 |
| 10 | condition 10000 d=10, matched | 303000 to 303299 |
| 11 | condition 1e+08 d=10, matched | 303300 to 303599 |
| 12 | condition 1e+12 d=10, matched | 303600 to 303899 |
| 13 | condition 10000 d=10, wide sd 100 | 303900 to 304199 |
| 14 | condition 1e+08 d=10, wide sd 100 | 304200 to 304499 |
| 15 | condition 1e+12 d=10, wide sd 100 | 304500 to 304799 |
| 16 | N(1e6, I) d=5 | 304800 to 305099 |
| 17 | N(0,I) + 1e9 d=5 | 305100 to 305399 |
| 18 | N(0,I) - 1e12 d=5 | 305400 to 305699 |
| 19 | banana | 305700 to 305999 |
| 20 | funnel | 306000 to 306299 |
| 21 | mixture | 306300 to 306599 |
| 22 | pumps | 306600 to 306899 |
| 23 | Student t3 d=1 | 306900 to 307199 |
| 24 | Student t3 d=5 | 307200 to 307499 |
| 25 | Laplace d=5 | 307500 to 307799 |
| 26 | Laplace d=100 | 307800 to 308099 |
| 27 | half-normal boundary d=3 | 308100 to 308399 |
| 28 | logistic n=1e4 d=5 | 308400 to 308699 |
| 29 | logistic n=2000 d=50 | 308700 to 308999 |
| 30 | float32 N(0,I) d=5 | 309000 to 309299 |
| 31 | float32 N(0,I) - 1e4 d=5 | 309300 to 309599 |
| 32 | float32 Student t3 d=5 | 309600 to 309899 |
| 33 | logistic computed in float32 d=5 | 309900 to 310199 |
| 34 | point start N(0,I) d=5, random point | 310200 to 310499 |
| 35 | point start logistic d=5, at 0 | 310500 to 310799 |
| 36 | point start Student t3 d=5, random point | 310800 to 311099 |
| 37 | scales 1e-3..1e3 d=7, matched | 311100 to 311399 |
| 38 | t3 scales 1e-3..1e3 d=7, isotropic init 2e3 | 311400 to 311699 |
| 39 | N(0,I) d=5, init sd 1e-12 at 1 | 311700 to 311999 |
| 40 | N(0,I) d=1 | 312000 to 312299 |
| 41 | N(0,I) d=50 | 312300 to 312599 |
| 42 | N(0,I) d=1000 | 312600 to 312899 |
| 43 | Student t3 d=1000 | 312900 to 313199 |
| 44 | flat log_prob 0.0, grad 0, d=3 | 313200 to 313499 |
| 45 | flat log_prob int 0, grad 0, d=3 | 313500 to 313799 |
| 46 | plateau (flat top |x|<10, Gaussian shoulders) d=3, init N(0,I) | 313800 to 314099 |
| 47 | hinge -relu(x) - x^2/2 d=3, point start at 0 | 314100 to 314399 |
| 48 | hinge d=3, init N(0,I) | 314400 to 314699 |
| 49 | asymmetric Laplace d=3, point start at 0 | 314700 to 314999 |
| 50 | Laplace d=3, point start at 0 | 315000 to 315299 |
| 51 | naive sum of 1e5 Gaussian terms, d=2 | 315300 to 315599 |
| 52 | Poisson regression n=2e4, naive sum, d=4 | 315600 to 315899 |
| 53 | Student t3 d=5, init sd 1e6 | 315900 to 316199 |
| 54 | Student t3 d=1 (Cauchy-like), init sd 1e8 | 316200 to 316499 |
| 55 | Student t3 d=1000, init sd 1e3 | 316500 to 316799 |
| 56 | rotated cond 1e12 d=50, matched | 316800 to 317099 |
| 57 | rotated cond 1e12 d=50, isotropic sd 1 | 317100 to 317399 |
| 58 | rotated cond 1e12 d=100, matched | 317400 to 317699 |
| 59 | rotated cond 1e8 d=2, matched | 317700 to 317999 |
| 60 | rotated cond 1e12 d=3, matched | 318000 to 318299 |
| 61 | N(0,I) d=2 | 318300 to 318599 |
| 62 | N(0,I) d=3 | 318600 to 318899 |
| 63 | Student t3 d=2 | 318900 to 319199 |
| 64 | N(0,I) d=5, init var 0 in coords 0,1 (at 0) | 319200 to 319499 |
| 65 | N(0,I) d=5, init sd 1e-150 | 319500 to 319799 |
| 66 | N(0,I) d=5, Empirical of 2 points | 319800 to 320099 |
| 67 | N(0,I) d=5, point start at 0 | 320100 to 320399 |
| 68 | Student t3 d=5, point start at 0 | 320400 to 320699 |
| 69 | funnel, point start at 0 | 320700 to 320999 |
| 70 | banana, point start at 0 | 321000 to 321299 |
| 71 | logistic n=2000 d=50, point start at 0 | 321300 to 321599 |
| 72 | float32 N(0,I) + 1e7 d=5 | 321600 to 321899 |
| 73 | float32 N(0,I) - 1e9 d=5 | 321900 to 322199 |
| 74 | float32 N(1e3, I) d=5 (all in float64, cast out) | 322200 to 322499 |
| 75 | float64 log_prob, float32 grad: N(0,I) d=5 | 322500 to 322799 |
| 76 | float64 log_prob, float32 grad: N(1e3,I) d=5 | 322800 to 323099 |
| 77 | float64 log_prob, float32 grad: N(1e5,I) d=5 | 323100 to 323399 |
| 78 | float64 log_prob, float32 grad: N(1e6,I) d=5 | 323400 to 323699 |
| 79 | float64 log_prob, float32 grad: logistic n=1e4 d=5 | 323700 to 323999 |
| 80 | float64 log_prob, float32 grad: rotated cond 1e4 d=10 | 324000 to 324299 |
| 81 | float64 log_prob, float32 grad: rotated cond 1e8 d=10 | 324300 to 324599 |
| 82 | float64 log_prob, float32 grad: rotated cond 1e12 d=10 | 324600 to 324899 |
| 83 | not vectorized: N(0,I) d=5 | 324900 to 325199 |
| 84 | not vectorized: logistic n=1e4 d=5 | 325200 to 325499 |
| 85 | not vectorized: banana | 325500 to 325799 |
| 86 | float32 throughout, N(0, I) d=5 | 325800 to 326099 |
| 87 | float32 throughout, N(10, I) d=5 | 326100 to 326399 |
| 88 | float32 throughout, N(100, I) d=5 | 326400 to 326699 |
| 89 | float32 throughout, N(300, I) d=5 | 326700 to 326999 |
| 90 | float32 throughout, N(1000, I) d=5 | 327000 to 327299 |
| 91 | float32 throughout, N(10000, I) d=5 | 327300 to 327599 |
| 92 | N(0,I) d=2000, start sd 2 | 327600 to 327899 |
| 93 | N(0,I) d=5000, start sd 2 | 327900 to 328199 |
| 94 | N(0,I) d=10000, start sd 2 | 328200 to 328499 |
| 95 | Student t3 d=2000, start sd 2 | 328500 to 328799 |
| 96 | Student t3 d=5000, start sd 2 | 328800 to 329099 |
| 97 | Student t3 d=10000, start sd 2 | 329100 to 329399 |
| 98 | Laplace d=2000, start sd 2 | 329400 to 329699 |
| 99 | Laplace d=5000, start sd 2 | 329700 to 329999 |
| 100 | Laplace d=10000, start sd 2 | 330000 to 330299 |
| 101 | rotated cond 1e4 d=2000, matched | 330300 to 330599 |
| 102 | scales (1, 0.01 x4), start N(0, I) | 330600 to 330899 |
| 103 | N(0,I) d=5, start sd (0.01, 2, 2, 2, 2) | 330900 to 331199 |
| 104 | N(0,I) - 1e4 d=5, start sd 1e-3 | 331200 to 331499 |
| 105 | N(0,I) d=5, start sd 1000 | 331500 to 331799 |
| 106 | constant log_prob -3.7, grad 0, d=3 | 331800 to 332099 |
| 107 | ReLU network, 50 units, d=10 | 332100 to 332399 |
| 108 | piecewise-linear table of -x^2/2, spacing 0.01, d=5 | 332400 to 332699 |

**Group B, 300 seeds each.**

| B | Family | Seeds |
|---|---|---|
| 0 | Jacobi solver rtol 1e-3 d=5 (exact gradient) | 335000 to 335299 |
| 1 | Jacobi solver rtol 1e-5 d=5 (exact gradient) | 335300 to 335599 |
| 2 | Jacobi solver rtol 1e-8 d=5 (exact gradient) | 335600 to 335899 |
| 3 | N(0,I) d=5: grad x 1.005 | 335900 to 336199 |
| 4 | N(0,I) d=1000: grad x 1.0001 | 336200 to 336499 |
| 5 | N(0,I) d=5: grad + 0.02 | 336500 to 336799 |
| 6 | logistic n=1e4 d=5: without the prior | 336800 to 337099 |
| 7 | logistic n=1e4 d=5, start sd 10: without the prior | 337100 to 337399 |
| 8 | N(0,I) d=5, start 100x wide: grad x 1.005 | 337400 to 337699 |

**Group C, wrong gradients, 400 seeds each.**

| C | Family | Seeds |
|---|---|---|
| 0 | N(0,I) d=5: wrong sign | 340000 to 340399 |
| 1 | N(0,I) d=5: twice | 340400 to 340799 |
| 2 | N(0,I) d=50: twice | 340800 to 341199 |
| 3 | N(0,I) d=200: twice | 341200 to 341599 |
| 4 | Student t3 d=5, start sd 2: gradient of N(0, I) | 341600 to 341999 |
| 5 | equicorrelated rho 0.9 d=10: gradient of rho 0.8 | 342000 to 342399 |
| 6 | N(0,I) d=5, start sd 1: sign of coordinate 0 | 342400 to 342799 |
| 7 | N(0,I) d=50, start sd 1: sign of coordinate 0 | 342800 to 343199 |
| 8 | N(0,I) d=200, start sd 1: sign of coordinate 0 | 343200 to 343599 |
| 9 | N(0,I) d=1000, start sd 1: sign of coordinate 0 | 343600 to 343999 |
| 10 | N(0,I) d=2000, start sd 1: sign of coordinate 0 | 344000 to 344399 |
| 11 | N(0,I) d=5000, start sd 1: sign of coordinate 0 | 344400 to 344799 |
| 12 | N(0,I) d=10000, start sd 1: sign of coordinate 0 | 344800 to 345199 |
| 13 | N(0,I) d=5, start sd 2: sign of coordinate 0 | 345200 to 345599 |
| 14 | N(0,I) d=50, start sd 2: sign of coordinate 0 | 345600 to 345999 |
| 15 | N(0,I) d=200, start sd 2: sign of coordinate 0 | 346000 to 346399 |
| 16 | N(0,I) d=1000, start sd 2: sign of coordinate 0 | 346400 to 346799 |
| 17 | N(0,I) d=2000, start sd 2: sign of coordinate 0 | 346800 to 347199 |
| 18 | N(0,I) d=5000, start sd 2: sign of coordinate 0 | 347200 to 347599 |
| 19 | N(0,I) d=10000, start sd 2: sign of coordinate 0 | 347600 to 347999 |
| 20 | N(0,I) d=10000, start sd 2: signs of coordinates 0..9 | 348000 to 348399 |
| 21 | N(0,I) d=5, start sd 2: coordinate 0 set to 0 | 348400 to 348799 |
| 22 | scales (1, 0.1 x4), start N(0, I): sign of the wide coordinate | 348800 to 349199 |
| 23 | scales (1, 0.01 x4), start N(0, I): sign of the wide coordinate | 349200 to 349599 |
| 24 | scales (1, 0.1 x4), start N(0, 0.1^2 I): sign of the wide coordinate | 349600 to 349999 |
| 25 | scales (1, 0.01 x4), start N(0, 0.01^2 I): sign of the wide coordinate | 350000 to 350399 |
| 26 | N(0,I) d=5, start sd (0.1, 2, ...): sign of coordinate 0 | 350400 to 350799 |
| 27 | N(0,I) d=100, start sd (0.1, 2, ...): sign of coordinate 0 | 350800 to 351199 |
| 28 | N(0,I) d=1000, start sd (0.1, 2, ...): sign of coordinate 0 | 351200 to 351599 |
| 29 | N(0,I) d=5, start sd (0.01, 2, ...): sign of coordinate 0 | 351600 to 351999 |
| 30 | N(0,I) d=100, start sd (0.01, 2, ...): sign of coordinate 0 | 352000 to 352399 |
| 31 | N(0,I) d=1000, start sd (0.01, 2, ...): sign of coordinate 0 | 352400 to 352799 |
| 32 | N(0,I) d=5, start sd 100: grad + 2 | 352800 to 353199 |
| 33 | N(0,I) d=5, start sd 300: grad + 2 | 353200 to 353599 |
| 34 | N(0,I) d=5, start sd 1000: grad + 2 | 353600 to 353999 |
| 35 | N(0,I) d=5, start sd 10000: grad + 2 | 354000 to 354399 |
| 36 | N(0,I) d=5, start sd 1000: sign of coordinate 0 | 354400 to 354799 |
| 37 | rotated cond 100 d=10, matched: sign along the widest axis | 354800 to 355199 |
| 38 | rotated cond 10000 d=10, matched: sign along the widest axis | 355200 to 355599 |
| 39 | rotated cond 1e4 d=2000, matched: sign along the widest axis | 355600 to 355999 |
| 40 | float32 grad, N(0,I) d=5: sign of coordinate 0 | 356000 to 356399 |
| 41 | float32 grad, N(0,I) d=1000, start sd 2: sign of coordinate 0 | 356400 to 356799 |
| 42 | float32 grad, N(1e3,I) d=5: sign of coordinate 0 | 356800 to 357199 |
| 43 | plateau d=3, start N(0, I): grad = -x | 357200 to 357599 |
| 44 | N(0,I) d=2, point start (1, 1): grad + 2 (1, -1) | 357600 to 357999 |
| 45 | N(0,I) d=5, random point start: sign of coordinate 0 | 358000 to 358399 |

**Formula checks, 1,000 seeds each.**

| F | Family | Seeds |
|---|---|---|
| 0 | N(0,I) d=10000, start sd (0.01, 2, ...): sign of coordinate 0 | 360000 to 360999 |
| 1 | N(0,I) d=1000, start sd (0.003, 2, ...): sign of coordinate 0 | 361000 to 361999 |
| 2 | N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0 | 362000 to 362999 |
| 3 | scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate | 363000 to 363999 |
| 4 | N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0 | 364000 to 364999 |
| 5 | N(0,I) d=1000, start sd (0.5, 2, ...): sign of coordinate 0 | 365000 to 365999 |
| 6 | N(0,I) d=1000, start sd (0.25, 2, ...): sign of coordinate 0 | 366000 to 366999 |

## 4. Harm

**The measure** of a gradient against a baseline is the ratio of mean meeting times,
`Lambda = mean tau(given gradient) / mean tau(baseline)`, over 5 seeds of 100 pairs each
(500 pairs a side), coupled MALA with lag 1, a pair unmet at `max_iter = 10,000` counted
at 10,000 (so that `Lambda` is then a lower bound). Both gradients of a seed run with the
same seed. Its standard error is `sd(r_1..r_5) / sqrt(5)` from the 5 per-seed ratios of
means. The baseline is the derivative of `log_prob` as computed: the family's correct
gradient; for the Jacobi solvers of group B, the gradient of the solver's quadratic form
at its iteration count (`families.jacobi_gradient`); for the logistic regressions of
group B, the gradient with its prior.

**Classes.** Harmful iff `Lambda - 2 se >= 2` (`>= 3` with the proxy below); harmless iff
`Lambda + 2 se <= 1.5`; otherwise unclassed: reported, and outside every bar.
Undetermined when the baseline leaves more than 5% of its 500 pairs unmet.

**Samplers.**

- *Package*, at `d <= 200`: `CoupledSampler(log_prob, init, grad=..., vectorize=True,
  kernel="mala", rng=seed).meeting_times(100, max_iter=10_000)`, default tuning. Package
  baseline `b` (the list `harm.PACKAGE`, 35 baselines) runs on seeds
  `370000 + 10 b + (0..4)`, 370000 to 370344.
- *Proxy*, above `d = 200` (targets `N(0, I_d)` only): the package's coupled MALA step
  rewritten for `cov = I` in `O(d)` per step (proposal `x + (eps^2 / 2) g(x) + eps z`,
  the reflection-maximal coupling of the two proposals, one uniform for both
  acceptances, lag 1), with the step whose mean acceptance at stationarity is 0.574 for
  the gradient in use (bisection on `max(200, 4,000,000 / d)` draws of `pi` from
  generator 380000), which is what the package's dual averaging aims at. Proxy baseline
  `b` (`harm.PROXY`, 18 baselines) runs on seeds `375000 + 10 b + (0..4)`, 375000 to
  375174. Because the design document's exploratory runs found that it overstates the package's
  slowdown by about 1.3 at `d = 50` and 200, its bar for harmful is 3. Each chain's
  marginal transition is MALA's: the reflection-maximal coupling gives each proposal the
  law of the uncoupled proposal, and each chain accepts with its own ratio and a uniform
  that is uniform for it; chains that propose the same point from the same state accept
  together and stay equal. The proxy is not a kernel of the package, so
  `telescopic.testing.check_coupling` does not apply; `harm.py marginals` checks the
  same property: one coupled step from fixed states against the package's
  `MALA(eps, cov=I)` step, 20,000 pairs, at `d = 5` (the derivative and a wrong sign)
  and `d = 50`, generator 381500, with a KS test along 3 random directions and a test of
  the rate of staying put for each chain, and equal pairs staying equal.
  **Validation**, reported: the proxy also runs the 6 families of `N(0, I_50)` and
  `N(0, I_200)` that the package runs (sign of coordinate 0 from starts of sd 1 and 2,
  twice the gradient from sd 2), and the ratio of the two `Lambda` is reported.
- *By equivalence*: the rotated Gaussian of condition 1e4 at `d = 2000` with the sign
  of its widest axis wrong takes the harm of `N(0, I_2000)` from start sd 2 with the
  sign of coordinate 0 wrong. MALA with the target's covariance on the rotated target is
  MALA with `cov = I` on the whitened target, where the matched start (4 cov) is
  `N(0, 4 I)` and the wrong gradient flips one coordinate; this is exact for the proxy,
  which has `cov = I` in whitened coordinates.

**Beside each `Lambda`**, reported and classing nothing: the ratio of tuned steps (given
over baseline) and `delta_0`, the root mean square under `pi` of the drift error in
proposal standard deviations, `(eps / 2) |C^T e(x)|`, at the baseline's tuned step
`eps` and covariance `C C^T` (package: averaged over the 5 seeds, 2,000 draws of `pi`
each from the generator of the seed; the Laplace approximation of `pi` for the logistic
targets; proxy: `C = I`, 2,000 draws from generator 380001).

## 5. Predictions

For each family of groups B, C and F, the probability that a line warns and that a
check warns, by the detection formula of the design document (section 5),
computed by `predictions.py` before any registered run (`predictions.out`). In float64,
where rounding sets the tolerance, a line warns iff

    (N)  |e.u + (h^2 / 6) d3| > C_N L + S sum_i max(U eps_g |g_i|, q_i) |u_i| + (S h^2 / 2) |d3|,
    (H)  the same, and |e.u + (h^2 / 6) d3| > PHI,

with `e` the gradient's error against the derivative of `log_prob` as computed, `u =
a - b`, `L = |log_prob(x)|`, `C_N = S U eps_64 / h = 2.35e-8`, `d3` the third derivative
along `u`, and `q_i` the probe's movement for a float32 gradient. The lines of a check are
independent given the initial distribution, so a check warns with probability
`1 - (1 - p)^8` for a line rate `p`; from a point start the lines share their point and
the check is simulated whole. "diagonal" is the reduced form of the design document for diagonal
Gaussians (exact in distribution, 2 million draws, averaged in closed form over one
normal where one coordinate is wrong); "direct" evaluates the expression above on draws
of the family's own start with its own functions; "jacobi" uses the solver's quadratic
form; "point" simulates 20,000 whole checks. `1-x` means `1 - x`.

| Group | Family | Model | (N) line | (N) check | (H) line | (H) check |
|---|---|---|---|---|---|---|
| B | Jacobi solver rtol 1e-3 d=5 (exact gradient) | jacobi | 1-8.5e-05 | 1 | 0.0000 | 0.0000 |
| B | Jacobi solver rtol 1e-5 d=5 (exact gradient) | jacobi | 0.9929 | 1 | 0.0000 | 0.0000 |
| B | Jacobi solver rtol 1e-8 d=5 (exact gradient) | jacobi | 0.0024 | 0.0188 | 0.0000 | 0.0000 |
| B | N(0,I) d=5: grad x 1.005 | diagonal | 1-1.5e-06 | 1 | 0.0001 | 0.0005 |
| B | N(0,I) d=1000: grad x 1.0001 | diagonal | 0.9979 | 1 | 0.0000 | 0.0000 |
| B | N(0,I) d=5: grad + 0.02 | diagonal | 1-2.5e-06 | 1 | 0.0051 | 0.0402 |
| B | logistic n=1e4 d=5: without the prior | direct | 0.9948 | 1 | 0.0000 | 0.0000 |
| B | logistic n=1e4 d=5, start sd 10: without the prior | direct | 1-5.0e-06 | 1 | 0.9990 | 1 |
| B | N(0,I) d=5, start 100x wide: grad x 1.005 | diagonal | 1-2.0e-06 | 1 | 1-5.3e-04 | 1 |
| C | N(0,I) d=5: wrong sign | diagonal | 1 | 1 | 0.9867 | 1-1.0e-15 |
| C | N(0,I) d=5: twice | diagonal | 1 | 1 | 0.9735 | 1-2.5e-13 |
| C | N(0,I) d=50: twice | diagonal | 1-5.0e-07 | 1 | 0.9929 | 1 |
| C | N(0,I) d=200: twice | diagonal | 1 | 1 | 0.9965 | 1 |
| C | Student t3 d=5, start sd 2: gradient of N(0, I) | direct | 1 | 1 | 0.9108 | 1-4.0e-09 |
| C | equicorrelated rho 0.9 d=10: gradient of rho 0.8 | direct | 1 | 1 | 0.8567 | 1-1.8e-07 |
| C | N(0,I) d=5, start sd 1: sign of coordinate 0 | diagonal | 1-1.6e-07 | 1 | 0.7453 | 1-1.8e-05 |
| C | N(0,I) d=50, start sd 1: sign of coordinate 0 | diagonal | 1-2.0e-06 | 1 | 0.7454 | 1-1.8e-05 |
| C | N(0,I) d=200, start sd 1: sign of coordinate 0 | diagonal | 1-8.5e-06 | 1 | 0.7455 | 1-1.8e-05 |
| C | N(0,I) d=1000, start sd 1: sign of coordinate 0 | diagonal | 1-3.3e-05 | 1 | 0.7451 | 1-1.8e-05 |
| C | N(0,I) d=2000, start sd 1: sign of coordinate 0 | diagonal | 1-6.8e-05 | 1 | 0.7453 | 1-1.8e-05 |
| C | N(0,I) d=5000, start sd 1: sign of coordinate 0 | diagonal | 1-1.6e-04 | 1 | 0.7452 | 1-1.8e-05 |
| C | N(0,I) d=10000, start sd 1: sign of coordinate 0 | diagonal | 1-3.0e-04 | 1 | 0.7454 | 1-1.8e-05 |
| C | N(0,I) d=5, start sd 2: sign of coordinate 0 | diagonal | 1-1.8e-07 | 1 | 0.9088 | 1-4.8e-09 |
| C | N(0,I) d=50, start sd 2: sign of coordinate 0 | diagonal | 1-2.0e-06 | 1 | 0.9087 | 1-4.8e-09 |
| C | N(0,I) d=200, start sd 2: sign of coordinate 0 | diagonal | 1-8.1e-06 | 1 | 0.9088 | 1-4.8e-09 |
| C | N(0,I) d=1000, start sd 2: sign of coordinate 0 | diagonal | 1-3.5e-05 | 1 | 0.9089 | 1-4.8e-09 |
| C | N(0,I) d=2000, start sd 2: sign of coordinate 0 | diagonal | 1-6.7e-05 | 1 | 0.9088 | 1-4.8e-09 |
| C | N(0,I) d=5000, start sd 2: sign of coordinate 0 | diagonal | 1-1.5e-04 | 1 | 0.9089 | 1-4.8e-09 |
| C | N(0,I) d=10000, start sd 2: sign of coordinate 0 | diagonal | 1-3.0e-04 | 1 | 0.9088 | 1-4.8e-09 |
| C | N(0,I) d=10000, start sd 2: signs of coordinates 0..9 | diagonal | 1-9.5e-06 | 1 | 0.9914 | 1 |
| C | N(0,I) d=5, start sd 2: coordinate 0 set to 0 | diagonal | 1-4.5e-07 | 1 | 0.8453 | 1-3.3e-07 |
| C | scales (1, 0.1 x4), start N(0, I): sign of the wide coordinate | diagonal | 1-1.5e-05 | 1 | 0.7455 | 1-1.8e-05 |
| C | scales (1, 0.01 x4), start N(0, I): sign of the wide coordinate | diagonal | 0.9990 | 1 | 0.7454 | 1-1.8e-05 |
| C | scales (1, 0.1 x4), start N(0, 0.1^2 I): sign of the wide coordinate | diagonal | 1-1.6e-05 | 1 | 0.0000 | 0.0000 |
| C | scales (1, 0.01 x4), start N(0, 0.01^2 I): sign of the wide coordinate | diagonal | 1-1.0e-03 | 1 | 0.0000 | 0.0000 |
| C | N(0,I) d=5, start sd (0.1, 2, ...): sign of coordinate 0 | diagonal | 1-5.1e-05 | 1 | 0.0000 | 0.0000 |
| C | N(0,I) d=100, start sd (0.1, 2, ...): sign of coordinate 0 | diagonal | 0.9990 | 1 | 0.0000 | 0.0000 |
| C | N(0,I) d=1000, start sd (0.1, 2, ...): sign of coordinate 0 | diagonal | 0.9921 | 1 | 0.0000 | 0.0000 |
| C | N(0,I) d=5, start sd (0.01, 2, ...): sign of coordinate 0 | diagonal | 0.9965 | 1 | 0.0000 | 0.0000 |
| C | N(0,I) d=100, start sd (0.01, 2, ...): sign of coordinate 0 | diagonal | 0.9453 | 1-8.0e-11 | 0.0000 | 0.0000 |
| C | N(0,I) d=1000, start sd (0.01, 2, ...): sign of coordinate 0 | diagonal | 0.6916 | 1-8.2e-05 | 0.0000 | 0.0000 |
| C | N(0,I) d=5, start sd 100: grad + 2 | diagonal | 1-5.0e-07 | 1 | 1-4.4e-04 | 1 |
| C | N(0,I) d=5, start sd 300: grad + 2 | diagonal | 1-4.5e-06 | 1 | 1-1.4e-04 | 1 |
| C | N(0,I) d=5, start sd 1000: grad + 2 | diagonal | 1-6.0e-06 | 1 | 1-5.2e-05 | 1 |
| C | N(0,I) d=5, start sd 10000: grad + 2 | diagonal | 1-7.7e-05 | 1 | 1-7.3e-05 | 1 |
| C | N(0,I) d=5, start sd 1000: sign of coordinate 0 | diagonal | 1-1.8e-07 | 1 | 1-1.3e-06 | 1 |
| C | rotated cond 100 d=10, matched: sign along the widest axis | direct | 1 | 1 | 0.9079 | 1-5.2e-09 |
| C | rotated cond 10000 d=10, matched: sign along the widest axis | direct | 1 | 1 | 0.9085 | 1-4.9e-09 |
| C | rotated cond 1e4 d=2000, matched: sign along the widest axis | direct | 1 | 1 | 0.9085 | 1-4.9e-09 |
| C | float32 grad, N(0,I) d=5: sign of coordinate 0 | direct | 1-6.5e-04 | 1 | 0.9087 | 1-4.8e-09 |
| C | float32 grad, N(0,I) d=1000, start sd 2: sign of coordinate 0 | direct | 0.9253 | 1-9.7e-10 | 0.9079 | 1-5.2e-09 |
| C | float32 grad, N(1e3,I) d=5: sign of coordinate 0 | direct | 0.9827 | 1-7.9e-15 | 0.9086 | 1-4.9e-09 |
| C | plateau d=3, start N(0, I): grad = -x | direct | 1 | 1 | 0.8436 | 1-3.6e-07 |
| C | N(0,I) d=2, point start (1, 1): grad + 2 (1, -1) | point | - | 0.9961 | - | 0.9961 |
| C | N(0,I) d=5, random point start: sign of coordinate 0 | point | - | 1-1.0e-04 | - | 0.8337 |
| F | N(0,I) d=10000, start sd (0.01, 2, ...): sign of coordinate 0 | diagonal | 0.0927 | 0.5410 | 0.0000 | 0.0000 |
| F | N(0,I) d=1000, start sd (0.003, 2, ...): sign of coordinate 0 | diagonal | 0.0749 | 0.4635 | 0.0000 | 0.0000 |
| F | N(0,I) d=5, start sd (1e-4, 2, ...): sign of coordinate 0 | diagonal | 0.0288 | 0.2087 | 0.0000 | 0.0000 |
| F | scales (1, 1e-4 x4), start N(0, I): sign of the wide coordinate | diagonal | 0.1850 | 0.8053 | 0.1842 | 0.8039 |
| F | N(0,I) - 1e4 d=5, start sd 1e-3: sign of coordinate 0 | diagonal | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| F | N(0,I) d=1000, start sd (0.5, 2, ...): sign of coordinate 0 | diagonal | 1-4.5e-04 | 1 | 0.4102 | 0.9853 |
| F | N(0,I) d=1000, start sd (0.25, 2, ...): sign of coordinate 0 | diagonal | 0.9984 | 1 | 0.0618 | 0.3997 |

A rate shown as `1` is within rounding of 1 (a miss below `1e-16` per check). The standard errors of the model's Monte Carlo are in `predictions.out`; the largest on a predicted rate between 0.05 and 0.95 is `5e-4`, and on a point start `2.6e-3`.

**Reading the table against the criteria.** A rate of at least 0.999 puts a harmful
family under the detection bar (criterion 2); below 0.001 it is a predicted miss
(criterion 4); the formula checks (F) are tested against their predictions on 1,000
seeds (criterion 3); the rest are reported against their prediction with a binomial
z-score and enter no verdict:

- (N): every family of group C is predicted at or above 0.999 except
  "N(0,I) d=2, point start (1, 1): grad + 2 (1, -1)" (0.9961: with 8 sign vectors every
  line is orthogonal to the error with probability `2^-8`).
- (H): predicted misses (below 0.001) on "scales (1, 0.1 x4), start N(0, 0.1^2 I)",
  "scales (1, 0.01 x4), start N(0, 0.01^2 I)", and the six starts tight in the wrong
  coordinate (sd 0.1 and 0.01 at d = 5, 100, 1000); between 0.001 and 0.999 on the two
  point starts (0.9961, 0.834); at or above 0.999 on the rest of group C.
- On the formula checks, (N) predicts 0.541, 0.464, 0.209, 0.805, 0, 1, 1 and (H)
  predicts 0, 0, 0, 0.804, 0, 0.985, 0.400, in the order of F.
- Group B: (N) predicts a warning on every family except the solver at 1e-8 (0.019);
  (H) predicts silence (at most 0.04) except from the two wide starts (1 - 1e-24 and
  more).

## 6. Seeds

| What | Seeds or generators |
|---|---|
| Group A, family `f` (0 to 108) | `300000 + 300 f + (0..299)`: 300000 to 332699 |
| Group B, family `j` (0 to 8) | `335000 + 300 j + (0..299)`: 335000 to 337699 |
| Group C, family `f` (0 to 45) | `340000 + 400 f + (0..399)`: 340000 to 358399 |
| Formula checks, family `k` (0 to 6) | `360000 + 1000 k + (0..999)`: 360000 to 366999 |
| Harm, package baseline `b` (0 to 34) | `370000 + 10 b + (0..4)`; `delta_0` from generators of the same numbers |
| Harm, proxy baseline `b` (0 to 17) | `375000 + 10 b + (0..4)` |
| Proxy step bisection; proxy `delta_0`; proxy marginals | generators 380000; 380001; 381500 |
| Cost (criterion 5) | 390000 to 390004 |
| Byte-identical runs (criterion 6) | 391000, 391001 |
| Prediction model (already run) | generators 395000 to 395061, 395100, 395101 |
| Phase 1 (seen): rewrite against the prototype, smoke runs | 220000 to 229999 |

The check of seed `s` draws its points from the stream of root seed `s`, as a sampler
built with `rng=s` would.

## 7. Criteria, with the power of each test

The verdict is for each design separately. Every probability below is exact binomial
(`predictions.out`, last part).

1. **False alarms, group A.** At most 2 warnings in 300 on every family, and at most 16
   in the 32,700 pooled (0.05%). The pooled test holds with probability `2.7e-13` if the
   true rate is 1 in 500, 0.0010 at 1 in 1,000, 0.9995 at 1 in 5,000 and 1.0000 at 1 in
   10,000; the design document predicts below 1 in 10,000 (its section 6). The per-family test holds with
   probability 0.0057 for a family whose true rate is 3%, 0.060 at 2%, 0.9964 at 0.1%;
   if every family had the worst rate of the design document's noise model, 0.04%, all 109 would
   hold with probability 0.972.
2. **Detection, group C and F.** On every family classed harmful whose predicted rate
   for the design is at least 0.999: at least 390 warnings in 400 (on the 1,000 seeds of
   a formula check, at least 975). The test holds with probability 0.0094 if the true
   rate is 0.95 and 0.9973 if it is 0.99 (on 1,000 seeds: `5.1e-5` and 0.99998).
3. **Formula checks.** On each family of F, for each design: the measured rate within
   0.05 of the registered prediction. For a predicted rate between 0.2 and 0.8 the test
   fails with probability at most 0.0013 if the formula is right and holds with
   probability at most 0.0009 if the true rate is 0.1 away. A design whose formula fails
   has an unexplained behavior and does not pass, whatever its rate.
4. **Predicted misses.** On a family classed harmful where a design's predicted rate is
   below 0.001, the design fails criterion 2 by prediction, and the count confirms it:
   at most 10 warnings in 400 (this holds with probability 0.0094 if the true rate were
   5%). More than 10 refutes the prediction and counts as a failure of criterion 3.
5. **Cost** (`cost.py`): at `d = 1000` and `d = 10000`, on `N(0, I_d)` from a callable
   start, with a float64 target, a float32 gradient, and a target wholly in float32: at
   most 60 evaluations of `log_prob` and 25 of `grad`, and a tracemalloc peak during the
   check of at most 20 vectors of `d` float64 (it includes the user's functions, the
   draws and the check's fixed objects).
6. **No effect on results** (`integration.py`): 24 of 24 settings byte-identical with and
   without the check.

A design **passes** iff it meets all six.

**Predicted outcome**, so that the experiment can refute it: (N) passes all six and has
no predicted miss on a family classed harmful, unless "N(0,I) - 1e4 d=5, start sd 1e-3"
(a formula check predicted at 0 for both designs, the registered blind spot of section
8 of the design document) turns out harmful. (H) passes 1, 3, 5 and 6, and fails 2 through 4 on
the starts tight in the wrong coordinate if they are classed harmful, as the review
measured two of them (slowdowns of 3.2 and 4.2). If (N) fails 1, the next registration
would require two lines to disagree, which squares the per-line rates.

## 8. What is reported without being part of a verdict

- **Group B**, the evidence for the ruling the design document leaves to the lead (what size of
  inconsistency is worth a warning): the warnings of (N), of (H) and of the relative
  floors, with the harm of each family.
- **The relative floors**: for every family of every group, the warnings of (N) with a
  floor `rho |g o v|` added to its tolerance (`rho` = 1e-6, 1e-4, 1e-2), from the same
  checks.
- The largest ratio of gap to tolerance per family, unusable lines and skipped points.
- Families of C whose predicted rate is between 0.001 and 0.999 and that are not formula
  checks, against their predictions; families classed unclassed or undetermined; the
  proxy's validation against the package; `delta_0` and the step ratios.

**Failures of the experiment itself**, which void the verdicts they touch: a runner that
errors; a check that raises on a registered family; a registered output that is not
complete; the proxy's marginal check failing (smallest of its p-values below
`1e-3 / 8`), which voids the classes made with the proxy. A family classed unclassed or
undetermined is not a failure: it is reported and leaves criteria 2 and 4; if every
family that a criterion would test leaves, that criterion is inconclusive.

## 9. Choices this registration made where the design document left them open

- One call of `check` per seed, with the decisions of (N), of (H) and of the relative
  floors read from its lines. This is the same computation as separate calls (section
  10).
- Group A has 109 families, so the pooled bar of 0.05% is 16 of 32,700.
- "One coordinate set to 0": `N(0, I_5)` from start sd 2. "The signs of 10 coordinates
  at d = 10,000": from start sd 2. "Starts tight in the wrong coordinate": sd 2 in the
  others. The large-dimension correct families of group A start from sd 2. "A constant
  `log_prob` other than 0": -3.7 with `grad` 0 at d = 3. The ReLU network: d = 10, 50
  units, weights `N(0, 1/d)`, biases `N(0, 1)`, output weights `N(0, 1/4)`, plus
  `-|x|^2 / 2`, generator 0. The table: each coordinate the piecewise-linear interpolant
  of `-t^2 / 2` on knots of spacing 0.01 on [-8, 8], start sd 2, d = 5. The rotated
  Gaussian at d = 2,000: the widest axis.
- Group B: the factor 1.005 at d = 5 and 1.0001 at d = 1,000, both from start sd 2, as in
  the design document's exploration. "The same from a start 100 times wide" is read both ways:
  the logistic gradient without its prior from start sd 10, and the factor 1.005 from
  start sd 200 (the design document's exploration used the second).
- Criterion 3 for both designs on every formula check, each with its own prediction (the
  design document names one design per family). Criterion 2 on 1,000 seeds at 975.
- The harm's standard error from the 5 per-seed ratios of means; the proxy's bar for
  harmless is the package's, 1.5.
- The marginal check of the proxy in place of `check_coupling`, which needs a kernel of
  the package.

## 10. What phase 1 checked, and the frozen code

Phase 1 wrote the code and this file; phase 1b made the folder pass the repository's
gate (lint, format and spelling checks, with the fixes made at their causes: the runner
catches only `GradientCheckError` from a check, `zip` states `strict`, the commit query
states `check`), put the prototype in the folder, split `predictions.py` into parts
that can run separately, and ran the following again on the code as frozen.

- **The prototype.** `prototype.py` is the prototype of the design round, which the
  design document's header records with sha256
  `f174cf81f3077b4d51f9ed6377b0eefdff9130c2fa1658af8811871b64833f3d` and of which only
  this formatted copy is kept: the same code with a header docstring, formatted, and its
  one `dict(...)` call written as a literal, with its arithmetic unchanged.
  `equivalence.out` is the check of the registered code against it.
- **The rewrite against the prototype** (`equivalence.py`, `equivalence.out`): every
  family of A, B, C and F, 10 seeds each from 220,000 to 223,419, 1,710 checks. The
  usable lines (point, gap, tolerance) are bit for bit those of the prototype in 1,710 of
  1,710 checks, the decisions of (N) and of (H) agree in 1,710, and the unusable lines,
  skipped points and evaluation counts in 1,710. A `log_prob` whose dtype narrows to
  float32 only after the first point (seeds 229,000 to 229,049): lines and decisions
  identical in 50 of 50, counts in 25 of 50 (the 25 restarted sweeps). A `log_prob` that
  raises inside the check: `GradientCheckError` with the message of section 2, chained
  to the original error. Writing the code found no defect in the design document.
- **The prediction model** (`predictions.py`, `predictions.out`) on its registered
  generators. In phase 1b it ran again in six parts (models 0 to 2, 3 to 29, 30 to 47,
  48, 49 to 52, 53 to 61 with the comparison and the power); joined, their lines after
  the header are identical to `predictions.out` (an empty diff). The reduced form agrees
  with the direct evaluation on two families (line rates 0.7456 against 0.7446 +- 0.0010
  and 0.9990 against 0.9990 +- 0.0001).
- **Smoke runs** on seeds 220,000 and 220,001, of every runner, after the fixes:
  `run_checks.py` on A, B, C and F (342 checks, no error); `cost.py` (every count and
  peak within the bounds of criterion 5: 40 + 8, 48 + 16 and 56 + 24 evaluations; peaks
  of 15.3 to 16.4 vectors of `d` at `d = 1000` and 13.4 to 14.4 at `d = 10000`);
  `integration.py` (24 of 24 byte-identical); `harm.py marginals` (smallest p-value
  0.046 against a bound of `1.25e-4`; equal pairs stay equal); `harm.py proxy` (all 18
  baselines) and `harm.py package` (all 35, baseline 4 split with `--only` and
  `--seeds`), with 10 pairs and `max_iter = 1000`; and `harm.py summarize` on the split
  outputs (67 families and the one by equivalence, no family incomplete). These outputs
  are on seen seeds and are not part of the record.

**Compute expected for phase 2**, one core, one process at a time: about 2.5 to 3.5
hours in all. The checks (criteria 1, 2 and 3 on counts, 5 and 6) take about 30 minutes;
the harm runs the rest. Default tuning takes about 70 s a run at `d = 200`; one wrong
sign at `d = 10000` takes the proxy about 2 minutes a seed, and its ten wrong signs may
reach the cap in every pair (about 8 minutes a seed).

**The procedure of phase 2.** The commands below run in this order, from the root of
the repository at the registration commit, with `PYTHONPATH=src` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a time.
Each writes its output to a scratch folder and is copied into this folder when it is
complete, so that a tracked output is never half written; a run that is cut off is run
again whole. The checks come first, so that criteria 1, 2 (counts), 3, 5 and 6 are
complete before the harm runs; the harm runs are split by baseline, by gradient
(`--only`) and by seed (`--seeds i j`) so that each takes a few minutes and at most 25.
The last command joins the harm outputs into the classes.

```
 1. python notes/experiments/x123/run_checks.py A 0 40 > run_checks_A1.out
 2. python notes/experiments/x123/run_checks.py A 40 80 > run_checks_A2.out
 3. python notes/experiments/x123/run_checks.py A 80 109 > run_checks_A3.out
 4. python notes/experiments/x123/run_checks.py B > run_checks_B.out
 5. python notes/experiments/x123/run_checks.py C 0 23 > run_checks_C1.out
 6. python notes/experiments/x123/run_checks.py C 23 46 > run_checks_C2.out
 7. python notes/experiments/x123/run_checks.py F > run_checks_F.out
 8. python notes/experiments/x123/cost.py > cost.out
 9. python notes/experiments/x123/integration.py > integration.out
10. python notes/experiments/x123/harm.py marginals > harm_marginals.out
11. python notes/experiments/x123/harm.py proxy 0 2 > harm_proxy_00.out
12. python notes/experiments/x123/harm.py proxy 2 4 > harm_proxy_02.out
13. python notes/experiments/x123/harm.py proxy 4 5 > harm_proxy_04.out
14. python notes/experiments/x123/harm.py proxy 5 6 > harm_proxy_05.out
15. python notes/experiments/x123/harm.py proxy 6 7 --seeds 0 1 > harm_proxy_06_s0.out
16. python notes/experiments/x123/harm.py proxy 6 7 --seeds 1 2 > harm_proxy_06_s1.out
17. python notes/experiments/x123/harm.py proxy 6 7 --seeds 2 3 > harm_proxy_06_s2.out
18. python notes/experiments/x123/harm.py proxy 6 7 --seeds 3 4 > harm_proxy_06_s3.out
19. python notes/experiments/x123/harm.py proxy 6 7 --seeds 4 5 > harm_proxy_06_s4.out
20. python notes/experiments/x123/harm.py proxy 7 8 --only "sign of coordinate 0" --seeds 0 1 > harm_proxy_07a_s0.out
21. python notes/experiments/x123/harm.py proxy 7 8 --only "sign of coordinate 0" --seeds 1 2 > harm_proxy_07a_s1.out
22. python notes/experiments/x123/harm.py proxy 7 8 --only "sign of coordinate 0" --seeds 2 3 > harm_proxy_07a_s2.out
23. python notes/experiments/x123/harm.py proxy 7 8 --only "sign of coordinate 0" --seeds 3 4 > harm_proxy_07a_s3.out
24. python notes/experiments/x123/harm.py proxy 7 8 --only "sign of coordinate 0" --seeds 4 5 > harm_proxy_07a_s4.out
25. python notes/experiments/x123/harm.py proxy 7 8 --only "signs of coordinates 0..9" --seeds 0 1 > harm_proxy_07b_s0.out
26. python notes/experiments/x123/harm.py proxy 7 8 --only "signs of coordinates 0..9" --seeds 1 2 > harm_proxy_07b_s1.out
27. python notes/experiments/x123/harm.py proxy 7 8 --only "signs of coordinates 0..9" --seeds 2 3 > harm_proxy_07b_s2.out
28. python notes/experiments/x123/harm.py proxy 7 8 --only "signs of coordinates 0..9" --seeds 3 4 > harm_proxy_07b_s3.out
29. python notes/experiments/x123/harm.py proxy 7 8 --only "signs of coordinates 0..9" --seeds 4 5 > harm_proxy_07b_s4.out
30. python notes/experiments/x123/harm.py proxy 8 10 > harm_proxy_08.out
31. python notes/experiments/x123/harm.py proxy 10 11 --seeds 0 1 > harm_proxy_10_s0.out
32. python notes/experiments/x123/harm.py proxy 10 11 --seeds 1 2 > harm_proxy_10_s1.out
33. python notes/experiments/x123/harm.py proxy 10 11 --seeds 2 3 > harm_proxy_10_s2.out
34. python notes/experiments/x123/harm.py proxy 10 11 --seeds 3 4 > harm_proxy_10_s3.out
35. python notes/experiments/x123/harm.py proxy 10 11 --seeds 4 5 > harm_proxy_10_s4.out
36. python notes/experiments/x123/harm.py proxy 11 14 > harm_proxy_11.out
37. python notes/experiments/x123/harm.py proxy 14 18 > harm_proxy_14.out
38. python notes/experiments/x123/harm.py package 0 1 > harm_package_00.out
39. python notes/experiments/x123/harm.py package 1 4 > harm_package_01.out
40. python notes/experiments/x123/harm.py package 4 5 --only twice --seeds 0 1 > harm_package_04a_s0.out
41. python notes/experiments/x123/harm.py package 4 5 --only twice --seeds 1 2 > harm_package_04a_s1.out
42. python notes/experiments/x123/harm.py package 4 5 --only twice --seeds 2 3 > harm_package_04a_s2.out
43. python notes/experiments/x123/harm.py package 4 5 --only twice --seeds 3 4 > harm_package_04a_s3.out
44. python notes/experiments/x123/harm.py package 4 5 --only twice --seeds 4 5 > harm_package_04a_s4.out
45. python notes/experiments/x123/harm.py package 4 5 --only "sign of coordinate 0" --seeds 0 1 > harm_package_04b_s0.out
46. python notes/experiments/x123/harm.py package 4 5 --only "sign of coordinate 0" --seeds 1 2 > harm_package_04b_s1.out
47. python notes/experiments/x123/harm.py package 4 5 --only "sign of coordinate 0" --seeds 2 3 > harm_package_04b_s2.out
48. python notes/experiments/x123/harm.py package 4 5 --only "sign of coordinate 0" --seeds 3 4 > harm_package_04b_s3.out
49. python notes/experiments/x123/harm.py package 4 5 --only "sign of coordinate 0" --seeds 4 5 > harm_package_04b_s4.out
50. python notes/experiments/x123/harm.py package 5 6 --seeds 0 1 > harm_package_05_s0.out
51. python notes/experiments/x123/harm.py package 5 6 --seeds 1 2 > harm_package_05_s1.out
52. python notes/experiments/x123/harm.py package 5 6 --seeds 2 3 > harm_package_05_s2.out
53. python notes/experiments/x123/harm.py package 5 6 --seeds 3 4 > harm_package_05_s3.out
54. python notes/experiments/x123/harm.py package 5 6 --seeds 4 5 > harm_package_05_s4.out
55. python notes/experiments/x123/harm.py package 6 14 > harm_package_06.out
56. python notes/experiments/x123/harm.py package 14 15 > harm_package_14.out
57. python notes/experiments/x123/harm.py package 15 16 > harm_package_15.out
58. python notes/experiments/x123/harm.py package 16 22 > harm_package_16.out
59. python notes/experiments/x123/harm.py package 22 29 > harm_package_22.out
60. python notes/experiments/x123/harm.py package 29 32 > harm_package_29.out
61. python notes/experiments/x123/harm.py package 32 35 > harm_package_32.out
62. python notes/experiments/x123/harm.py summarize harm_proxy_*.out harm_package_*.out > harm_summary.out
```

**Frozen code** (sha256):

```
520162e3b0dff8b3f2896d409932f582cb222109c7be9f363ad05e0db1e8d217  gradcheck.py
0fc8917e28972726cce1547f2deb618444f4992efc8737b94d3df9d4f14517ea  families.py
f65d952eb8a3b8e9eaa47476e4966f518bad5d84a620ebaead6c22e816a6db43  runner.py
9aa36f393acafde8dcaf0953fec145b999124de9bafb21e86e86b6837ad9fd53  run_checks.py
57503989362f80594be6d641f46883e62ca20449b9f3cd9a232eddab07959994  harm.py
81c28e952e3ba70e260958aa81d32fd67ace746692335ce336c12c279a170d64  predictions.py
20c3d97d50613ad7f79f8c3bb37bfe31075330d4f5311d83749a2db555e50041  cost.py
265e964e89c562e60ec30a40916503a09057bb94ddbda95ae56d207bc340a531  integration.py
04426b1f8ae0f450f2b7559cfe2ff1ebb4de8188c3f1832705193cb50f53987b  equivalence.py
82ca3d2b567361855009e9777ecb9b08855beca631efdeb54f040f2716f2f021  prototype.py
```
