# x123, design round: the gradient check for a second experiment (#123)

This is a derivation with exploratory runs, not a result. Every number below names the
script that produced it; scripts and outputs are beside this file, and `README.md` lists
them and says how to run them. All runs used the package at commit 532137c, Python
3.13.18, NumPy 2.5.4, SciPy 1.18.1, x86_64 with 4 CPUs,
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. The prototype changed twice on
the exploratory seeds (section 12), and its final version is `../prototype.py`, a
formatted copy of the file these runs used (sha256
`f174cf81f3077b4d51f9ed6377b0eefdff9130c2fa1658af8811871b64833f3d` before formatting; the
unformatted file is not kept, and the scripts here that run the check import
`../prototype.py`). So none
of the rates below is a test: the second experiment must run on seeds nobody has used.

## 0. Answer

1. **Register (N), the noise-limited check, first, with (H) beside it as a comparator.**
   (N) warns when `grad` and a central difference of `log_prob` along a line disagree by
   more than ten times the estimated error of the comparison. It has no floor. (H) is (N)
   plus a floor derived from the sampler: 0.35 nats of mispredicted change of `log_prob`
   over the displacement between two draws of the initial distribution.
2. **Why no floor.** A wrong gradient harms through one quantity, the shift of the
   proposal's mean in proposal standard deviations, $\delta$ (section 1). Evaluating
   $\delta$ needs the sampler's covariance and step, which do not exist when the sampler
   is built, and needs the error where the chains will be, not where they start. Section 2
   shows that a directional check cannot both stay silent on gradients that are right to
   a relative accuracy $r$ and catch a wrong coordinate whose share of the line's
   predicted change is below $r/2$. The review's missed families have shares down to
   $1.2\times10^{-5}$. Every floor tried so far is a choice of $r$; only $r$ at the
   resolution of the difference catches all of them.
3. **Both designs use 8 points with one line each**, the line along the difference of two
   further draws of the initial distribution: 40 evaluations of `log_prob` and 8 of
   `grad` at any dimension.
4. **Predicted detection of one wrong-sign coordinate among $d$**, from a start at least
   as wide as the target: (N) misses with probability $3\times10^{-36}$ per check at
   $d = 1{,}000$ and $6\times10^{-29}$ at $d = 10{,}000$; (H) detects 0.99998 at both
   (the rate does not depend on $d$). With the start 200 times tighter in the wrong
   coordinate than in the others, (N) detects 0.9999 at $d = 1{,}000$ and 0.54 at
   $d = 10{,}000$, and (H) detects nothing (`predictions.out`, formulas checked in
   `formulas.out`).
5. **The price of (N)** is that it warns on any inconsistency it can resolve, about
   $3\times10^{-7}$ relative at $d = 1{,}000$: a gradient that is exact while `log_prob`
   comes from a solver at tolerance $10^{-5}$ warns 100 of 100 times (`diag_r.out`).
   That is a question of meaning for Josh, and the second experiment is laid out so that
   one run gives him the whole trade (section 10, group B).

## 1. What a wrong gradient does to the sampler

Write $f = \log\pi$ for the user's `log_prob`, $g$ for the user's `grad`, and
$e(x) = g(x) - \nabla f(x)$ for the gradient error. MALA with covariance
$\Sigma = CC^\top$ and step $\epsilon$ proposes
$x' = x + \tfrac{\epsilon^2}{2}\Sigma g(x) + \epsilon C z$ and accepts with the ratio that
uses $f$ and the proposal density built from $g$, so the chain stays exact for $\pi$.

**The drift error in proposal units.** The wrong gradient moves the proposal's mean by
$\tfrac{\epsilon^2}{2}\Sigma e(x)$. In units of the proposal's standard deviation
$\epsilon C$ this shift has length

$$\delta(x) = \tfrac{\epsilon}{2}\,\lVert C^\top e(x)\rVert .$$

Three facts follow, each checked in `harm_model.py`.

*The proposals differ by a known amount.* The right and the wrong proposal are Gaussians
with the same covariance and means $\delta$ standard deviations apart, so their total
variation distance is $2\Phi(\delta/2) - 1$: 0.05 at $\delta = 1/8$, 0.38 at $\delta = 1$.

*The acceptance rate falls by a known factor.* Writing $y = C^{-1}x$,
$\tilde g = C^\top g$ and $\tilde e = C^\top e$, the log acceptance ratio is
$f(y') - f(y) - \tfrac12 (y'-y)\cdot(\tilde g(y) + \tilde g(y')) + \tfrac{\epsilon^2}{8}(\lVert\tilde g(y)\rVert^2 - \lVert\tilde g(y')\rVert^2)$.
Substituting $\tilde g = \nabla f + \tilde e$ with $\tilde e$ locally constant leaves
the correct gradient's ratio plus

$$-\epsilon\, z\cdot\tilde e - \tfrac{\epsilon^2}{2}\lVert\tilde e\rVert^2 - \tfrac{\epsilon^2}{2}\nabla f\cdot\tilde e + O(\epsilon^3\lVert\tilde e\rVert)
\;\approx\; \mathcal N(-2\delta^2,\, 4\delta^2).$$

If the correct gradient's log ratio is $\mathcal N(-s_0^2/2, s_0^2)$, with acceptance
$a_0 = 2\Phi(-s_0/2)$, the wrong gradient's acceptance is

$$a = 2\Phi\big(-\tfrac12\sqrt{s_0^2 + 4\delta^2}\big),$$

which is $2\Phi(-\delta)$ when the correct gradient accepts everything. Simulated against
this model for a constant offset, at the step that gives 0.574 with the correct gradient
(`harm_model.out`, part 1):

| $d$ | $\delta$ | model | simulated |
|---|---|---|---|
| 100 | 0.125 | 0.565 | 0.566 |
| 100 | 0.5 | 0.452 | 0.465 |
| 100 | 1 | 0.251 | 0.276 |
| 1,000 | 0.25 | 0.538 | 0.539 |
| 1,000 | 1 | 0.251 | 0.262 |
| 1,000 | 2 | 0.038 | 0.042 |
| 1,000, sign of one coordinate, rms $\delta$ 0.52 | | 0.470 | 0.465 |

At $d = 5$ the model is rougher (0.251 against 0.319 at $\delta = 1$): the term
$\tfrac{\epsilon^2}{2}\nabla f\cdot\tilde e$ is not small when $\epsilon = 1.29$.

*Tuning answers with a smaller step.* Dual averaging restores the rate 0.574. With
$s_0^2 \propto \epsilon^6$ and $\delta \propto \epsilon$, the step shrinks by the factor
$c$ that solves

$$c^6 + \frac{4\delta_0^2}{s_*^2}\,c^2 = 1, \qquad s_*^2 = 1.264,$$

where $\delta_0$ is the root mean square of $\delta$ under $\pi$ at the correct
gradient's step. Simulated (`harm_model.out`, part 2): an offset with $\delta_0 = 1$
gives $c = 0.566$ at $d = 100$ against the model's 0.554; $\delta_0 = 2.9$ gives 0.194
against 0.194; twice the gradient at $d = 1{,}000$ ($\delta_0 = 8.2$) gives 0.069 against
0.068; the sign of one coordinate at $d = 1{,}000$ gives 0.882 against 0.850. The package
agrees: twice the gradient at $d = 50$ has $\delta_0 = \epsilon\sqrt d/2 = 3.0$, the
model gives $c = 0.19$, and default tuning gave steps of 0.173, 0.181, 0.153 against
0.837, 0.843, 0.827, ratios 0.19 to 0.21 (`harm_pkg_twice.out`); the review's
"+2 from a wide start" has $\delta_0 = 2.9$ and step ratios 0.26/1.29 = 0.20.

**What size of error is worth a warning.** The scale is $\delta_0$ (`predictions.out`,
part 3):

| $\delta_0$ | proposals differ (TV) | acceptance factor $2\Phi(-\delta_0)$ | step ratio $c$ | $1/c^2$ |
|---|---|---|---|---|
| 0.125 | 0.050 | 0.901 | 0.992 | 1.02 |
| 0.25 | 0.099 | 0.803 | 0.967 | 1.07 |
| 0.5 | 0.197 | 0.617 | 0.862 | 1.34 |
| 0.75 | 0.292 | 0.453 | 0.703 | 2.02 |
| 1 | 0.383 | 0.317 | 0.554 | 3.26 |
| 2 | 0.683 | 0.046 | 0.281 | 12.7 |

An error with $\delta \le \delta^* = 1/8$ cannot matter: the wrong sampler's proposal
coincides with the right one's except with probability 0.05, and the step moves by less
than 1%. An error with $\delta_0 \ge 1$ costs at least a factor 3 in $\epsilon^2$. So
$\delta^* = 1/8$ is the floor that follows from the sampler, and it is the floor of (H).

**Two cautions that the experiment must respect.**

- $\delta_0 \ge 1$ is where harm can start, not a measure of harm. Twice the gradient at
  $d = 5$ has $\delta_0 = 1.4$ and no measured slowdown (first experiment: 1.0), because
  an error parallel to $\nabla f$ only contracts harder; at $d = 50$ the same error slows
  meeting 84 to 94 times (`harm_pkg_twice.out`). The first experiment classed it
  harmless from $d = 5$ alone.
- A wrong sign hurts meeting far more than it shrinks the step, because the drift of that
  coordinate pushes the two chains apart. One wrong-sign coordinate has
  $\delta_0 = \epsilon$, which falls as $d^{-1/6}$, yet the slowdown of meeting grows
  with $d$: ratios of mean meeting times of 6.7 at $d = 50$ and 200, 9.6 at 1,000, 11.9
  at 3,000 and 13.1 ± 1.8 at 10,000, with step ratios of 0.74, 0.82, 0.89, 0.93, 0.95
  (`harm_diag_*.out`, a diagonal re-implementation of the coupled step; the package gave
  4.7, 4.8, 6.9 at $d = 50$ in `harm_pkg_twice.out`, and the review 4.5 to 5.2 at
  $d = 200$, so the re-implementation overstates by about 1.3). **One wrong coordinate
  among 10,000 is harmful**, so the high-dimensional families count.

For HMC with trajectories of length about 1 in whitened coordinates, the same argument
gives an energy error of standard deviation about $\lVert C^\top e\rVert$ over a
trajectory, that is $\delta$ with $\epsilon$ replaced by 1. This was not simulated.

## 2. What a directional check can know, and why (N) has no floor

A check that spends a fixed number of evaluations observes, on each line with step
vector $v$, one number: the gap $\Delta = e(x)\cdot v$, the change of `log_prob` over the
step that the gradient mispredicts. Everything else it holds ($g(x)$, $v$, the values of
$f$) is the same for a right and a wrong gradient to first order.

Take two gradients on the same line. In the first, every component of $g$ carries an
independent relative error of size $r$ (an approximate gradient: single precision, a
solver's tolerance, a small term left out). Then
$\Delta \sim r\,\lVert g\circ v\rVert\,\mathcal N(0,1)$, with
$\lVert g\circ v\rVert^2 = \sum_i g_i^2v_i^2$. In the second, the gradient is exact
except that coordinate 0 has the wrong sign, and coordinate 0 carries the share
$q = |g_0v_0|/\lVert g\circ v\rVert$ of the line's predicted change. Then
$\Delta = 2 g_0 v_0$, of size $2q\lVert g\circ v\rVert$. **The two give the same
observations when $r \approx 2q$**, so a rule that is silent on the first at accuracy $r$
misses the second at shares below $r/2$, whatever it is called.

The shares of the review's missed families follow from the scales. With
$\kappa_i$ the start's spread in coordinate $i$ over the target's scale, $g_iv_i$ is of
order $\kappa_i^2$, so $q \approx \kappa_0^2/\sqrt{\sum_i\kappa_i^4}$: $1/\sqrt d$ for
one coordinate among $d$ from a matched start (0.01 at $d = 10{,}000$), $5\times10^{-5}$
for scales (1, 0.01 ×4) from $\mathcal N(0, I)$, $1.2\times10^{-5}$ for a start of spread
0.01 in the wrong coordinate and 2 in four others. The first experiment's floor was
$r = 10^{-2}$.

The table makes the trade concrete. It adds a relative floor $r\lVert g\circ v\rVert$ to
(N)'s tolerance and counts warnings (`diag_r.out`; wrong gradients 200 seeds each, the
others 100).

| Gradient | $r = 0$, (N) | $10^{-6}$ | $10^{-4}$ | $10^{-2}$ | (H) |
|---|---|---|---|---|---|
| start sd (0.01, 2, 2, 2, 2): sign of coordinate 0 | 200 | 200 | 88 | 0 | 0 |
| start sd (0.1, 2, 2, 2, 2): the same | 200 | 200 | 200 | 77 | 0 |
| scales (1, 0.01 ×4), start $\mathcal N(0,I)$: sign of the wide coordinate | 200 | 200 | 189 | 0 | 200 |
| rotated, condition $10^8$: sign along the widest axis | 200 | 200 | 200 | 7 | 200 |
| start sd 1,000: gradient + 2 | 200 | 200 | 200 | 15 | 200 |
| start sd 0.001 in every coordinate: sign of coordinate 0 | 200 | 200 | 200 | 200 | 0 |
| Jacobi solver, tolerance $10^{-3}$, exact gradient | 100 | 100 | 100 | 0 | 0 |
| Jacobi solver, tolerance $10^{-5}$, exact gradient | 100 | 100 | 0 | 0 | 0 |
| Jacobi solver, tolerance $10^{-8}$, exact gradient | 3 | 0 | 0 | 0 | 0 |
| gradient × 1.005, $d = 5$ | 100 | 100 | 100 | 6 | 0 |
| logistic $n = 10^4$ without its prior | 100 | 100 | 89 | 0 | 0 |
| gradient × 1.005 from a start 100 times wide | 100 | 100 | 100 | 3 | 100 |

A floor in the sampler's units does not escape this. $\delta$ needs $C$ and $\epsilon$,
which tuning produces after $10d^2$ evaluations, and it needs $e$ where the chain lives.
At construction the only stand-in for $C$ is the initial distribution, and the error at
the start can be smaller than at stationarity by $\kappa_0^2$ (a tight start) or larger
by the same factor (a wide one). That is exactly (H): safe on the wide side, blind on
the tight side. The review found that the 1% floor "carries none of the false-alarm
result and all of the power loss"; section 6 finds the same without it on 92 correct
families.

So the choice is one of meaning, and it needs a ruling:

- **(N)**: the warning says "`grad` is not the derivative of `log_prob` as computed, to
  the accuracy differences resolve". It says nothing of speed. Small true disagreements
  warn, and the message must give their size.
- **(H)**: the warning says "the disagreement would shift a proposal by more than an
  eighth of its standard deviation if the target had the scales of the start".

I recommend (N) because its failure is a warning about a real, small inconsistency,
which a message can size and `check_grad=False` can silence, while (H)'s failure is
silence on a gradient that slows meeting 4 times, which is what the check exists to
prevent.

## 3. Candidate (N): the statistic and its invariances

**Points and lines.** From the generator
`replicate_generator(root_seed(rng), 3, 0)`, a namespace no run reads, draw $3P$ points of
the initial distribution $\pi_0$, $P = 8$: $x_p$, $a_p$, $b_p$ for $p = 1..P$, then two
arrays of random signs. The line at $x_p$ has direction $u_p = a_p - b_p$ and step vector
$v_p = h\,u_p$ with $h = \epsilon_f^{1/3}$ ($6.06\times10^{-6}$ for float64), where
$\epsilon_f$ is the epsilon of the dtype `log_prob` returns. Its points are
$x_p + t\,v_p$, $t \in \{-2, -1, 1, 2\}$.

The direction is a difference of two draws, not the spread times random signs, for three
reasons: it transforms with the parameters under any affine map (below); it follows a
correlated initial distribution, where the first experiment's direction, scaled by
each coordinate's spread, stepped about 6 target standard deviations along the narrowest
axis of a rotated target of condition $10^{12}$ ($h$ times a spread of order 1 over a
scale of $10^{-6}$); and draws cost no evaluation. Both directions detect the same on the review's
families (`det_run.out`, columns N and N,signs).

**Fallbacks.** Where $u_{pi} = 0$ (a point start, or an empirical start that drew the
same point twice), $u_{pi}$ is a random sign times the range of the $3P$ draws in
coordinate $i$, or $|x_{pi}|$ where that range is 0, or 1 where $x_{pi} = 0$. Where
$|u_{pi}| < \sqrt{\epsilon_{64}}\,|x_{pi}|$ it is raised to that size, so a step cannot
round away.

**The comparison**, per unit of $t$ and with the points as rounded. Let
$m_t = \mathrm{fl}(x_p + t v_p) - x_p$, $w_1 = (m_1 - m_{-1})\cdot v_p / \lVert v_p\rVert^2 \approx 2$
and $w_2 \approx 4$ likewise, and $f_t$ the value of `log_prob` at the point $t$. Then

$$D = \frac{f_1 - f_{-1}}{w_1},\quad D' = \frac{f_2 - f_{-2}}{w_2},\quad
G = \frac{g(x_p)\cdot(m_1 - m_{-1})}{w_1},\quad G' = \frac{g(x_p)\cdot(m_2 - m_{-2})}{w_2},$$

and the gap is $\Delta = |G - D|$, which estimates $|e(x_p)\cdot v_p|$ in nats. The line
warns when

$$\Delta > S\,(T + R_f + R_g), \qquad S = 10,$$

with the three error terms of section 4, and the check warns when any usable line does.
The inequality is strict and nothing is divided, so $0 > 0$ is silence. A line is
unusable when a value on it is $-\infty$ or when $w_1 \le 0$ or $w_2 \le w_1$; a point is
skipped when its density is zero or its gradient is not finite.

**Units.** Every term is a change of `log_prob` over the step $v_p$, in nats. No term
compares components of the gradient with each other, which was the first floor's defect.

**Invariances.** In real arithmetic the rounding terms vanish and the rule is
$\Delta > S\,T$.

*Affine maps of the parameters.* Let $x = Ay + b$ with $A$ invertible, and transform the
three inputs together: $\tilde f(y) = f(Ay + b) + k$, $\tilde g(y) = A^\top g(Ay + b)$,
and $\tilde\pi_0$ the law of $A^{-1}(X - b)$. Then $\tilde y_p = A^{-1}(x_p - b)$,
$\tilde u_p = A^{-1}u_p$, so the points of each line map onto each other:
$A(\tilde y_p + t\tilde v_p) + b = x_p + t v_p$. Hence every $\tilde f_t = f_t + k$, so
$D$, $D'$ are unchanged, and $\tilde G = (A^\top g)\cdot A^{-1}v_p = g\cdot v_p = G$. The
decision is unchanged. This covers every rescaling and rotation of the parameters, and
the rule needs no estimate of scale at all.

*Scaling of the log-density.* Under $f \to cf$, $g \to cg$ with $c > 0$, $D$, $D'$, $G$,
$G'$, the gap and $T$ are multiplied by $c$, and so are the rounding terms, which are
proportional to $|f|$ or $|g|$. The decision is unchanged, exactly when $c$ is a power of
2.

*Rescaling of the initial distribution alone.* The decision depends on $\pi_0$ only
through where the lines are. There is no floor that the spread enters, so a start that
does not match the target changes the power (section 5), not the meaning.

*What is not invariant, and must not be.* Adding a constant $K$ to $f$ leaves the rule
unchanged in real arithmetic, but the rounding of $f$ grows with $|f|$, so the smallest
detectable error grows as $|K|$ (section 5). The rounding term of the gradient uses
$\sum_i|g_i||v_i|$, which a rotation changes, as it changes the rounding itself.

Measured on a Student-t target under a random affine map of condition 100, 50 seeds, for
a wrong and a correct gradient (`invariance.out`): the decision of (N) differs on 0 of 50
under the affine map, under $c = 8$ and $c = 7.3$, and under $K = 10^6$; the ratio of gap
to tolerance is unchanged to 0 under $c = 8$, changes by at most 0.6% under $c = 7.3$ and
by at most 18% under the affine map (the truncation estimate then holds different
rounding). Under $K = 10^6$ the ratio falls by about the factor by which $|f|$ grew, as
stated (the reported relative change saturates at 1), and no decision changed because
the signal was far above both tolerances.

## 4. The error model and the threshold

Each term is an error of $D$ or $G$, per unit of $t$.

**Truncation, $T = |(D - G) - (D' - G')|$.** In units of the step,
$D = f' + \tfrac16 f''' + \tfrac1{120}f^{(5)} + \dots$ and
$D' = f' + \tfrac46 f''' + \tfrac{16}{120}f^{(5)} + \dots$ along the line, and $G = G'$
up to the rounding of the points. So $T = \tfrac12|f'''|(1 + O((h/\ell)^2))$, three times
the truncation error of $D$, where $\ell$ is the scale on which $f$ varies in units of the
direction. A correct gradient on a function that is smooth at the scale of the step
therefore has $\Delta/(S\,T) = 1/30$. Measured where truncation dominates: 0.0333 on the
banana, the funnel, pumps and Student-t (`fa_correct.out`, `fa_fa1.out`). Truncation
cannot cause a false alarm unless the fifth-order term rivals the third, that is, unless
$f$ has structure at the scale of the step.

**Rounding of `log_prob`, $R_f$.** Let each value carry an error of standard deviation
$\sigma$. The error of $D$ is at most $2\sigma/w_1$. The fourth difference
$F_4 = f_{-2} - 4f_{-1} + 6f_0 - 4f_1 + f_2$ has no contribution from a cubic, and its
rounding part has variance $(1 + 16 + 36 + 16 + 1)\sigma^2 = 70\sigma^2$, so
$|F_4|/\sqrt{70}$ estimates $\sigma$ with one degree of freedom per line. With
$L = \max_t|f_t|$ on the line,

$$R_f = \frac{2}{w_1}\max\Big(\hat r\,L,\ \hat s,\ \frac{|F_4|}{\sqrt{70}},\ U\epsilon_f L\Big),
\quad \hat r^2 = \operatorname{mean}_p\frac{(F_4/L)^2}{70},\quad
\hat s^2 = \operatorname{mean}_p\frac{F_4^2}{70},\quad U = 64.$$

The four entries answer four situations. $\hat r L$: rounding proportional to $|f|$,
pooled over the $P$ lines (a sum of terms of one sign). $\hat s$: rounding set by
cancelling terms larger than $f$, pooled in absolute size; the first version of the
prototype lacked it and reached a ratio of 0.84 on a correct gradient (rotated, condition
$10^{12}$, $d = 3$; `fa_fa1_v0.out`), 0.37 with it. $|F_4|/\sqrt{70}$ of the line itself:
a kink, which is an event of one line that pooling would dilute (below). $U\epsilon_f L$:
a floor for the case where the estimates are 0 by chance, because rounding errors of a
few units in the last place are not Gaussian.

$U = 64$ is the one constant that is inherited and not derived (both reviews upheld it).
On correct Gaussians the ratio $\Delta/(S\cdot U\epsilon L)$ has a median near
$7\times10^{-4}$ (`fa_correct.out`), so the actual error of $D$ is about 0.45 units in
the last place of $L$ and $S\,U = 640$ leaves a margin of 1,400. Every detection formula
of section 5 is linear in $S\,U$, so lowering $U$ is the cheapest gain in power; it
should be a registered variant of a later round, not a change made after seeing seeds.

**Rounding of `grad`, $R_g$.** The dot product and the gradient's own last bits give
$R_g \ge U\epsilon_g\sum_i|g_i||m_1 - m_{-1}|_i/w_1$, with $\epsilon_g$ the epsilon of
the dtype `grad` returns. For float64 this is $h$ times smaller than $R_f$ and never
binds.

**A float32 gradient of a float64 log-density.** A gradient computed in single precision
is, at best, the exact gradient at an input perturbed by a relative $\epsilon_g/2$ in
each coordinate, plus its own last bits. Its error along the line is therefore
$v\cdot H\theta$ with $|\theta_i| \le \epsilon_g|x_i|/2$ and $H$ the Hessian, which
$|g|$ does not bound: on $\mathcal N(10^6, I)$ it is 0.03 per coordinate where
$|g_i| \approx 2$. The check measures it instead of modeling it: one more evaluation of
`grad` per point at $x_p(1 + 4\epsilon_g s)$, $s$ random signs, gives
$q_i = |g_i(x_p(1 + 4\epsilon_g s)) - g_i(x_p)|$, the movement of the gradient under a
perturbation 8 times the cast's, and

$$R_g = \frac{1}{w_1}\sum_i \max\big(U\epsilon_g|g_i|,\ q_i\big)\,|m_1 - m_{-1}|_i .$$

This is an estimate of a condition number, not a bound: the probe has the size of the
true error times 8, summed without signs. It is done only where `grad` returns a dtype
narrower than float64. Measured (`fa_fa1.out`, rows 31 to 38, 100 seeds each): no warning
and a largest ratio of 0.26 on float32 gradients of $\mathcal N(0, I)$,
$\mathcal N(10^3..10^6, I)$, a logistic regression and rotated Gaussians of condition
$10^4$, $10^8$, $10^{12}$; the review counted 240 of 500 at $\mathcal N(10^6, I)$ and 83
of 500 at condition $10^{12}$ for the first design, and the prototype without the probe
warned 77 of 100 at condition $10^8$ (`fa_fa1_v1.out`), where the float32 gradient does
no measured harm (`harm_pkg_f32.out`: ratios of mean meeting times 1.01, 0.99, 1.01).
The probe does not blind the check: with a float32 gradient the sign of one coordinate
is detected 100 of 100 times on $\mathcal N(0, I)$ at $d = 5$ and $d = 1{,}000$ and on
$\mathcal N(10^3, I)$, and a factor of 1.0001 at $d = 5$ 87 of 100 times (`diag_r.out`,
rows 8 to 12 of part 2).

**A float32 log-density.** The points $x_p$ and the points of the lines are rounded to
the dtype of `log_prob` before use, so that the function sees the points the check
computes with; the step is $\epsilon_{32}^{1/3} = 4.9\times10^{-3}$ and the floor
$U\epsilon_{32}L$. Without the rounding, a target computed wholly in float32 reached a
ratio of 0.49 in the review; with it the largest ratio is $1.4\times10^{-3}$ on
$\mathcal N(0..10^4, I)$ (`fa_all32.out`).

**A log-density that is exactly constant on the line.** Then $D = D' = 0$ and $F_4 = 0$.
If $g(x_p) = 0$, every term is 0 and $0 > 0$ is silence: 0 of 300 on a flat density, a
flat integer one and a flat-topped one (`fa_fa1.out`, rows 0 to 2; the first experiment's
code warned 500 of 500). If $g(x_p) \ne 0$ the gap is $|G|$ against
$S(U\epsilon_f L + U\epsilon_g\sum|g_iv_i|)$, and the check warns, which is right: 50 of
50 on the flat top with `grad` $= -x$ (`invariance.out`).

**A kink or a boundary inside the line.** Let $f$ have a slope jump $J$ (per unit of
$t$) at position $t_0$. For $|t_0| < 1$ the gap is $J(1 - |t_0|)/2$, the truncation
estimate is $T = J|t_0|/4$ and $|F_4| = J\,|2 - 3|t_0||$; for $1 < |t_0| < 2$ the gap is
0. So

$$\frac{\Delta}{S(T + 2|F_4|/(w_1\sqrt{70}))} = \frac{(1 - |t_0|)/2}{2.5|t_0| + 1.195\,|2 - 3|t_0||} \le 0.209,$$

with the maximum at $t_0 = 0$, the case of a start exactly on a kink where `grad` returns
either one-sided slope. $T$ alone would fail for $|t_0| < 1/6$; the line's own fourth
difference is what covers it. Model: largest ratio 0.209 over 2 million positions
(`fa_model.out`, part 2). Measured: 0.209 on Laplace at $d = 1{,}000$ and 0.173 on a
hinge started at its kink. A jump $A$ in value has gap $A/2$ against $S\,T = 2.5A$. A
value of $-\infty$ makes the line unusable. Two or more slope jumps on one line can
cancel in $T$ and $F_4$: given two, a line warns with probability $2.6\times10^{-3}$
(equal signs) or $1.1\times10^{-3}$ (random signs and sizes), given three,
$8.9\times10^{-3}$ (`fa_model.out`).

## 5. Detection probability

**The general formula.** A line warns iff $h\,|e(x_p)\cdot u_p|$ exceeds the tolerance.
In float64, where rounding sets the tolerance, $T \approx 0$ and $R_f = U\epsilon L$, so
the line warns iff

$$|e(x_p)\cdot u_p| > C_N\,L + \tfrac{S h^2}{2}\,|\partial_u^3 f|, \qquad
C_N = S\,U\,\epsilon^{2/3} = 2.35\times10^{-8},\quad \tfrac{S h^2}{2} = 1.8\times10^{-10},$$

where $\partial_u^3 f$ is the third derivative along $u_p$ (0 for a Gaussian target). For
a float32 log-density the constants are $1.55\times10^{-2}$ and $1.2\times10^{-4}$. The
check warns iff any of its $P$ lines does; the lines are independent given $\pi_0$, so
the miss probability is the product.

Take $\pi = \mathcal N(0, \operatorname{diag}\sigma_i^2)$ and
$\pi_0 = \mathcal N(0, \operatorname{diag}s_i^2)$, write $\kappa_i = s_i/\sigma_i$,
$\xi = x_p/s$ and $\zeta = (a_p - b_p)/(\sqrt2 s)$, both standard normal. Then
$L \approx \tfrac12\sum_i\kappa_i^2\xi_i^2$.

**One wrong coordinate among $d$** (the sign of coordinate 0). Here
$e\cdot u = 2\sqrt2\,\kappa_0^2\,\xi_0\zeta_0$, so the line misses iff
$|\xi_0\zeta_0| < t^*$ with

$$t^* = \frac{C_N\sum_i\kappa_i^2\xi_i^2}{4\sqrt2\,\kappa_0^2} \approx 4.15\times10^{-9}\, d\,\frac{\overline{\kappa^2}}{\kappa_0^2},
\qquad P(\text{miss}) = F(t^*)^P,\quad F(t) = \frac2\pi\int_0^t K_0(w)\,dw \approx \frac{2t}{\pi}(1.116 - \ln t),$$

$F$ being the distribution function of the absolute product of two standard normals.
From a start matched in every coordinate the miss per line is $F(4.15\times10^{-9}d)$:
$4\times10^{-5}$ at $d = 1{,}000$, $3\times10^{-4}$ at 10,000, 0.017 at $10^6$. The
dimension enters only through $|f|$, whose rounding hides the one coordinate.

**A dense error of relative size $r$** ($g = (1 + r)\nabla f$). Here
$e\cdot u = r\sqrt2\sum_i\kappa_i^2\xi_i\zeta_i \approx r\sqrt2\,\kappa^2\sqrt d\,n$ for a
uniform $\kappa$, with $n$ standard normal, so the line warns iff

$$|n| > \frac{C_N\sqrt d}{2\sqrt2\,r} = 8.3\times10^{-9}\,\frac{\sqrt d}{r},$$

independent of $\kappa$. Of 100 checks, 99 detect $r = 3.4\times10^{-7}$ at
$d = 1{,}000$ and $1.1\times10^{-6}$ at $d = 10{,}000$ (`predictions.out`, part 2).

**A start tight or far wider in the wrong coordinate.** The formula for one coordinate
holds with $\kappa_0 \ne \overline\kappa$: the miss probability rises as
$(\overline\kappa/\kappa_0)^2 d$. A start wider than the target by $W$ in every
coordinate leaves $t^*$ unchanged for errors proportional to $x$ (signal and $|f|$ both
grow as $W^2$). For a constant offset $c$ in every coordinate,
$e\cdot u = c\sqrt2\,W\sqrt d\,n$ and the line warns iff
$|n| > C_N W\sqrt d/(2\sqrt2\,c)$, which is $9\times10^{-6}$ for $W = 1{,}000$, $d = 5$,
$c = 2$. A constant $K$ inside `log_prob` replaces $L$ by $|K|$:
$t^* = C_N|K|/(2\sqrt2\kappa_0^2)$.

**Model against measurement** (`formulas.out`; the model is a Monte Carlo of the
inequality with $L$ random, 2 million lines; the measurement is the prototype on 250
checks, 2,000 lines; target $\mathcal N(0, I_d)$):

| Design, error | $d$ | $\kappa_0$ | other $\kappa$ | size | line, model | line, measured | check, model | check, measured |
|---|---|---|---|---|---|---|---|---|
| N, sign | 100 | 0.01 | 2 | | 0.946 | 0.934 ± 0.006 | 1.000 | 1.000 |
| N, sign | 1,000 | 0.01 | 2 | | 0.692 | 0.687 ± 0.010 | 1.000 | 1.000 |
| N, sign | 10,000 | 0.01 | 2 | | 0.093 | 0.100 ± 0.007 | 0.541 | 0.544 |
| N, sign | 1,000 | 0.003 | 2 | | 0.075 | 0.071 ± 0.006 | 0.465 | 0.452 |
| N, sign | 5 | 0.001 | 2 | | 0.848 | 0.845 ± 0.008 | 1.000 | 1.000 |
| N, sign | 5 | 0.0001 | 2 | | 0.029 | 0.038 ± 0.004 | 0.211 | 0.268 |
| N, sign | 5 | 1 | $10^4$ | | 0.185 | 0.183 ± 0.009 | 0.806 | 0.816 |
| N, sign | 1,000 | 2 | 2 | | 1.0000 | 0.9995 | 1.000 | 1.000 |
| N, sign | 10,000 | 2 | 2 | | 0.9997 | 0.9995 | 1.000 | 1.000 |
| N, factor | 1,000 | 2 | 2 | $10^{-7}$ | 0.009 | 0.009 ± 0.002 | 0.070 | 0.072 |
| N, factor | 1,000 | 2 | 2 | $3\times10^{-7}$ | 0.382 | 0.378 ± 0.011 | 0.979 | 0.972 |
| N, factor | 10,000 | 2 | 2 | $10^{-6}$ | 0.407 | 0.404 ± 0.011 | 0.985 | 0.984 |
| N, factor | 1,000 | 100 | 100 | $3\times10^{-7}$ | 0.382 | 0.377 ± 0.011 | 0.979 | 0.972 |
| N, offset 2 | 5 | 1,000 | 1,000 | | 1.000 | 1.000 | 1.000 | 1.000 |

Every row of `formulas.out` agrees within 2.1 standard errors except the row at
$\kappa_0 = 10^{-4}$ (0.029 against 0.038 ± 0.004), where the tolerance is all rounding
and the model's $L$, taken at the center of the line, is rougher.

**On the review's families** (`det_run.out`, 200 seeds each, 0 warnings on every correct
counterpart): (N) detects 200 of 200 on 16 of 18, including every family of the review's
table. The two exceptions are at the limit the formula gives: scales
(1, $10^{-4}$ ×4) from $\mathcal N(0, I)$, 157 of 200 (model 0.81), where the starts are
$10^4$ standard deviations out and $|f| \approx 2\times10^8$; and a start of spread
$10^{-4}$ in the wrong coordinate, 53 of 200 (model 0.21).

## 6. False alarms on a correct gradient

**The argument, term by term.** For a correct gradient the gap is the error of $D$ and
of $G$.

- *Truncation* contributes at most $T/3$ against $S\,T$, a ratio of 1/30, unless $f$ has
  structure at the scale of the step (section 4).
- *One slope jump or one jump in value on a line* gives a ratio of at most 0.209
  (section 4), deterministically.
- *Rounding below $U$ units in the last place* gives a ratio near $10^{-3}$.
- *Rounding far above it* (cancellation): with independent Gaussian errors of equal size
  on the lines, the gap $|\eta_1 - \eta_{-1}|/2$ must exceed 10 times the larger of the
  pooled and the line's own estimate, and $T$, which contains the same errors, must also
  be small. Model: 0 warnings in 4 million checks (`fa_model.out`, part 1). When one line
  has $k$ times the noise of the others, the pooled estimate fails it and the line's own
  estimate has one degree of freedom: $5\times10^{-7}$ per check at $k = 3$,
  $1.4\times10^{-5}$ at $k = 10$, $3.9\times10^{-4}$ at $k = 100$. That last figure is
  the worst case of the noise model.

**Measured.** (N) warned on 0 of 9,200 checks over 92 correct families, 100 seeds each
(`fa_correct.out`: the first experiment's 44; `fa_fa1.out`: the review's 42 probes;
`fa_all32.out`: 6 float32 targets), with a largest ratio of 0.37 (rotated, condition
$10^{12}$, $d = 3$, where the difference is all rounding) and 0.15 on the first
experiment's 44. (H) warned on 0 of 9,200 too. On Laplace from $\mathcal N(0, 4I)$, (N)
warned on 0 of 5,000 checks at $d = 1{,}000$ and 0 of 1,000 at $d = 10{,}000$, where 1.2%
and 12.6% of the lines held a kink (`fa_model.out`, part 3). These seeds chose the
revisions of section 12, so the counts are a diagnostic.

**Where the rate is not negligible.**

1. *Structure at the scale of the step*: 23 of 100 with $10^5$ fringes across the start,
   100 of 100 with $10^6$ (`fa_reach.out`). Unchanged from the first experiment.
2. *Two or more kinks on one line.* A Laplace prior on $d$ coordinates puts a kink on a
   line with probability about $1.2\times10^{-5}d$ (measured), so two fall on one line
   with probability about $(1.2\times10^{-5}d)^2/2$, and the check warns about
   $8 \times 2.6\times10^{-3}$ times that: $2\times10^{-4}$ per check at $d = 10{,}000$,
   of order 1% at $d = 10^5$.
3. *`grad` and `log_prob` computed by different approximations*: the exact gradient of a
   log-density from an iterative solver warns 100 of 100 at tolerances $10^{-3}$ and
   $10^{-5}$ and 2 to 3 of 100 at $10^{-8}$ (`fa_reach.out`, `diag_r.out`). Under (N)'s
   meaning this is a true statement about `log_prob`, and it is the main cost of (N).
4. *Cancellation with very unequal noise between the points*: up to $4\times10^{-4}$ per
   check in the model above.
5. *A float32 gradient whose error is larger than its conditioning shows*: the probe is
   an estimate, tested on 8 families.
6. *A `log_prob` that is not deterministic.*

## 7. Cost

(N) and (H) evaluate `log_prob` at the $P$ points and at 4 points on each line, and
`grad` at the $P$ points: $5P = 40$ evaluations of `log_prob` and $P = 8$ of `grad` at
any $d$. A gradient of a narrower dtype adds $P$ evaluations of `grad` (the probe), and a
log-density of a narrower dtype adds $P$ of each (the points are evaluated again after
rounding). The prototype measured 40 + 8 for float64 and 48 + 16 with a float32 gradient
at $d = 1{,}000$ and $d = 10{,}000$, in 7 ms and 16 ms (`cost.out`); its extra 8 of
`log_prob` come from `Target`, which has no call for the gradient alone. The largest
count over all families was 56 + 24 (both dtypes narrow). It draws $3P = 24$ points of
the initial distribution.

Memory: the prototype peaks at 0.85 MB at $d = 1{,}000$ and 8.5 MB at $d = 10{,}000$,
107 vectors of $d$ float64, because it holds every draw and sign at once (`cost.out`).
Nothing in the design needs more than one line at a time: three draws, one direction,
four points and their differences, about 15 vectors of $d$ (0.12 MB at $d = 1{,}000$,
1.2 MB at $d = 10{,}000$), and the largest batch passed to the user's functions is 4
points. Both figures are linear in $d$; there is no $d^2$ term.

## 8. What (N) cannot detect, in a user's terms

- **Silence is not a pass.** The check compares at 8 points where the chains start.
- **An error that vanishes where the chains start.** If every chain starts in a small
  ball around the mode, a gradient with the wrong sign or a wrong factor is close to the
  right one there, since both are close to 0. With starts 1,000 times tighter than the
  target and a `log_prob` whose value is of order $10^4$, a wrong-sign coordinate is not
  detected ($t^* = C_N|K|/(2\sqrt2\kappa_0^2) = 83$).
- **An error in a parameter whose starts are much tighter than the others'**, when the
  dimension or $|$`log_prob`$|$ is large: the limit is
  $4\times10^{-9}\,d\,(\overline\kappa/\kappa_0)^2 \approx 1$, for example 200 times
  tighter at $d = 10{,}000$ (detected 54% of the time).
- **A gradient that is wrong only where the starts do not reach.**
- **Starts so far in the tail that `log_prob` is huge**: at $10^4$ standard deviations
  out, the rounding of `log_prob` hides a wrong-sign coordinate one time in five.
- **Errors below about $2\times10^{-8}\,|$`log_prob`$|$ nats across the spread of the
  start** in float64, and below $1.5\times10^{-2}\,|$`log_prob`$|$ when `log_prob` is
  float32.
- **With a float32 gradient, errors along the wide directions of a target whose scales
  span more than about $10^3$**: the gradient's own rounding is larger there.
- **Lines that reach zero density** are not compared; points of zero density are skipped.
- **Which parameter is wrong.** The warning can give the size of the disagreement along
  a line, not a coordinate.
- **And a warning is not a verdict on speed.** (N) warns on a term left out on purpose
  and on a gradient that is exact where `log_prob` is approximate.

## 9. Candidate (H): (N) with the floor that the sampler gives

(H) keeps everything of (N) and warns on a line only when also

$$\frac{\Delta}{h} > \varphi = 2\sqrt2\,\delta^* = 0.354 \text{ nats}, \qquad \delta^* = \tfrac18 .$$

**Derivation.** $\Delta/h$ estimates $|e(x_p)\cdot u_p|$, the change of `log_prob` that
the gradient mispredicts over the displacement between two draws of $\pi_0$. If the
covariance of $\pi_0$ equals the sampler's $\Sigma$, then
$e\cdot u \sim \mathcal N(0, 2\lVert C^\top e\rVert^2)$, with standard deviation
$2\sqrt2\,\delta/\epsilon$. Taking $\epsilon = 1$, the length of an HMC trajectory in
whitened coordinates and within a factor 1.65 of MALA's step at every $d$, the gap of a
line exceeds $\varphi$ with probability $2\Phi(-\delta^*/\delta)$.

**Invariances.** The affine invariance of (N) holds: $e\cdot u$ is unchanged. Adding a
constant to $f$ changes nothing. Scaling $f$ by $c$ changes the decision, as it should:
nats are absolute and $c\,f$ is another target with $\delta$ scaled by $c$
(`invariance.out`: (H) warns 0, 50, 50 of 50 on a small offset at $c = 1, 10, 100$).
Rescaling $\pi_0$ alone by $W$ multiplies $e\cdot u$ by $W^2$ for errors proportional to
$x$ and by $W$ for constant ones: the floor's meaning moves with the start.

**Detection.** For the sign of one coordinate, the line warns iff
$|\xi_0\zeta_0| > \delta^*/\kappa_0^2$, at any $d$: $P(\text{miss}) = F(0.125/\kappa_0^2)^P$,
that is $1.8\times10^{-5}$ for $\kappa_0 = 1$, $5\times10^{-9}$ for $\kappa_0 = 2$, 0.015
for $\kappa_0 = 0.5$, 0.60 for $\kappa_0 = 0.25$ and 1 for $\kappa_0 \le 0.1$
(`predictions.out`). For a dense relative error, the line warns iff
$|n| > 0.25/(r\kappa^2\sqrt d)$; 99% of checks detect $r = 1.0\%$ at $d = 1{,}000$ and
0.32% at $d = 10{,}000$ for $\kappa = 1$, whose $\delta_0$ is 0.08 and 0.06, below harm.
Model against measurement (`formulas.out`):

| Error | $d$ | $\kappa_0$ | other $\kappa$ | size | line, model | line, measured | check, model | check, measured |
|---|---|---|---|---|---|---|---|---|
| sign | 5 | 1 | 1 | | 0.745 | 0.743 ± 0.010 | 1.000 | 1.000 |
| sign | 1,000 | 1 | 1 | | 0.746 | 0.742 ± 0.010 | 1.000 | 1.000 |
| sign | 10,000 | 1 | 1 | | 0.745 | 0.745 ± 0.010 | 1.000 | 1.000 |
| sign | 1,000 | 0.5 | 2 | | 0.410 | 0.411 ± 0.011 | 0.985 | 0.988 |
| sign | 5 | 0.01 | 2 | | 0.000 | 0.000 | 0.000 | 0.000 |
| factor | 1,000 | 2 | 2 | 0.001 | 0.048 | 0.049 ± 0.005 | 0.326 | 0.332 |
| factor | 1,000 | 2 | 2 | 0.003 | 0.510 | 0.501 ± 0.011 | 0.997 | 0.996 |
| factor | 1,000 | 1 | 1 | 0.01 | 0.429 | 0.419 ± 0.011 | 0.989 | 0.980 |
| offset 0.1 | 5 | 1 | 1 | | 0.264 | 0.248 ± 0.010 | 0.914 | 0.896 |

**What it buys and what it loses** (`diag_r.out`, `det_run.out`). It is silent on the
solver at $10^{-3}$, on a factor 1.005, on the prior left out at $n = 10^4$ and on a
float32 gradient, where (N) warns. It detects 0 of 200 on every family whose start is
tight in the wrong coordinate (spread 0.1, 0.01, 0.001; a ball of 0.001; scales
(1, 0.01 ×4) from a start matched to the narrow scale), two of which the review measured
as slowing meeting 3.2 and 4.2 times. From wide starts it warns on harmless errors
(a factor 1.005 from a start 100 times wide: 100 of 100).

False alarms, cost and memory are those of (N).

## 10. A third candidate, not for this registration: (T), after the pilot

The floor of section 1 is well defined once the kernel is tuned: the pilot's last states
are near stationarity, and $\Sigma$ and $\epsilon$ exist. Lines along $v = \epsilon C z$
from those states have $e\cdot v \sim \mathcal N(0, 4\delta^2)$ exactly, so the gap
divided by $h$ measures $2\delta$ in the sampler's own units, at the cost of 4 points.
This is the only place where "would this error slow the sampler" has a direct answer,
and it covers the tight starts. It comes after the $10d^2$ evaluations of the pilot, not
when the sampler is built, and it does not exist when the user gives the kernel's
parameters, so it complements (N) and does not replace it. I did not prototype it.

## 11. Proposal for the second pre-registration

**Designs.** (N) and (H) as in `../prototype.py`, $P = 8$, frozen by hash before any run on the
new seeds. The verdict is for each design separately. For every line the runner records
the gap, the tolerance and $\lVert g\circ v\rVert$, so that the table of section 2 (the
relative floors $r = 10^{-6}, 10^{-4}, 10^{-2}$) is reported from the same runs without
being part of a verdict.

**Seeds.** A range nobody has used; the first experiment used 0 to 41,004, its review
60,000 to 189,999, and this design 200,000 to 219,999. I suggest 300,000 to 399,999,
assigned by family index in the registration.

### Group A: correct gradients (a warning is a false alarm), 300 seeds each

About 100 families, 30,000 checks per design:

- the first experiment's 44, and the review's 42 probes, `cases.REVIEW_CORRECT` (flat, flat integer
  and flat-topped log-densities; kinks at and near the start; naive sums of $10^5$ terms;
  heavy tails from starts of spread $10^6$ and $10^8$; rotated targets of condition
  $10^{12}$; degenerate, two-point and point starts; float32 log-densities; float32
  gradients of float64 log-densities at offsets to $10^6$ and conditions to $10^{12}$;
  functions that are not vectorized), and the 6 float32 targets, `cases.all32`;
- new, for the review's findings: $\mathcal N(0, I)$, Student-t and Laplace at
  $d = 2{,}000$, 5,000 and 10,000; a rotated Gaussian of condition $10^4$ at
  $d = 2{,}000$; correct gradients from the starts that the wrong families use (scales
  (1, 0.01 ×4) from $\mathcal N(0, I)$; spread 0.01 in one coordinate; a ball of spread
  $10^{-3}$ around the mode with $-10^4$ added to `log_prob`; starts 1,000 times wide);
  a constant `log_prob` other than 0; a ReLU network and a piecewise-linear table.

### Group B: gradients that are not the derivative of `log_prob` as computed and do no harm (reported; the ruling's evidence)

The Jacobi solver at tolerances $10^{-3}$, $10^{-5}$, $10^{-8}$ with the exact gradient;
a factor of 1.005 and 1.0001; an offset of 0.02; the logistic gradient without its prior
at $n = 10^4$; the same from a start 100 times wide. For each: the warnings of (N), of
(H) and of the relative floors, and the harm. No bar. Predicted from `diag_r.out`: (N)
warns on all but the solver at $10^{-8}$; (H) on none but the wide start.

### Group C: wrong gradients, 400 seeds each

Every family comes with its harm (below) and with the rate each design's formula
predicts, written into the registration.

- **Whole gradient**: wrong sign ($d = 5$); twice the gradient at $d = 5$ (predicted
  harmless), 50 and 200 (predicted harmful, measured 84 to 94 at $d = 50$); the gradient
  of $\mathcal N(0, I)$ on a Student-t; a wrong correlation.
- **One coordinate among $d$**: the sign of coordinate 0 of $\mathcal N(0, I_d)$ at
  $d = 5$, 50, 200, 1,000, 2,000, 5,000, 10,000, from starts of $\kappa = 1$ and
  $\kappa = 2$; the signs of 10 coordinates at $d = 10{,}000$; one coordinate set to 0.
- **Starts not matched to the target's scales**: scales (1, 0.1 ×4) and (1, 0.01 ×4)
  from $\mathcal N(0, I)$, sign of the wide coordinate; the same from a start matched to
  the narrow scale.
- **Starts tight in the wrong coordinate**: spread 0.1 and 0.01 in coordinate 0 and 2 in
  the others, at $d = 5$, 100, 1,000.
- **Starts far wider than the target**: gradient + 2 from spreads of 100, 300, 1,000 and
  $10^4$; the sign of one coordinate from a spread of 1,000.
- **Rotated targets**: the sign along the widest principal axis at conditions $10^2$ and
  $10^4$ ($d = 10$), and of one principal axis of a randomly rotated Gaussian of
  condition $10^4$ at $d = 2{,}000$.
- **Other precisions and flat lines**: a float32 gradient with a wrong-sign coordinate
  ($\mathcal N(0, I)$ at $d = 5$ and 1,000, $\mathcal N(10^3, I)$); a flat-topped
  log-density with `grad` $= -x$.
- **Point starts**: gradient + 2(1, −1) from the point (1, 1) at $d = 2$ (with 8 sign
  vectors the fallback direction is orthogonal to the error every time with probability
  $2^{-8}$), and a wrong-sign coordinate from a random point at $d = 5$.
- **Registered as formula checks, 1,000 seeds each, because the predicted rate is
  between 0.05 and 0.95**: spread 0.01 in the wrong coordinate at $d = 10{,}000$ ((N):
  0.54); spread 0.003 at $d = 1{,}000$ (0.46); spread $10^{-4}$ at $d = 5$ (0.21); scales
  (1, $10^{-4}$ ×4) from $\mathcal N(0, I)$ (0.81); a ball of spread $10^{-3}$ with
  $-10^4$ added, sign of one coordinate ((N): 0); and for (H), the sign of one
  coordinate from $\kappa_0 = 0.5$ (0.985) and $\kappa_0 = 0.25$ (0.40).

### Harm

The first experiment took the median over 5 seeds of ratios of medians of meeting times
that are small integers (3 against 9), which jumps near its threshold of 3. Replace it:

- **The measure** is the ratio of mean meeting times,
  $\Lambda = \overline{\tau}_{\text{wrong}}/\overline{\tau}_{\text{correct}}$, over 5
  seeds of 100 pairs each (500 pairs a side), default-tuned coupled MALA, with a pair
  unmet at `max_iter` = 10,000 counted at 10,000 (so $\Lambda$ is then a lower bound),
  and with a standard error from the 5 per-seed ratios. At $d = 50$ one seed already
  gives a standard error near 9% (`harm_diag_50.out`).
- **Classes.** Harmful iff $\Lambda - 2\,\mathrm{se} \ge 2$. Harmless iff
  $\Lambda + 2\,\mathrm{se} \le 1.5$. Otherwise unclassed: reported, and outside both
  bars. A family whose correct gradient leaves more than 5% of pairs unmet is
  undetermined, as before.
- **Above $d = 200$**, where the pilot costs $10d^2$ steps and the package's MALA holds a
  dense covariance: the diagonal coupled step of `harm_diag.py` with the step that gives
  0.574 at stationarity for the gradient in use, validated in the same experiment against
  the package at $d = 50$ and $d = 200$ on the same families. Exploratory agreement:
  6.7 against 4.7 to 6.9 at $d = 50$, 6.7 against the review's 4.5 to 5.2 at $d = 200$,
  so it overstates by about 1.3 and a family is harmful there iff
  $\Lambda - 2\,\mathrm{se} \ge 3$.
- **Beside each $\Lambda$**, the ratio of tuned steps and $\delta_0$, which are
  continuous and cheap, so that the reader sees where a family sits on the scale of
  section 1. They do not class a family: a wrong sign has a step ratio of 0.95 at
  $d = 10{,}000$ and slows meeting 13 times.

### Criteria, with the power of each test

All probabilities are from `predictions.out`, part 4.

1. **False alarms, group A.** At most 2 warnings in 300 on every family, and at most 15
   in the 30,000 pooled (0.05%). The pooled test passes with probability
   $4\times10^{-12}$ if the true rate is 1 in 500, 0.002 if it is 1 in 1,000, and 0.9995
   if it is 1 in 5,000; section 6 predicts below 1 in 10,000 on these families. The
   per-family test passes with probability 0.006 for a family whose true rate is 3%, 0.06
   at 2% and 0.9964 at 0.1%; if every family had the worst rate of the noise model,
   0.04%, all 100 would pass with probability 0.974.
2. **Detection, group C.** On every harmful family whose predicted rate is at least
   0.999: at least 390 warnings in 400. The test passes with probability 0.009 if the
   true rate is 0.95 and 0.9973 if it is 0.99. For comparison, the first experiment's bar
   of 180 in 200 passed with probability 0.79 at the true rate of 0.913 that the review
   measured: the bar sat within two standard errors of the truth, which is what "a
   favorable draw" means. Here every bar is at least 5 standard errors from the predicted
   rate.
3. **Formula checks.** On each family listed as such: the measured rate within 0.05 of
   the registered prediction, on 1,000 seeds. If the formula is right the test fails
   with probability at most 0.0015; if the true rate is off by 0.1 it fails with
   probability above 0.99. A design whose formula fails has an unexplained behavior and
   does not pass, whatever its rate.
4. **Predicted misses.** Where a design's formula predicts a rate below 0.001 on a
   harmful family (for (H): every start tight in the wrong coordinate by 10 or more),
   the registration says so, and the design fails criterion 2 there by prediction; the
   count confirms it (at most 10 in 400 holds with probability 0.009 if the true rate
   were 5%).
5. **Cost at $d = 1{,}000$ and $d = 10{,}000$**: at most 60 evaluations of `log_prob`
   and 25 of `grad`, and a peak of at most 20 vectors of $d$ float64 during the check.
   The prototype holds 107 vectors, so the frozen code must work one line at a time
   (section 7); that rewrite changes no number of the statistic.
6. **No effect on results**: runs with and without the check byte-identical, as the
   first review verified for this namespace.

**Predicted outcome**, so that the experiment can refute it: (N) passes 1, 2, 3, 5, 6 and
has no predicted miss among the families classed harmful, unless the ball of spread
$10^{-3}$ with a large `log_prob` turns out harmful, which is its registered blind spot.
(H) passes 1, 3, 5, 6 and fails 2 through 4 on the tight starts. If (N) fails 1, the
first thing to register next is "two lines must disagree", which squares the per-line
rates of section 6.

**Compute.** A check takes 7 ms at $d = 1{,}000$ and 16 ms at $d = 10{,}000$, so group A
and group C are minutes. The harm runs dominate: default tuning at $d = 200$ took the
review about 95 s a run, so two families at $d = 200$ cost about 30 minutes; the diagonal
step took 52 s for 50 pairs at $d = 10{,}000$.

## 12. What is uncertain

- **The meaning of a warning needs Josh's ruling** (section 2). (N) will warn for users
  whose gradient is exact and whose log-density is a solver's output at tolerances looser
  than about $10^{-8}$. If that is not acceptable, the alternative that the evidence
  supports is a relative floor $r$, with the misses of section 2's table, not (H).
- **$U = 64$ is inherited**, with a measured margin of 1,400 on well-conditioned
  targets. All power figures scale with it.
- **The noise model is Gaussian and independent**; real rounding is neither. The worst
  case of the model is $4\times10^{-4}$ per check and the largest ratio seen on a correct
  gradient is 0.37, a margin of 2.7 on a family where the difference is pure rounding.
- **The probe for narrow gradients is an estimate**, argued and tested on 8 families.
- **Harm above $d = 200$ rests on a re-implementation** that overstates by about 1.3 at
  $d = 50$ and 200. HMC's harm was not simulated at all; only MALA's.
- **The prototype was revised twice on the exploratory seeds**: the first version,
  `prototype_v0.py` (relative pooling only; every point rounded to the narrowest dtype),
  reached a ratio of 0.84 on a correct gradient and could not step at
  $\mathcal N(10^6, I)$ in float32 (`fa_fa1_v0.out`); the second, `prototype_v1.py`
  (absolute pooling added; points rounded to the gradient's dtype), warned 77 of 100 on a
  harmless float32 gradient at condition $10^8$ (`fa_fa1_v1.out`); the final version,
  `../prototype.py`, adds the probe. The float64 results of `formulas.out`,
  `fa_model.out` and `det_run.out` were first produced by the second version; the last
  revision does not change them, and all three were rerun with the final prototype and
  are identical.
- **Not tested here**: byte-identical runs (the prototype is not in the package); errors
  raised by the target inside the check (the first experiment's `GradientCheckError`
  carries over); detection from point starts; empirical starts with few points beyond
  two false-alarm families.
- **A larger step would reach tighter starts** (the signal grows as $h$, the rounding
  does not), but at $h = \epsilon^{1/6}$ a Laplace target at $d = 1{,}000$ would have
  about 10 kinks on every line, where section 6 gives false alarms at the percent level.
  I kept $h = \epsilon^{1/3}$ and state the blind spot.

## 13. Scripts, outputs and seeds

Each script is run from the root of the repository as
`PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/design/<script>`,
with the three thread variables set to 1; `README.md` lists every file.

| Script | Output | Seeds |
|---|---|---|
| `../prototype.py` | the final prototype; `prototype_v0.py` and `prototype_v1.py` are its two earlier versions | |
| `cases.py` | the targets of the first experiment and of its review that the scripts run on | |
| `fa_run.py` | `fa_correct.out`, `fa_reach.out`, `fa_fa1.out`, `fa_all32.out`; with the earlier prototypes, `fa_fa1_v0.out` and `fa_fa1_v1.out` | 200000..204399, 204400..205099, 205100..209299, 209300..209899 (each run three times, once per version) |
| `fa_model.py` | `fa_model.out` | generator 218000; Laplace on 200000..204999 ($d = 1{,}000$) and 200000..200999 ($d = 10{,}000$) |
| `det_run.py` | `det_run.out` | 210000..213599 |
| `diag_r.py` | `diag_r.out` | 210000..213599 again, and 216000..217299 |
| `formulas.py` | `formulas.out` | 213600..213849, the same for every row; generator 213600 |
| `predictions.py` | `predictions.out` | generator 213600 |
| `harm_model.py` | `harm_model.out` | generators 217000..217002 and 217006..217008 |
| `invariance.py` | `invariance.out` | 217500..217549; generator 217500 |
| `harm_pkg.py` | `harm_pkg_twice.out`, `harm_pkg_f32.out` | 219000..219002 |
| `cost.py` | `cost.out` | 219500 |
| `harm_diag.py` | `harm_diag_50.out`, `_200`, `_1000`, `_3000`, `_10000` | generators 219600..219602 |
| a smoke test of the prototype | none kept | 200000, 200001 |

Every seed is in 200000..219999. No other range was used.
