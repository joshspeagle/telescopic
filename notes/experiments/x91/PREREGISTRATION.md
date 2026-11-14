# x91: the test of contraction-first tuning (#91) and of two remedies for the leapfrog resonance (#137), registered before any run

This file fixes, before any run of the experiment, its arms, targets, seeds, measurements,
predictions, criteria and decision rules. It answers Josh's rulings of 2026-11-13 on #89
(question 1: contraction-first tuning is opt-in now and becomes the default only if its
pre-registered test passes) and of 2026-11-14 on #137 (a guard that acts only near a
resonance and a shared step jitter go to this test, with the Gaussian of unequal scales
added; the guard as tested in the experiment of #148 is not adopted). It is not changed
after it is posted on #91. Where it must be corrected, a dated note is added below the
text and the outcome is scored against the text as posted. Section 14 lists what changed
after an adversarial review of the first version.

Short names: **P** is the package as it is; **G** the guard; **J2, J3, J5** the shared
jitter at widths 0.2, 0.3 and 0.5; **CF** the contraction-first tuning of #91; **P.2** and
**CF.2** the same two at a meeting weight of 0.2; **MP, MJ5, MCF** the corresponding MALA
arms. A **cell** is a target with its settings; a **record** is one arm at one seed of one
cell.

## 1. What was seen before this registration

- **The experiment of #137** (`notes/experiments/i137_*`, records at commit `b0b8f71`,
  seeds 0 to 99 for tunings and 0 to 29 for pilots): its outputs, its predictions file
  with the notes added after the runs, and its stores of records.
- **The design probes of this registration**, run at commit `4cb3acd` (the package's
  source is the same at `1d7ba37`), one thread, tuning only: no pilot pair and no
  replicate of any arm was run. They are the files listed in section 13, to be committed
  beside this file.
  - The package's tuning of `examples.gaussian(10)` with `n_steps` 3, 5 and 10 at seeds 0
    to 99 (`tunings_d10.jsonl`), and of the Gaussian of unequal scales with
    `cov=np.eye(10)` and `n_steps` default, 3 and 10 at seeds 0 to 29, with the pilot
    covariance (`tunings_unequal.jsonl`). These are i137's seeds. With the package at
    `4cb3acd` the tuned steps differ from the stored ones by up to 3.7% (relative), far
    more than the last bits that decision row 189 changed, so the probe tunings are a
    second sample of these cells, not a copy of i137's.
  - The tunings of P and CF on nine targets at probe seeds 900 to 902
    (`probe_tune.out`), of P on rotated Gaussians of condition number 10^4 at d = 10, 20
    and 50, seeds 900 and 901 (`probe_rot.out`), and at d = 100, seed 900, of P and CF
    (`probe100.out`).
  - For the held-out cells added after the review: P's tunings of `examples.gaussian(5)`
    with `n_steps` 2, 3, 4, 6, 8 and of Gaussians of unequal scales in 6 and 8 dimensions
    at probe seeds 950 to 955, used to choose the cells (`scan_heldout.out`), then 100
    tunings of `gaussian(5)` with `n_steps` 3, 4, 6 at probe seeds 900 to 999 and 30 of
    the chosen unequal Gaussian at 900 to 929 (`tunings_g5.jsonl`,
    `tunings_uneq8.jsonl`, `heldout.out`).
  - Computations on stored and probe tunings only: the guard's rule and its
    alternatives (`evalguard.out`, `evalunequal.out`, `post.out`, `trig.out`,
    `phases.out`), the jittered contraction factors (`jpred.out`), the meeting-time
    model (`model.out`), the spreads between seeds and the power of the criteria
    (`spreads.out`, `ineffpower.out`, `kpower.out`), and the stores' recorded seconds
    (`costs.out`).
- **The adversarial review of the first version** and its probes (`p1_guard`,
  `p2_claims`, `p3_power`, `p4_mala`; tuning only, probe seed 900, and i137's stores).
  Figures quoted from them name the probe.
- **Not seen:** any meeting time, `k`, estimate or inefficiency of any arm at any seed
  registered in section 5, and any meeting time of G, J2, J3, CF, P.2, CF.2 or the MALA
  arms at any seed.

## 2. The arms

On a whitened Gaussian mode of frequency $w$, with $h = \varepsilon w$ and
$\theta(h) = 2\arcsin(h/2)$ for $h < 2$, one accepted trajectory of $n$ leapfrog steps
with a shared momentum multiplies the gap between the chains along that mode by
$\cos(n\theta)$ (`i137_reproduce_unequal.out` finds the ratio of distances equal to
$\cos(n\theta)$ at three tuned steps). At $n\theta = j\pi$, $j \ge 1$, the pair does not
contract along that mode: a resonance.

### HMC arms

| Arm | What it is | What it changes |
|---|---|---|
| P | `HMC(step_size=None, n_steps=..., cov=...)` as on `main` at the run's commit | nothing |
| G | P, then the guard below moves the tuned step | the tuned step only; P's kernel and draws. Where it keeps the step, its record is P's, bit for bit |
| J2, J3, J5 | P (acceptance 0.8) with every trajectory's step $\varepsilon(1 + j(2u - 1))$, one uniform $u$ per step, shared by the two chains of a pair; $j$ = 0.2, 0.3, 0.5 | the kernel: one more uniform per step, so the numbers of every HMC run |
| CF | #91 exactly: J5, with dual averaging toward an acceptance rate of 0.99 instead of 0.8 | the kernel and the tuning |
| P.2, CF.2 | P and CF with `meeting_weight=0.2` | as P and CF, and the meeting weight |

**The jitter (J2, J3, J5, CF, CF.2).** The nominal step $\varepsilon$ is what dual
averaging returns, tuned on the jittered kernel itself (at each of its 1,000 steps each of
its 4 chains draws its own $u$). The number of leapfrog steps is fixed by the nominal
step: the given `n_steps`, or $\lceil 1/\varepsilon\rceil$ (at most 1,000), so the jitter
also varies the integration time between $1 - j$ and $1 + j$ of its nominal value, as #91
states. The draw order of a step is: the `Mixture`'s uniform that chooses the component;
then, for a trajectory, the jitter's uniform, `d` normals for the momentum, the
acceptance uniform; for the random-walk meeting step, the package's draws, unchanged and
not jittered. A coupled step reads the same numbers, each shared by the two chains
(`i137_common._JitterLeapfrog` is this kernel). The meeting step keeps the package's
`meeting_weight=0.05` (0.2 for P.2 and CF.2) and `meeting_scale=1e-3`. Exactness: for a
fixed step the leapfrog Metropolis kernel leaves $\pi$ invariant; $u$ is drawn
independently of the state, so the jittered kernel is a mixture of $\pi$-invariant
kernels; each chain reads its numbers in the order of the step alone, so its marginal
transition is that of the uncoupled kernel, and equal chains get identical arithmetic.

**The guard (G).** After the package's tuning returns $\varepsilon_0$ and the count
$n(\varepsilon)$ (the given `n_steps`, or $\lceil 1/\varepsilon\rceil$):

1. *Frequencies read:* the predicted whitened frequencies
   $w_i = \sqrt{\lambda_i(C^\top S^{-1} C)}$, with $CC^\top$ the kernel's covariance and
   $S$ the pilot covariance (`PilotRun.cov`). With the package's tuned covariance
   $C C^\top = S$, so every $w_i = 1$.
2. *Distance to a resonance:*
   $r(\varepsilon, w) = \min_{j \ge 1} \lvert n(\varepsilon)\theta(\varepsilon w) - j\pi\rvert$,
   and $r = 0$ where $\varepsilon w \ge 2$ (unstable). A phase below $\pi/2$ has
   $r > \pi/2$: the $j = 0$ end is never guarded.
3. *Near:* the tuned step is near a resonance when $\min_i r(\varepsilon_0, w_i) < \delta$,
   $\delta = \arccos 0.9 = 0.4510$ (a predicted factor $\lvert\cos n\theta\rvert$ above
   0.9 at some predicted frequency). If it is not near, the step is kept exactly.
4. *Move:* otherwise the step becomes the grid step $\varepsilon_0 e^{s}$,
   $s \in \{-2, -1.999, \dots, 0.1\}$, of smallest $\lvert s\rvert$ (the smaller step on a
   tie) at which $r(\varepsilon, w_i f) \ge \delta$ for every $i$ and every
   $f$ in 41 points geometric from $1/1.3$ to $1.3$ (the band), with $n(\varepsilon)$
   recomputed at each grid step.
5. *No step satisfies 4:* the tuned step is kept (the record says "fallback").

The record keeps $\varepsilon_0$, the action (kept, moved, fallback), and
$\min_i r$ before and after.

**Combinations.** None is run. J alone covered every case the old guard covered in
i137, so G + J could only matter if G passes the Gaussian table and every J width fails a
cost criterion; that outcome is the trigger for a next experiment, not a reason to spend
this one's budget on it.

### MALA arms

G does not apply to MALA: its coupling is the reflection-maximal coupling of the two
proposals, not a shared-noise trajectory, and a single step has no resonance below the
stability limit $h = 2$.

| Arm | What it is | What it changes |
|---|---|---|
| MP | `MALA()` as on `main` | nothing |
| MJ5 | MALA at the step $\varepsilon(1 + 0.5(2u - 1))$, one uniform $u$ per step shared by the pair, dual averaging toward 0.574 on the jittered kernel | the kernel |
| MCF | MJ5 with dual averaging toward 0.99 (#91: "repeat the grid for MALA") | the kernel and the tuning |

Draw order: the jitter's uniform, then the package's draws of the step (or of the
coupled step), unchanged. The coupled proposals are the package's reflection-maximal
coupling of $\mathcal N(m_x, \varepsilon_u^2\Sigma)$ and $\mathcal N(m_y, \varepsilon_u^2\Sigma)$,
computed with the factor $\varepsilon_u C$ of each pair's own step; the acceptance ratio
is MALA's at $\varepsilon_u$.

**Code.** The runner and the kernels of every arm, the MALA ones included, are committed
to the repository before the first registered seed runs; a dated note below this text
names that commit, so the order of writing can be checked.

## 3. Exactness before any run

Every new coupled step (J2, J3, J5, CF, MJ5, MCF; CF.2 differs from CF only in the
mixture weight and is checked too) passes `telescopic.testing.check_coupling` before any of
its records counts: each kernel tuned on `examples.gaussian(d)` at seed 9700, d = 2 and
10, HMC arms at `n_steps` 2 and 5 and default, from a near pair (the origin and a point
2e-4 away along each axis) and a far pair (the origin and (1, ..., 1)), 20,000 pairs,
check seed 9701. Pass: `passed()` at its default level 1e-3 for the near pair; for the
far pair the same without the condition that pairs met (as `i137_check_coupling.py`). A
failed check is repeated once with 80,000 pairs at check seed 9702 and counts as a failure
only if it fails again (about 70 checks of 4 p-values each at 1e-3 give a correct kernel
set a chance failure with probability up to about 0.2). A kernel that fails the repeat is
fixed or dropped before the runs, and the failure is reported.

## 4. Targets and cells

All runs: `CoupledSampler(log_prob, initial, grad=..., vectorize=True, kernel=..., rng=seed)`.
**Pilot** means `tune()`, then `meeting_times(100)` at the package's cap of 100,000
iterations, then `choose_k`. **Full run** means `tune()` and the pilot as above, then,
only if no pilot pair is unmet at the cap and `choose_k` gives `k` at most 20,000,
`run(100, max_iter=10**6)` with that `k`, lag `k` and `m = 10k` (the pilot is not
repeated). Otherwise the record holds the pilot only and says why. If any replicate reaches
10^6 iterations unmet, the record's replicates are discarded (a censored replicate would
bias the estimates) and the record counts as one without replicates. A seed without
replicates in a full-run cell counts, for K, as a seed where `k` could not be chosen
(section 7). One commit for every record.

| Part | Cell | Start | Covariance | `n_steps` | Run | Arms | Seeds |
|---|---|---|---|---|---|---|---|
| A | `examples.gaussian(2)` and `(10)` | its N(0, 4I) | tuned | default, 1, 2, 3, 5, 10 | pilot | P, G, J2, J3, J5, CF; P.2, CF.2 at the default only | 30; P 5 in the three resonant cells (d = 2, n = 2; d = 10, n = 3 and 10) |
| A' | `examples.gaussian(5)` (held out) | its N(0, 4I) | tuned | 3, 4, 6 | pilot | P, G, J2, J3, J5 | 30; P 5 but where G copies it |
| B | N(0, diag(s^2)), d = 10, `s = np.geomspace(1, 0.25, 10)` | N(0, 4I) | given `np.eye(10)` | default, 3, 10 | pilot | P, G, J2, J3, J5, CF | 30 |
| B' | N(0, diag(s^2)), d = 8, `s = np.geomspace(1, 0.2, 8)` (held out) | N(0, 4I) | given `np.eye(8)` | default | pilot | P, G, J2, J3, J5 | 30 |
| C1 | `examples.banana()` | exact draws | tuned | 2 | full | P, G, J2, J3, J5, CF | 20 |
| C2 | `examples.banana()` | exact draws | tuned | default | full | P, G, J2, J3, J5, CF | 10 |
| C3 | `examples.mixture()` | its N(10, 10^2) | tuned | 2 | full | P, G, J2, J3, J5, CF | 20 |
| C4 | `examples.mixture()` | its N(10, 10^2) | tuned | default | full | P, G, J2, J3, J5, CF | 20 |
| C5 | logistic regression, below | N(fit, 100 H^-1) | tuned | default | full; P.2, CF.2 pilot | P, G, J2, J3, J5, CF, P.2, CF.2 | 20 |
| D1 | rotated Gaussian, d = 20, condition 10^4, below | N(0, 4I) | tuned | default | pilot | P, G, J2, J3, J5, CF, P.2, CF.2 | 10 |
| D2 | banana with every coordinate curved, d = 20, below | exact draws | tuned | default | full | P, G, J5 (5 seeds), CF (10) | 5 or 10 |
| D3 | `examples.banana()` | its U[-5, 5]^2 | tuned | default | full | P, G, J5, CF | P 3, J5 5, CF 10 |
| E1 | `examples.banana()` | U[-5, 5]^2 | tuned | | full | MP, MJ5, MCF | MP 3, others 5 |
| E2 | logistic regression | N(fit, 100 H^-1) | tuned | | full | MP, MJ5, MCF | 10 |
| E3 | `examples.gaussian(10)` | N(0, 4I) | tuned | | full | MP, MJ5, MCF | 10 |
| E4 | `examples.banana()` | exact draws | tuned | | full | MP, MJ5, MCF | 10 |
| F | `examples.funnel(10)` | its default | tuned | default | pilot at a cap of 10,000, reported only | P, J5, CF | 3 |
| T | the guard's tunings, below | as the cell | as the cell | as the cell | tuning only | G | 300 per cell |

- **Tuning-only test of G (T):** G's tuning, with no pilot, at 300 new seeds of each of the
  cells where P resonates: A's d = 2, n = 2 and d = 10, n = 3 and 10; B's three cells;
  A''s `n_steps` 3 and 6; B'. Each record holds the exact largest factor
  $\lvert\cos n\theta\rvert$ after the guard.
- **Logistic regression** (C5, E2): the first design of `m10_mala_logistic.py`: 200
  outcomes, $x_i = (1, z_i)$, $z_i \sim N(0, I_5)$, coefficients (-0.5, 1, -1, 0.5, 0, 2),
  data from `np.random.default_rng(1)`, prior $N(0, 5^2 I_6)$, start
  $N(\hat\theta, 100 H^{-1})$ with BFGS's fit and inverse-Hessian approximation from 0.
  It is the log-concave target where the automatic MALA fails (19 of 100 pilot pairs
  unmet at 3,000, `m10_mala_logistic.out`), the logistic target #91 and #89 name, and the
  one where #91's evidence put CF's median cost closest to the threshold (+17%, with CF at
  a meeting weight of 0.2). Its reference values are computed independently, by
  self-normalized importance sampling from a multivariate t with 5 degrees of freedom
  located at the posterior mode with scale matrix the inverse of the exact Hessian there,
  10^6 draws from `np.random.default_rng(9799)`, with the delta-method standard error; if
  the effective sample size is below 10^5, the draws are increased tenfold.
- **Rotated Gaussian** (D1): $\Sigma = Q\,\mathrm{diag}(\lambda)\,Q^\top$, $\lambda$
  geometric from 1 to 10^-4, $Q$ the Q factor of a 20 x 20 standard normal matrix from
  `np.random.default_rng(91)`. #91 planned d = 100; see section 13.
- **Banana in d = 20** (D2): $x_0 \sim N(1, 1/2)$, $x_j \mid x_0 \sim N(x_0^2, 1/20)$,
  $j = 1, \dots, 19$, $\log\pi = -(1 - x_0)^2 - 10\sum_j (x_j - x_0^2)^2$, started from
  exact draws, as #91's test lists it.
- **Held-out cells** (A', B'): chosen after the guard's rule was fixed, by scanning P's
  tunings alone at probe seeds 950 to 955 for cells that resonate (`scan_heldout.out`);
  no rule or constant of any arm was set on them.
- **Test functions and exact values:** bananas $(x_0, x_1)$, (1, 1.5); mixture
  $(x, x^2)$, (0, 17); logistic $\beta_0, \dots, \beta_5$, the reference; funnel
  $(v, x_1)$, (0, 0); `examples.gaussian(10)` $(x_0, x_0^2)$, (0, 1). The inefficiency
  is that of the first function.

## 5. Seeds

Each seed is the `rng` of the sampler; every arm of a cell uses the same seeds, so P and
G share their tuning bit for bit. Where fewer seeds are listed for an arm, it takes the
first ones of the block.

| Part | Seeds |
|---|---|
| A, A' | 9100 to 9129; 9130 to 9159 |
| B, B' | 9200 to 9229; 9230 to 9259 |
| C1, C2, C3, C4, C5 | 9300 to 9319, 9320 to 9329, 9340 to 9359, 9360 to 9379, 9380 to 9399 |
| D1, D2, D3 | 9400 to 9409, 9420 to 9429, 9440 to 9449 |
| E1, E2, E3, E4 | 9500 to 9504, 9520 to 9529, 9540 to 9549, 9560 to 9569 |
| F | 9600 to 9602 |
| T, cell $c$ = 0 to 8 in the order of section 4 | 20000 + 1000 $c$ + (0 to 299) |
| Confirmation of a Z failure (section 7) | 9800 onward, in the cell's order of seeds |
| check_coupling; V's bootstrap; logistic reference | tuning 9700, checks 9701, repeats 9702; 9798; 9799 |

Where G keeps the step at a seed for which P has no record (resonant cells of A, A' and B'
beyond P's 5 seeds, D3 beyond P's 3), P's record at that seed is computed and used for
both. In B and B', where the guard predicts a fallback at nearly every seed, P therefore
runs at all 30.

**How many, and why.** The numbers come from the spreads between seeds in i137's stores
(`spreads.out`, `ineffpower.out`) and from the review's recomputation (`p3_power.out`).

- *Pilot cells, 30 seeds.* A failing seed at the package's cap leaves between 12 and 96 of
  its 100 pairs unmet in the six such seeds on record (31, `i137_baseline.out`; 14 and 96,
  `i137_default_cap.out`; 54, 23, 12, `i137_reproduce_unequal.out`), and at a cap of
  10,000 as few as 4 (`p2_claims.out`, part d). So U (section 7) fails an arm whose
  tunings fail at a rate $p$ with probability 0.14, 0.28, 0.59, 0.86 at $p$ = 0.01, 0.02,
  0.05, 0.10, and some seed raises while U holds with probability 0.12, 0.18, 0.20, 0.10
  (`p3_power.out`). Criterion K (at most 1 of 30 seeds where `k` cannot be chosen) fails
  with probability 0.036, 0.12, 0.45, 0.82 at the same rates (`kpower.out`). P's failing
  fractions at the package's cap were 0.09 to 0.49 (#137's validation, 100 seeds), so 5
  seeds of P in the resonant cells are a reference, not a test.
- *G's tuning test, 300 tunings.* At most 3 of 300 above 0.9998 fails with probability
  0.013, 0.35 and 0.85 at true rates 0.003, 0.01 and 0.02, and 1.000 at 2 of 30
  (`kpower.out`).
- *Median cost (M).* The per-seed log median meeting time has a standard deviation
  between seeds of 0.10 (P) and 0.08 (J5) on the banana with `n_steps=2`; on the nine
  cells of A where P meets, 0.06 to 0.08 for P at d = 2 with default and 1 and at d = 10
  with default, 1 and 2, and 0.16 to 0.69 at the four others (d = 2 with 3, 5, 10;
  d = 10 with 5), where M's standard error is 0.03 to 0.13 but the ratios are far below 1
  (`p2_claims.out`, part c); up to 0.38 for J5 on the mixture, whose medians are bimodal.
  With 20 seeds per arm M fails an arm on the banana with probability below 0.001, 0.001,
  0.5 and 0.997 at true ratios 1.0, 1.1, 1.2 and 1.3, and 0.84 at 1.3 if both spreads,
  each estimated from 5 seeds, lie at the upper end of their 95% intervals
  (`p3_power.out`); with the mixture's spreads, 0.04, 0.21, 0.5, 0.78 and 0.93 at 1.0 to
  1.4. The old test's cost arm on the banana failed an arm only above 3.19 times the
  package's mean inefficiency (`i137_nongaussian.out`). On D1 no spread between seeds is
  known; M's power there is not stated.
- *Cost per effective replicate (V).* The per-seed inefficiency of the banana is
  heavy-tailed (log standard deviation 2.6 for P with `n_steps=2`, set by one seed of
  five; 0.9 robustly). At 20 seeds V fails an arm 4 times costlier than P with a
  probability between 0.3 and 1.0 depending on which spread is true (0.30 with lognormal
  seeds at 2.6 for both arms, 0.48 with the arm at J5's 0.39, 1.0 at the robust 0.9,
  `p3_power.out`); at a ratio of 1.3 it fails rarely (0.04 to 0.08). On the mixture it
  fails an arm at 1.3 with probability 0.62 to 0.97 and at 1.0 with probability below
  0.001 (`ineffpower.out`, `p3_power.out`). **On C2** (10 seeds, P's log spread 3.8) V
  fails a 4-fold arm with probability 0.27: it cannot fail there in practice, and M
  carries C2's cost; with the spreads of the banana with `n_steps=5` (0.28 and 0.11), M
  fails an arm on C2 at ratios 1.0, 1.3 and 1.5 with probability 0.03, 0.80 and 0.99.
- *Parts D, E, F:* sized by cost; D2 and D3 test U on 500 to 1,000 pairs of CF, so that
  one unmet pair in 100 is distinguishable from none.

## 6. Measurements and records

Each record is one JSON line in a store per part (in a folder git ignores, as
`notes/experiments/_rows/`), keyed by part, cell, arm and seed, carrying the setup line
of `i85_setup.setup()` (commit, processor, thread variables, Python, NumPy and SciPy
versions). Outputs are printed from the stores, and raise if the records mix commits
(as `i137_common.header`). A record holds:

- the nominal step, `n_steps`, jitter width, target acceptance, meeting weight; for G
  its $\varepsilon_0$, action and $\min_i r$ before and after; for the Gaussians of
  parts A, A', B, B', D1 and T the exact whitened frequencies and the largest factor
  $\lvert\cos n\theta\rvert$ (for jittered arms its geometric mean over the jitter, as
  `i137_common.jitter_factor`), and for B and B' the pilot's predicted frequencies;
- gradient and log-density evaluations of tuning, of the pilot and of the replicates,
  as the sampler's `Target` counts them;
- the 100 pilot meeting times (inf where unmet at the cap), and from them the unmet
  count, the quantiles 0.5, 0.9, 0.99, `k` or the error of `choose_k`, and
  $g$ = pilot gradient evaluations / sum of the pilot meeting times (unmet pairs at the
  cap), the gradients per pair-iteration;
- for a full run: whether replicates ran and, if not, why; `k`, `m`, replicates met,
  each function's estimate, standard error and z-score against its exact or reference
  value, and the inefficiency $\mathrm{SE}^2 \times$ (gradient evaluations of the
  replicates);
- the record's processor seconds; `same_as` where it is a copy of P's record.

The meeting cost of pair $i$ is $c_i = \tau_i g$; a pair is **unmet at the gradient
budget** when $c_i > 2 \times 10^5$ or $\tau_i$ is infinite.

## 7. Criteria

Each criterion is stated for one arm and one cell, against P on the same cell unless it
says otherwise.

- **U, unmet pairs.** The pooled fraction of pilot pairs unmet is at most 1% (at most 30
  of 3,000 in a cell of 30 seeds; 20 of 2,000; 10 of 1,000; 5 of 500), judged twice: at
  the package's cap of 100,000 iterations, and at the gradient budget of
  $2 \times 10^5$ per pair (#91's), which compares arms whose iterations cost different
  numbers of gradients. The threshold is Josh's (#89). An arm holds U when it holds it at
  both, with one exception: on a cell where P itself fails U at the gradient budget while
  holding it at the cap (slow meeting, not failed meeting; D1 and B's `n_steps` 3 and 10
  are the likely cases), U at the gradient budget is reported for every arm there and
  U at the cap alone is judged, since the budget there measures the target's slowness,
  not the arm's.
- **K, seeds that raise.** On every pilot cell, `k` is chosen at every seed but at most 1
  of 30 (at most 1 of the cell's seeds when there are fewer). One seed in 30 that raises
  #137's error is what a user meets in about 3% of runs. In a full-run cell, a seed
  without replicates (section 4: a pilot pair unmet at the cap, `k` above 20,000, or a
  replicate unmet at 10^6 iterations) counts as such a seed, so that an arm's slowest
  seeds cannot leave V and Z unseen by any criterion.
- **G's tuning test.** On each cell of T, at most 3 of the 300 tunings (1%) have an exact
  largest factor above 0.9998 after the guard: the threshold that separated failing from
  passing seeds at the package's cap in #137's validation (failing from 0.99986, passing
  up to 0.99982).
- **M, median cost.** On every cell outside T where P leaves no pair unmet at the cap at
  any seed: the ratio
  $\rho_M = \exp(\overline{\ln m_A} - \overline{\ln m_P})$, with $m$ a seed's median
  meeting cost $\mathrm{median}_i(\tau_i)\,g$ and the bar a mean over seeds. The arm fails
  when $\rho_M > 1.20$ (Josh's threshold, #89). For P.2 and CF.2, M compares CF.2 with
  P.2. M fails on a point estimate, with the burden of proof on neither side: the
  threshold is a ruling about the cost a default may add, the ratio is measured with a
  standard error of 0.02 to 0.04 on most cells where it applies (section 5), and a pass
  or a failure is equally an outcome.
- **L, the tail, on the nine cells of T** (A's d = 2, n = 2 and d = 10, n = 3 and 10; B's
  three cells; A''s `n_steps` 3 and 6; B'), whatever P shows there; also on any other cell
  where P has a pair unmet at the cap, where M cannot apply. Let $T_A$
  be the geometric mean over seeds of $k\,g$ (the pilot's 99% quantile in gradients) for
  the seeds where `k` was chosen. An arm that holds K fails L when $T_A$ exceeds 3 times
  the smallest $T$ among the arms that hold K on that cell. This catches the resonance
  that shows as a `k` in the tens of thousands with no pair unmet
  (`HMC(n_steps=3)`, d = 10: `k` from 12,377 to 52,065 at 8 of 10 seeds,
  `i137_default_cap.out`).
- **V, cost per effective replicate** (full runs of part C). The ratio $\rho_V$ of the
  medians over seeds of the per-seed inefficiency, arm over P, over the seeds where both
  arms ran replicates; the arm fails when the 5% point of its bootstrap distribution
  (seeds resampled within each arm, 10,000 resamples, seed 9798) exceeds 1.20: the data
  then show a rise above 20% at one-sided 95%. Unlike M, V puts the burden of proof on
  the claim of harm, because the banana's inefficiency is heavy-tailed: as a point
  estimate against 1.20 it would fail an arm that costs the same as P with probability
  about 0.4 on the banana (normal approximation with the log spread 2.6 and 20 seeds). V is void on a cell where either arm has replicates at fewer
  than half the cell's seeds; the record gives the number of seeds used.
- **Z, unbiasedness** (every full run of every arm, over the seeds with replicates, whose
  number is reported). For each function, the pooled estimate is the mean over seeds of
  the seeds' estimates, its standard error the root of the sum of their squared standard
  errors divided by the number of seeds, and $z$ = (pooled estimate - exact value) / that
  error (for the logistic regression the reference's standard error is added in
  quadrature). The level is $\lvert z\rvert > 3.5$, a Bonferroni level for the about 140
  z-scores of the registered full runs (2 to 6 per cell and arm); under normality the
  family-wise false rate is about 0.06. It replaces #91's level of 3, which over this
  many scores would refuse a correct arm with probability about 0.3. A score beyond 3.5 is
  confirmed before it refuses: the arm runs the same cell again at as many fresh seeds
  from 9800, and fails Z only if the fresh pooled z of the same function has the same
  sign and exceeds 2 in absolute value. Where P's own $\lvert z\rvert$ exceeds 3.5 on a
  cell (confirmed the same way), each arm on that cell is judged instead by the z-score
  of the difference between its pooled estimate and P's, whose error is the root of the
  sum of the two squared errors, at the same level and with the same confirmation.
- **Funnel (part F):** no criterion; reported.

The bootstrap of V uses 10,000 resamples; its power in section 5 was simulated with
2,000 per experiment and 400 experiments per entry.

**Which criteria apply.**

| Arm | U and K on | G's test | L on | M on | V on | Z on |
|---|---|---|---|---|---|---|
| G | A, A', B, B' (purpose), C, D1; D2, D3 reported | T | the nine cells of T | every other cell of A, A', C, D1 where P has no pair unmet at the cap | C | every full run |
| J2, J3, J5 | A, A', B, B' (purpose), C, D1; D2, D3 reported | | as G | as G | C | every full run |
| CF | every non-funnel cell it runs (A, B, C, D) | | the six cells of T in A and B | every other cell of A, C, D1 where P has no pair unmet at the cap | C | every full run |
| CF.2 against P.2 | | | | A default, C5, D1 | | |
| MJ5, MCF | E1, E2 (purpose), E3, E4 | | | E3, E4 (and E2 if MP has no pair unmet at the cap) | | every full run |

G, J2, J3 and J5 are remedies for a resonance: on D2 and D3, whose failures are of
curvature, U and K are reported, not judged; Z is judged there as everywhere.

## 8. Decision rules

For each arm, in this order:

1. **Refused** if it fails Z anywhere, or fails `check_coupling`, or fails U or K on any
   cell of its purpose (A, A', B and B' for the jitters; A and A' for G, whose cells of B
   and B' decide its scope below; D2 and D3 for CF; E1 and E2 for the MALA arms). G is
   also refused if it fails its tuning test on any cell of T in A or A', even with no seed
   that raised: T alone refuses it, unless the check below makes T reported only.
   **The check of T's proxy.** Over the registered pilot records of P and G on the nine
   cells of T (unjittered kernels, whose exact largest factor each record holds), the
   runner tabulates "factor above 0.9998" against "`k` not chosen". T is judged only if at
   least 5 records lie above 0.9998 and `k` could not be chosen at at least half of them;
   otherwise the link from factor to failure that #137's validation showed on
   `gaussian(2)` and `(10)` is not confirmed here, and T is reported, not judged, for every
   cell.
2. **Default candidate** if it passes U, K, L, M and V on every cell where the table above
   applies them. A default candidate becomes the default only after Josh has seen the
   result (#89).
3. **Opt-in** otherwise: it holds U and K on the cells of its purpose and fails some other
   criterion of rule 2.
4. **Undecided** if a cell on which a criterion applies to it was cut (section 11): its
   verdict waits until that cell runs. Only parts E and F can be cut, so only the MALA
   arms can be undecided.

Between the remedies of #137:

- Among the jitter widths that are default candidates, the widest is recommended: it
  sweeps the most phase, which is the margin against resonances this test does not
  contain, and its cost is already within 20%. If none is a default candidate, the
  narrowest that holds U and K on A, A', B and B' is the opt-in recommended.
- G passes its purpose with U, K and L on A and A' and its tuning test on the cells of T
  in A and A'. Its scope for a given covariance is decided cell by cell on B and B': with a
  given `n_steps` (B's 3 and 10) it covers that case if it holds U, K, L and the tuning
  test there; with the default `n_steps` (B's default and B') likewise. G changes no
  kernel, so where it passes it is preferred to a jitter for that case.

**What refutes CF as the default at the package's meeting step** (the result that
answers #89's question 1 with no, for CF as #91 states it with the package's meeting
weight of 0.05): any non-funnel cell where it fails U or K; or $\rho_M > 1.20$ on any cell
where M applies; or L or V failing on any cell where they apply; or a failure of Z.
**What refutes it as an opt-in for curvature:** U or K failing on D2 or D3, or a failure
of Z. **What the arms at a meeting weight of 0.2 say:** if CF.2 passes M against P.2 on
all four of its cells, the cost CF shows at 0.05 comes from the meeting weight and not
from the tuning, and a full test of the pair at 0.2 is the next experiment; if CF.2 fails
M against P.2 on any of them, CF's cost is its own at both weights. They decide no default
here.

## 9. Predictions

Each prediction is an inequality, scored held or failed. The predictions for P are
references, not tests: P's intervals cover nearly every outcome at #137's rates. The
reasoning: (i) the factor $\lvert\cos n\theta\rvert$ at the exact frequencies, before and
after G, over the 100 stored or probe tunings of each Gaussian cell (`post.out`,
`heldout.out`); (ii) the geometric-mean factor over the jitter at P's tuned steps
(`jpred.out`, `heldout.out`); (iii) a meeting-time model,
$\tau \approx 1/\gamma + 6/(-\ln f)$, $f$ the slowest mode's factor, whose constant 6
was set so that it gives P's median at `examples.gaussian(2)` within 10% (28.2 against
25.8; `model.out`); (iv) i137's measured outcomes for J5 (remedy c there) and for the old
guard's steps, which G's equal where it moves (checked on the 30 stored pilot seeds of
the three resonant cells: identical steps at 89 of 90, the 90th kept by G; confirmed by
the review's independent implementation of the guard from this text, which agrees with
`guard.py` on all 990 stored and probe tunings, `p1_guard.out`); (v) for the failure
counts, the rate of tunings above 0.9998, the threshold at the package's cap.

**Part A** (median meeting time: median over seeds of the seed's median).

| Cell | P | G | J2 | J3 | J5 | CF |
|---|---|---|---|---|---|---|
| d = 2, n = 2 | `k` fails at 0 to 3 of 5 | moves at 30 of 30; U, K, L hold; tuning test holds; median 20 to 40 | U, K, L hold; median 40 to 100 | U, K, L hold; median 25 to 50 | U, K, L hold; median 25 to 40 | U, K, L hold |
| d = 10, n = 3 | `k` fails at 1 to 4 of 5 | moves at 30; U, K, L hold; tuning test holds; median 35 to 60 | U, K hold; L not predicted; median 60 to 140 | U, K, L hold; median 35 to 75 | U, K, L hold; median 30 to 45 | U, K, L hold |
| d = 10, n = 10 | `k` fails at 0 to 2 of 5 | moves at 26 or more; tuning test fails | U, K, L hold; median 25 to 60 | U, K, L hold; median 25 to 60 | U, K, L hold; median 35 to 55 | U, K, L hold |
| other 9 cells | `k` chosen at 30 of 30 | keeps at every seed of d = 2 with n = default, 1, 3 and of d = 10 with n = default, 1, 2, 5; moves at 0 to 4 of 30 with d = 2, n = 5 and 0 to 3 with n = 10 | U, K hold; $\rho_M \le 1.10$ | U, K hold; $\rho_M \le 1.15$ | U, K hold; $\rho_M \le 1.20$ at 8 or 9 of the 9 cells | U, K hold; $\rho_M \ge 2$ at the default `n_steps`, d = 2 and 10 |

At the default `n_steps`: CF.2 against P.2, $\rho_M \ge 1.8$ at d = 2 and d = 10 (model:
2.5 and 3.5); CF.2 against P, between 1.0 and 1.8 (model: 1.18 and 1.52, `model.out`).

Reasons. P: #137's validation at the package's cap gave failing fractions 0.21, 0.49 and
0.09. G: over 100 tunings each, after the guard the exact largest factor exceeds 0.9998
at 0 (d = 2, n = 2), 0 (d = 10, n = 3) and 2 (d = 10, n = 10) of 100, against 23, 42 and 13
for P (`post.out`); at d = 10, n = 10 the tunings it misses are those it keeps because the
predicted phase at $w = 1$ is at least 0.451 from every $j\pi$, while an exact mode,
between 0.71 and 1.29, is on a resonance. At a rate of 0.02, G's tuning test fails with
probability 0.85, K with 0.12, U with 0.28. Its moves elsewhere: 5 and 2 of 100 tunings at
d = 2, n = 5 and 10 (`evalguard.out`). Jitters: the worst geometric-mean factor at P's
steps is at most 0.93 (J2), 0.84 (J3), 0.61 (J5) in every cell (`jpred.out`), which the
model turns into the medians above; J5's are i137's 31.2, 37.0, 42.8 within 30%. L is not predicted for J2 on the cells
where its worst factor exceeds 0.9 (A's d = 10, n = 3; B's three cells; A''s 3; B'): the
model puts its tail there at 2.8 to 3.0 times J5's, on L's threshold of 3, so the outcome
is about even; J2 matters only if J3 and J5 are not default candidates.
$\rho_M$ for J5: i137's ratios were 1.06, 1.06, 1.15, 1.09, 1.17 at the cells of default
and small `n_steps` and below 1 at the others; if those are the true ratios, J5 passes M
on all 9 cells with probability 0.90 (`p2_claims.out`, part c); J3 and J2 are predicted to
scale the excess as $j^2$. CF: dual averaging toward 0.99 returned $\varepsilon$ = 0.31
and 0.21 with 4 and 5 leapfrog steps at d = 2 and 10, against P's 1.13 and 0.79 with 1 and
2 (`probe_tune.out`); the model gives $\rho_M$ = 3.4 and 3.1, because the wait for the
meeting step ($1/\gamma$ = 20 iterations) is paid at CF's longer trajectories.

**Part A'** (held out; `heldout.out`, 100 probe tunings per cell).

| Cell | P | G | J2 | J3 | J5 |
|---|---|---|---|---|---|
| `n_steps=3` | references: 3 of 100 tunings above 0.9998 | keeps at about 60% of seeds; 2 of 100 above 0.9998 after it; tuning test fails | U, K hold; L not predicted | U, K, L hold | U, K, L hold |
| `n_steps=4` | `k` chosen at 30 of 30 | keeps at every seed | U, K hold; $\rho_M \le 1.15$ | U, K hold; $\rho_M \le 1.15$ | U, K hold |
| `n_steps=6` | references: 15 of 100 above 0.9998 | keeps at about 70% of seeds; 11 of 100 above 0.9998 after it; K and the tuning test fail | U, K, L hold | U, K, L hold | U, K, L hold |

Reasons: on `gaussian(5)` the exact frequencies of the tuned covariance run from 0.84 to
1.18, so with 3 or 6 steps an exact mode is on a resonance while the predicted phase at
$w = 1$ is not within 0.451 of one; the jitters' worst factors are at most 0.92 (J2),
0.82 (J3), 0.59 (J5).

**Part B.**

| `n_steps` | P | G | J2 | J3 | J5 | CF |
|---|---|---|---|---|---|---|
| default | `k` fails at 0 to 8 of 30 (reference) | falls back at 28 or more of 30, so its records are P's; tuning test fails | U, K hold; L not predicted; median 60 to 140 | U, K, L hold; median 35 to 75 | U, K, L hold; median 35 to 55 | U, K, L hold |
| 3 | `k` at 30 of 30; median 400 to 1,100 | moves at 30; U, K, L, tuning test hold; median 60 to 110 | U, K hold; L not predicted | U, K, L hold | U, K, L hold; median 30 to 50 | U, K, L hold |
| 10 | `k` at 30 of 30; median 800 to 2,500 | moves at 30; U, K, L, tuning test hold; median 55 to 100 | U, K hold; L not predicted | U, K, L hold | U, K, L hold; median 35 to 55 | U, K, L hold |

Reasons. On seeds 0 to 29 retuned at `4cb3acd`, P's tunings have an exact factor above
0.999 at 7 and above 0.9998 at 2 (default); i137 had 3 of 10 failing at the cap. G finds
no step for its band at any default tuning and keeps the step (30 of 30); with a given
`n_steps` it moves every tuning to an exact factor of at most 0.908 (`evalunequal.out`,
`post.out`). The jitters' worst factors are 0.93, 0.84, 0.61 (`jpred.out`). i137's medians:
P 720 and 1,381; old guard 79 and 72; J5 43.0, 39.2, 44.8.

**Part B'** (held out; 30 probe tunings): P has 6 of 30 tunings above 0.9998, so `k`
fails at 2 to 12 of 30 seeds (reference); G falls back at every seed (its records are
P's) and fails K and its tuning test; J2, J3 and J5 hold U and K, and J3 and J5 hold L
(worst factors 0.92, 0.84, 0.56); L is not predicted for J2.

**Part C** (G keeps the step at every seed of C: the predicted phase is 1.7 or more from
$\pi$ on every stored banana and mixture tuning (`phases.out`) and 1.26 or more on the
probe tunings of the default `n_steps` of the banana, the mixture and the logistic
regression, so its records are P's and it passes by identity).

| Cell | J2 | J3 | J5 | CF |
|---|---|---|---|---|
| C1 banana, n = 2 | $\rho_M \le 1.10$; V holds | $\rho_M \le 1.12$; V holds | $\rho_M$ in 1.08 to 1.30 (i137: 1.18), M fails with probability about 0.3; V holds | $\rho_M > 1.20$ |
| C2 banana, default | U, K hold | U, K hold | U, K hold; $\rho_M < 1.20$ | $\rho_M > 1.20$ |
| C3 mixture, n = 2 | $\rho_M < 1$; V holds | $\rho_M < 1$; V holds | $\rho_M < 1$; $\rho_V$ in 1.15 to 1.45 (i137: 1.29), V fails with probability about 0.6 | V fails ($\rho_V \ge 1.5$) |
| C4 mixture, default | V holds | V holds | $\rho_V > 1.1$ | V fails |
| C5 logistic | $\rho_M \le 1.10$ | $\rho_M \le 1.15$ | $\rho_M \le 1.25$ | $\rho_M \ge 1.8$; CF.2 against P.2 $\ge 1.8$ |

Every arm passes Z on every cell. P's medians: banana n = 2 in 60 to 85 (i137 70.4),
mixture n = 2 in 190 to 270 (i137 228.4). Reasons for CF: 3 to 4 times the gradients per
trajectory at every probe tuning (banana 11 to 13 steps against 4; mixture 8 against 3;
logistic 5 against 2, `probe_tune.out`), with a meeting time that is mostly the wait for
the meeting step (1/0.05 = 20 iterations), which a longer trajectory does not shorten,
and on the mixture a mode-switching rate per iteration that the shorter step does not
raise.

**Part D.**

- D1 (rotated, d = 20): the tuned covariance does not whiten the target: P's whitened
  frequencies at probe seeds 900 and 901 run from 0.044 and 0.073 to 2.6 and 1.4
  (`probe_rot.out`). Every arm holds U and K; P's median meeting time is between 1,000
  and 10,000 iterations, set by the slowest mode (the model's $6/(-\ln\cos(w_{\min}T))$,
  with the probes' $w_{\min}$ and $T = n\varepsilon$ of 1.16 and 1.36, gives 4,500 and
  1,200). CF's $\rho_M \ge 2$; CF.2 against P.2 $\ge 1.8$. No prediction is made for the
  jitters on M at D1: the model does not resolve their cost there to within 20%, and no
  spread between seeds is known. G keeps the step (it reads $w = 1$), so its records are
  P's.
- D2 (banana, d = 20): CF holds U and K. No prediction is made for P and J5 on U: no
  measurement of either exists on a target with curvature in many directions.
- D3 (banana from U[-5, 5]^2): P leaves at least 50 of 100 pairs unmet at each seed (88
  of 100 in `m9`, 475 of 500 in #91's evidence); J5 fails U (#91's screen: jitter 0.5 at
  acceptance 0.9 left 25 of 100); CF holds U and K, with `k` below 1,500 at every seed
  (#91: 0 of 500 unmet, 99% quantile of meeting cost 8,073 gradients, measured at a meeting
  weight of 0.2; at 0.05 the model adds about 15 iterations of wait per meeting, which
  this bound allows for); G's records are P's.
- CF passes Z on D2 and D3.

**Part E (MALA).** MP fails U on E1 and E2 (`benchmarks/results/acb12f9.json`;
`m10_mala_logistic.out`). MJ5 holds U on E2 and fails it on E1. MCF holds U on E2; no
prediction for MCF on E1 (its step there is 0.054 against MP's 0.32, about 680
iterations per e-fold of the gap, `p4_mala.out`). MCF's $\rho_M \ge 2$ on E3 (step 0.34
against 1.10, so the drift contracts the gap about 10 times less per step,
`p4_mala.out`). MJ5's $\rho_M \le 1.25$ on E3. Reasons: on E2 the failing chains start
where the largest eigenvalue of $\Sigma H$ exceeds $4/\varepsilon^2$ at the tuned step
1.24, and steps of 0.7 met every pair (`m10_mala_logistic.out`); J5 draws a step below
0.75 times the nominal one with probability 0.25 per iteration.

**Part F.** Every arm leaves more than 1% of its pairs unmet at a cap of 10,000; CF's
tuning returns 40 or more leapfrog steps at some seed (42 to 180 at probe seeds,
`probe_tune.out`).

**The verdicts this implies.** CF: opt-in, not a default at the package's meeting step
(it fails M on the Gaussians and the logistic regression), and CF.2 fails M against P.2,
so its cost is its own. G: refused for a given `n_steps` (its tuning test fails at
d = 10, n = 10 and on A'), in scope for a given covariance with a given `n_steps` only
(B's 3 and 10). J3: a default candidate if it also passes M on D1, for which no
prediction is made; J2 likewise, and if it passes L where L is not predicted for it; J5: opt-in or default candidate depending on M on the banana and V on
the mixture.

## 10. What was fixed after seeing which data

| Choice | Set from | Held out |
|---|---|---|
| G's band 1.3 | i137's d = 10 tunings at seeds 0 to 99 (inherited from i137) | every registered seed; A', B', C, D |
| G's bound $\delta = \arccos 0.9$ | the old guard's bound, a design value; kept after comparing $\delta$ = 0.16, 0.20, 0.30, 0.451, 0.60, 0.75 on the stored d = 2 and the probe d = 10 tunings at seeds 0 to 99, scored with exact frequencies and a threshold of 0.999 learned from i137 (`evalguard.out`) | A', B' and every registered seed |
| G's trigger without a band, and no $j = 0$ guard | the review of #148 (the band and the $j = 0$ end moved steps on no resonance) and the same comparison | A', B' and every registered seed |
| G's fallback: keep the step | the simplest rule; it replaced, after the review, a fallback chosen on B's probe tunings (section 14, I2) | A', B' and every registered seed |
| The held-out cells A', B' | P's tunings alone at probe seeds 950 to 955 (`scan_heldout.out`), after every rule of G and the jitters was fixed | their registered seeds |
| The rotated Gaussian's d = 20 | probe tunings at d = 10, 20, 50 (seeds 900, 901) and 100 (seed 900) (`probe_rot.out`, `probe100.out`) | its registered seeds |
| Seeds per cell | the spreads of i137's stores (section 5) | |
| Cost estimates | the stores' seconds and the probe tunings (section 11) | |

New seeds of the cells of A and B are a second sample of the cells G's rule was chosen
on, not a held-out test; A' and B' are the held-out test, for G and the jitters alike.

The jitter widths (0.2, 0.3, 0.5) are the ruling's; CF's 0.99 and 0.5 are #91's; the
thresholds 1% and 20% are Josh's (#89); 3.5 is a Bonferroni level, which departs from
#91's 3 (section 7); K's one seed in 30, L's factor 3 and the bounds of 20,000 on `k`
and 10^6 iterations on a replicate were set by this registration after the review,
without data of any arm. G's 1% of tunings was set after the review too, but with G's
probe tunings after the guard already known (2 of 100 above 0.9998 at d = 10,
`n_steps=10`; 2 and 11 of 100 on A''s 3 and 6, `post.out`, `heldout.out`), so T's outcome
for G was predictable when it was set: A' is held out from G's rule, not from the
prediction of its result. None was set from meeting times.

**The guard's property on the stored tunings.** The old guard lowered the step at every
seed of the four cells where P gives `k` at every seed (d = 2 with `n_steps` 3, 5, 10;
d = 10 with 5) and raised or changed it at 2 of 5 seeds of the banana with `n_steps=2`,
by its $j = 0$ end and its fallback. Over 100 tunings each, G keeps the step at 100 of 100
(d = 2, n = 3), 95 (n = 5), 98 (n = 10), 100 (d = 10, n = 5), and at every stored banana and
mixture tuning (predicted phase at least 1.7 from $\pi$; `phases.out`). The 7
tunings it moves in the first group are those whose predicted phase at $w = 1$ lies
within 0.451 of $2\pi$ or $4\pi$; on i137's 30 pilot seeds they are 3 seeds whose median
meeting times were 98.5, 211.5 and 394.5, against cell medians of 66.2 and 39.2. No
$\delta$ keeps every tuning whose exact factor is below 0.99 and catches the resonances:
at $\delta$ = 0.16, which keeps all 400, 40 of 100 tunings of d = 10, n = 3 stay above 0.999
after the guard, against 0 at 0.451 (`evalguard.out`). With $w = 1$ predicted, the
phase of an exact mode is uncertain by $n h (1 - h^2/4)^{-1/2}$ times its relative
error, 0.86 radian at d = 2, n = 5 for an error of 0.1, so a guard that reads the
predicted frequencies cannot tell these 7 apart from resonant tunings, and misses
resonant ones in the same way (A').

## 11. Budget

Estimated processor time on one thread, from the stores' seconds (`costs.out`), the probe
tunings, and the review's throughput of the funnel (`p4_mala.out`), with the package's cap
of 100,000:

| Part | Estimate | How |
|---|---|---|
| A | 61 min | P: 119 s for the 270 pilots of the 9 cells without resonance (i137 at cap 10^4, all met, so equal at 10^5) + 95 s of tunings + about 950 s for 15 pilots of the resonant cells (one failing pilot at 10^5 took 43 s at d = 2; d = 10 pilots at 10^4 took up to 20 s, times 10); G: 126 s of tunings + 50 s of moved pilots + about 400 s for P's records where G keeps; J5: 167 s of pilots (i137) + 126 s; J2, J3: 1.6 times J5's pilots each; CF: 4 times J5's pilots + 160 s |
| A: P.2, CF.2 | 2 min | 60 pilots each at about 0.4 and 1.2 s, with tunings |
| A' | 13 min | P where G keeps (about 20 and 40 records at `n_steps` 6 and 3, of which 3 to 5 failing at about 60 s); J2, J3, J5 270 pilots at about 0.6 s; tunings 0.3 s each |
| B | 31 min | P at all 30 seeds of the default (about 960 s, failing seeds run 9 x 10^4 more iterations) + 158 s (`n_steps` 3, 10); G copies P at the default, 40 s of moved pilots and 40 s of tunings; J5 84 s; J2, J3 160 s each; CF 230 s |
| B' | 15 min | P at 30 seeds (about 6 failing at 100 s, 24 slow at 8 s), G copies it; jitters 63 s; tunings 54 s |
| T | 18 min | 2,700 tunings at about 0.4 s |
| C | 87 min | per run at R = 100, about 0.6 times i137's R = 200 seconds: banana n = 2 890 s; banana default 1,120 s (P's records like i137's n = 5, 89 s at R = 200); mixture n = 2 1,050 s; mixture default 1,575 s; logistic 520 s; P.2, CF.2 pilots on C5 about 40 s |
| D | 73 min | D1 940 s (slowest pairs near 10^4 iterations) + 450 s for P.2 and CF.2; D2 1,950 s (P's pilot at up to 150 s if unmet; CF at 10 seeds); D3 1,070 s (P 3 x 120 s, as `acb12f9`'s 119 s) |
| E | 32 min | MP on E1 3 x 60 s, on E2 10 x 49 s (`acb12f9`, `m10`); the rest about 1,200 s |
| F | 15 min | CF up to 10^4 iterations of up to 180 leapfrog steps: at most 75 s per seed (`p4_mala.out`) |
| Total | 5.8 h | tier 1 (A to D and T) 5.0 h |

A full run is bounded by its rule (section 4): `k` at most 20,000 and every replicate at
most 10^6 iterations, after which the record's replicates are discarded.

**Order and cuts.** The runner records each record's seconds and runs in this order:
A, A', B, B', T, C1 to C5, D3, D2, D1 (tier 1), then E (tier 2), then F (tier 3). A cell
is run whole or not at all. Tier 1 is never cut: C2 and C4, the default `n_steps` costed
on the banana and the mixture, are in it, so that a slow failure elsewhere cannot remove
a criterion an arm could fail; if tier 1 alone exceeds 6 hours, it is finished and the
overrun reported. When the sum of recorded seconds reaches 6 hours, no new cell of tiers
2 and 3 starts, and the cells not run are reported as cut. Cutting F loses the funnel,
which #89 asks for, reported separately; cutting E leaves the MALA arms undecided (rule
4) and #91's MALA grid untested.

**Cut to fit, against #91's plan:** one acceptance target and one width for CF instead of
the grid {0.97, 0.99, 0.995} x {0.3, 0.5}; 100 pairs per seed at the package's cap,
judged also at #91's budget of 2 x 10^5 gradients, instead of 500 pairs; 100 replicates
per run instead of 3 runs of 2,000; d = 20 instead of 100 for the ill-conditioned
Gaussian, and no identity-mass version of it; P at 3 to 10 seeds in cells where it is
known to fail and G does not copy it; the meeting weight of 0.2 only on four cells and in
pilots.

## 12. Limits

- **Coverage.** One to twenty dimensions, one funnel, one mixture, no hierarchical
  target. A pass says nothing about d = 100, about frequencies spanning more than the
  factor of 5 of part B' (except D1's tuned covariance), or about a given `n_steps`
  beyond 10.
- **The guard sees only predicted frequencies.** With the package's tuned covariance
  they are all 1, so G cannot act on a target whose tuned covariance does not whiten it
  (D1) and cannot tell an exact mode near a resonance from one beside it when $n h$ is
  large (section 10, A'). Its tests are A, A', B and B'; a pass does not extend to other
  covariance errors.
- **The cost criteria compare medians of meeting costs and of inefficiencies**, and L the
  pilot's 99% quantile where M does not apply. On the banana V detects a factor of 4
  with probability between 0.3 and 1.0, depending on the true spread, and on C2 it cannot
  fail in practice (section 5).
- **The default decision for the jitter** rests on cost cells where the jitter's excess
  over P is near the threshold (J5 at 1.15 to 1.17 in i137); a width's verdict there can
  turn on the seeds, with the probabilities stated in section 9.
- **One commit, one machine:** the records are comparable with each other, not with
  i137's, which ran at `b0b8f71` at a cap of 10,000 for the Gaussians.
- **CF is judged with the package's meeting step.** The arms at a meeting weight of 0.2
  test whether the weight explains its cost on four cells, in pilots; they do not test CF
  at 0.2 for unmet pairs, `k`, unbiasedness or cost per effective replicate.
- **Unbiasedness** is checked through pooled z-scores of first and second moments; a bias
  smaller than a few pooled standard errors is not detectable.

## 13. Choices this registration made where the issues left them open

Each can be overruled before the run.

1. **CF is one recipe**, acceptance 0.99 and jitter 0.5, not #91's grid of six; the grid
   would choose its best member after seeing the outcome. Alternative: the grid, with a
   rule for choosing among its members fixed here.
2. **CF keeps the package's meeting step** ($\gamma$ = 0.05, scale 1e-3), as #91's grid
   states, with CF.2 and P.2 at 0.2 on four cells. Alternative: #91's evidence
   configuration ($\gamma$ = 0.2, a MALA meeting step) in full.
3. **Dual averaging tunes the nominal step on the jittered kernel**, as i137 did; #91 says
   only that the target is reached by dual averaging. Alternative: tune without jitter,
   then jitter.
4. **The guard's $\delta$ = 0.451**, its band 1.3 for the move only, no $j = 0$ guard, and
   the fallback that keeps the step. Alternative: $\delta$ = 0.16, which keeps every stored
   tuning of the four cells and is predicted to fail at d = 10, n = 3.
5. **No combination arm** (section 2). Alternative: G followed by J2.
6. **The package's pilot cap and 100 pairs per seed**, judged at the cap and at #91's
   2 x 10^5 gradients, instead of #91's 500 pairs.
7. **The cost statistic is the median meeting cost in gradients** (M), as #89 and #91
   phrase it, with V for the cost per effective replicate and L for the tail.
   Alternative: V alone.
8. **The banana in d = 20 starts from exact draws**, so that it tests curvature, not the
   approach from a wide start (D3 tests that). Alternative: U[-5, 5]^20.
9. **The ill-conditioned Gaussian is d = 20** with the tuned covariance only. Alternatives:
   d = 100 (about 75 s per seed and arm for P's pilot alone, by the probe's frequencies),
   or an identity-mass version.
10. **MALA arms MJ5 and MCF** on four cells; G does not apply to MALA.
11. **The recommended jitter width** is the widest default candidate. Alternative: the
    cheapest.
12. **Seeds 9100 onward and 20000 onward**, unused by any earlier experiment of the
    package.
13. **C2 and C4 are in tier 1** (section 11), rather than an "undecided" verdict for the
    arms with a cut cell; "undecided" remains for the MALA arms of part E.
14. **The held-out cells** are `gaussian(5)` with `n_steps` 3, 4, 6 and the unequal
    Gaussian of 8 dimensions with scales to 1/5, not the review's example
    (`gaussian(5)` with 4 and 7, scales to 1/3 in 6 dimensions), at which P does not
    resonate: no probe tuning of those exceeds an exact factor of 0.99 (`heldout_review.out`: largest
    factors 0.925, 0.954 and 0.978 over 100, 100 and 30 probe tunings).

**Files of the design probes** (to be committed beside this file; the scripts read
i137's stores at `notes/experiments/_rows/`, relative to the root of a checkout that holds
them, and their own files beside them): `retune.py`, `unequal_tune.py`,
`unequal_tune2.py`, `scan_heldout.py`, `probe_tune.py`, `probe100.py`, `probe_rot.py`,
`targets.py`, `guard.py`, `evalguard.py`, `evalunequal.py`, `post.py`, `trig.py`,
`phases.py`, `jpred.py`, `heldout.py`, `model.py`, `spreads.py`, `ineffpower.py`,
`kpower.py`, `heldout_review.py`, `costs.py`, and their outputs `tunings_d10.jsonl`, `tunings_unequal.jsonl`,
`tunings_g5.jsonl`, `tunings_uneq8.jsonl`, `tunings_g5_review.jsonl`,
`tunings_uneq6.jsonl`, `heldout_review.out`, `probe_tune.out`, `probe_rot.out`,
`probe100.out`, `scan_heldout.out`, `evalguard.out`, `evalunequal.out`, `post.out`,
`trig.out`, `phases.out`, `jpred.out`, `heldout.out`, `model.out`, `spreads.out`,
`ineffpower.out`, `kpower.out`, `costs.out`, `setup.out`. With them, the review's report
and its probes.

## 14. Changes after the adversarial review (2026-11-14)

The first version is kept as `PREREGISTRATION.v1.md`. The review's labels:

- **B1** (U passes an arm whose runs raise; its stated power was wrong). Added K (at most 1
  seed of 30 where `k` cannot be chosen, on every pilot cell), G's tuning-only test (part
  T: at most 1% of 300 new tunings per resonant cell above 0.9998 after the guard), and L
  (the tail, $k\,g$, at most 3 times the best arm that holds K, where M does not apply).
  Section 5 now gives the six failing seeds at the cap (12 to 96 unmet pairs) and U's
  recomputed power (0.14, 0.28, 0.59, 0.86); the G predictions use the rate above 0.9998
  (2 of 100 at d = 10, n = 10), with U's probability 0.28, K's 0.12 and the tuning test's
  0.85.
- **B2** (a full run undefined or unbounded). A full run runs replicates only if no pilot
  pair is unmet and `k` is at most 20,000; Z and V use the seeds with replicates and report
  their number; V is void below half the seeds in either arm.
- **B3** (cuts that remove criteria). C2 and C4 moved into tier 1; only E and F can be cut,
  and a MALA arm with a cut cell is "undecided" (rule 4). Chosen over "undecided" for all
  arms because C2 and C4 are where the default `n_steps` is costed, which the default
  question needs, and the budget allows them.
- **I1** (CF's cost set by the meeting weight). Added P.2 and CF.2 at `meeting_weight=0.2`
  on A's default cells, C5 and D1, pilots only, judged by M against each other; section 8
  now says what refutes CF at the package's meeting step and what the arms at 0.2 decide;
  the model's predictions for them are in `model.out`.
- **I2** (G tested on the cells its rule was chosen on; its fallback). Added the held-out
  cells A' and B' for G and the jitters (section 13, choice 14, says why not the review's
  example cells). The fallback now keeps the step (it moved 3 of 30 tunings of B onto a
  resonance, `p1_guard.out`); B's cells and B' decide G's scope separately (rule, section
  8).
- **I3** (spreads and powers). Section 5 gives the spreads of all nine cells of A, M's
  power at the upper ends of the spreads (0.84), V's power on the banana as "between 0.3
  and 1.0 at a factor 4", and that V cannot fail on C2; section 7 states the opposite
  burdens of M and V and why; the prediction for the jitters on D1 is withdrawn and the
  implied verdict made conditional; J5's chance of passing M on part A (0.90) is stated.
- **I4** (an iteration cap is not a matched budget). U is judged at the cap and at
  $2 \times 10^5$ gradients per pair.
- **I5** (chance failures). A failed `check_coupling` is repeated once at 80,000 pairs; a
  Z refusal is confirmed on fresh seeds from 9800; where P's own z exceeds 3.5, the
  difference z-score replaces the voiding of the cell; Z applies to every full run of
  every arm, in the text, the table and rule 1.
- **I6** (no reference code for MALA). The runner and every arm's kernel are committed
  before the first registered seed, and a dated note names the commit; line 65 now says
  "each of its 4 chains".
- **Minor, P's intervals.** Stated as references, not tests; P's prediction at B's default
  widened to 0 to 8 of 30.
- **Minor, CF's `k` on D3.** The bound is now 1,500, allowing the model's 15 more
  iterations of wait at a meeting weight of 0.05, and the text says #91's figure was at 0.2.
- **Minor, D2 at 5 seeds.** CF runs at 10 seeds on D2.
- **Minor, the tie of step 5.** Step 5 now keeps the step, so it has no tie; step 4 states
  its tie.
- **Minor, the level 3.5.** Section 7 says it departs from #91's 3, and why.
- **Minor, MALA steps.** The review's dual-averaging probes (`p4_mala.out`) are quoted in
  part E's predictions.
- **Probe scripts.** The machine path to i137's stores is replaced by a path relative to
  the repository root; every probe whose output this version quotes was rerun with it
  (`evalguard`, `evalunequal`, `post`, `phases`, `trig`, `costs`, `spreads`, `jpred`,
  `model`), and `post.out` and `evalunequal.out` changed only by the new fallback.

### Changes after the second review (2026-11-14)

- **N1** (L or M decided by five seeds of P). L applies on the nine cells of T by name,
  whatever P shows there, and on any other cell where P has a pair unmet at the cap; M on
  every other cell where P has no pair unmet at the cap (section 7 and its table). The
  predictions of part B now give L where they gave M ($\rho_M < 1$ for CF on B's 3 and 10
  became "L holds").
- **N2** (two meanings of "unmet"). "Unmet at the cap" is written wherever unmet pairs
  decide which criterion applies or whether a full run runs (sections 4 and 7). On a cell
  where P itself fails U at the gradient budget while holding it at the cap, U at the
  budget is reported for every arm and U at the cap alone is judged.
- **N3** (T refuses on a proxy). Rule 1 now says that T alone refuses G. The runner checks
  the proxy inside the experiment: over the registered pilot records of P and G on the
  cells of T, factor above 0.9998 against `k` not chosen; T is judged only if at least 5
  records lie above 0.9998 and `k` failed at at least half of them, and is otherwise
  reported. Section 10 now says that G's 1% was set with its probe tunings after the guard
  known, and that A' is held out from G's rule but not from the prediction of its result.
- **N4** (a full run not bounded). Replicates run with `max_iter=10**6`; if any replicate
  reaches it, the record's replicates are discarded and the record counts as one without
  replicates. The sentence that no run can grow without bound is replaced by this rule.
- **N5** (a seed without replicates seen by no criterion). In a full-run cell such a seed
  counts as a seed where `k` could not be chosen, for K (sections 4 and 7).
- **N6** (J2 on L). L is no longer predicted for J2 on the cells where its worst factor
  exceeds 0.9 (A's d = 10, n = 3; B's three cells; A''s 3; B'), with the model's ratio of
  2.8 to 3.0 stated; the implied verdict for J2 says so.

