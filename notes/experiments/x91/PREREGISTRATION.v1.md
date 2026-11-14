# x91: the test of contraction-first tuning (#91) and of two remedies for the leapfrog resonance (#137), registered before any run

This file fixes, before any run of the experiment, its arms, targets, seeds, measurements,
predictions, criteria and decision rules. It answers Josh's rulings of 2026-11-13 on #89
(question 1: contraction-first tuning is opt-in now and becomes the default only if its
pre-registered test passes) and of 2026-11-14 on #137 (a guard that acts only near a
resonance and a shared step jitter go to this test, with the Gaussian of unequal scales
added; the guard as tested in the experiment of #148 is not adopted). It is not changed
after it is posted on #91. Where it must be corrected, a dated note is added below the
text and the outcome is scored against the text as posted.

Short names: **P** is the package as it is; **G** the guard; **J2, J3, J5** the shared
jitter at widths 0.2, 0.3 and 0.5; **CF** the contraction-first tuning of #91; **MP,
MJ5, MCF** the corresponding MALA arms. A **cell** is a target with its settings; a
**record** is one arm at one seed of one cell.

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
    (`probe_tune.out`), and of P on rotated Gaussians of condition number 10^4 at
    d = 10, 20 and 50, seeds 900 and 901 (`probe_rot.out`), and at d = 100, seed 900, of
    P and CF (`probe100.out`).
  - Computations on stored and probe tunings only: the guard's rule and its
    alternatives (`evalguard.out`, `evalunequal.out`, `post.out`, `trig.out`,
    `phases.out`), the jittered contraction factors (`jpred.out`), the spreads between seeds and the
    power of the cost criteria (`spreads.out`, `ineffpower.out`), and the stores'
    recorded seconds (`costs.out`).
- **Not seen:** any meeting time, `k`, estimate or inefficiency of any arm at any seed
  registered in section 5, and any meeting time of G, J2, J3, CF or the MALA arms at any
  seed.

## 2. The arms

On a whitened Gaussian mode of frequency $w$, with $h = \varepsilon w$ and
$\theta(h) = 2\arcsin(h/2)$ for $h < 2$, one accepted trajectory of $n$ leapfrog steps
with a shared momentum multiplies the gap between the chains along that mode by
$\cos(n\theta)$ (`i137_reproduce_unequal.out` finds the ratio of distances equal to
$\cos(n\theta)$ at three tuned steps). At
$n\theta = j\pi$, $j \ge 1$, the pair does not contract along that mode: a resonance.

### HMC arms

| Arm | What it is | What it changes |
|---|---|---|
| P | `HMC(step_size=None, n_steps=..., cov=...)` as on `main` at the run's commit | nothing |
| G | P, then the guard below moves the tuned step | the tuned step only; P's kernel and draws. Where it keeps the step, its record is P's, bit for bit |
| J2, J3, J5 | P (acceptance 0.8) with every trajectory's step $\varepsilon(1 + j(2u - 1))$, one uniform $u$ per step, shared by the two chains of a pair; $j$ = 0.2, 0.3, 0.5 | the kernel: one more uniform per step, so the numbers of every HMC run |
| CF | #91 exactly: J5, with dual averaging toward an acceptance rate of 0.99 instead of 0.8 | the kernel and the tuning |

**The jitter (J2, J3, J5, CF).** The nominal step $\varepsilon$ is what dual averaging
returns, tuned on the jittered kernel itself (each of its 1,000 steps draws its own $u$).
The number of leapfrog steps is fixed by the nominal step: the given `n_steps`, or
$\lceil 1/\varepsilon\rceil$ (at most 1,000), so the jitter also varies the integration
time between $1 - j$ and $1 + j$ of its nominal value, as #91 states. The draw order of a
step is: the `Mixture`'s uniform that chooses the component; then, for a trajectory, the
jitter's uniform, `d` normals for the momentum, the acceptance uniform; for the
random-walk meeting step, the package's draws, unchanged and not jittered. A coupled step
reads the same numbers, each shared by the two chains (`i137_common._JitterLeapfrog` is
this kernel). The meeting step keeps the package's `meeting_weight=0.05` and
`meeting_scale=1e-3`. Exactness: for a fixed step the leapfrog Metropolis kernel leaves
$\pi$ invariant; $u$ is drawn independently of the state, so the jittered kernel is a
mixture of $\pi$-invariant kernels; each chain reads its numbers in the order of the step
alone, so its marginal transition is that of the uncoupled kernel, and equal chains get
identical arithmetic.

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
5. *No step satisfies 4:* the grid step that maximizes $\min_i r(\varepsilon, w_i)$
   (predicted frequencies, no band), smallest $\lvert s\rvert$ on ties.

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

## 3. Exactness before any run

Every new coupled step (J2, J3, J5, CF, MJ5, MCF) passes `telescopic.testing.check_coupling`
before any of its records counts: each kernel tuned on `examples.gaussian(d)` at seed
9700, d = 2 and 10, HMC arms at `n_steps` 2 and 5 and default, from a near pair (the
origin and a point 2e-4 away along each axis) and a far pair (the origin and
(1, ..., 1)), 20,000 pairs, check seed 9701. Pass: `passed()` at its default level 1e-3
for the near pair; for the far pair the same without the condition that pairs met
(as `i137_check_coupling.py`). A kernel that fails one check is fixed or dropped before
the runs, and the failure is reported.

## 4. Targets and cells

All runs: `CoupledSampler(log_prob, initial, grad=..., vectorize=True, kernel=..., rng=seed)`.
**Pilot** means `tune()`, then `meeting_times(100)` at the package's cap of 100,000
iterations, then `choose_k`. **Full run** means `run(100)` with the automatic `k`, lag
`k` and `m = 10k` (its pilot is the same 100 pairs at the same cap). One commit for
every record.

| Part | Cell | Start | Covariance | `n_steps` | Run | Arms | Seeds |
|---|---|---|---|---|---|---|---|
| A | `examples.gaussian(2)` and `(10)` | its N(0, 4I) | tuned | default, 1, 2, 3, 5, 10 | pilot | P, G, J2, J3, J5, CF | 30; P 5 in the three resonant cells (d = 2, n = 2; d = 10, n = 3 and 10) |
| B | N(0, diag(s^2)), d = 10, `s = np.geomspace(1, 0.25, 10)` | N(0, 4I) | given `np.eye(10)` | default, 3, 10 | pilot | P, G, J2, J3, J5, CF | 30; P 10 at the default |
| C1 | `examples.banana()` | exact draws | tuned | 2 | full | P, G, J2, J3, J5, CF | 20 |
| C2 | `examples.banana()` | exact draws | tuned | default | full | P, G, J2, J3, J5, CF | 10 |
| C3 | `examples.mixture()` | its N(10, 10^2) | tuned | 2 | full | P, G, J2, J3, J5, CF | 20 |
| C4 | `examples.mixture()` | its N(10, 10^2) | tuned | default | full | P, G, J2, J3, J5, CF | 20 |
| C5 | logistic regression, below | N(fit, 100 H^-1) | tuned | default | full | P, G, J2, J3, J5, CF | 20 |
| D1 | rotated Gaussian, d = 20, condition 10^4, below | N(0, 4I) | tuned | default | pilot | P, G, J2, J3, J5, CF | 10 |
| D2 | banana with every coordinate curved, d = 20, below | exact draws | tuned | default | full | P, G, J5, CF | 5 |
| D3 | `examples.banana()` | its U[-5, 5]^2 | tuned | default | full | P, G, J5, CF | P 3, J5 5, CF 10 |
| E1 | `examples.banana()` | U[-5, 5]^2 | tuned | | full | MP, MJ5, MCF | MP 3, others 5 |
| E2 | logistic regression | N(fit, 100 H^-1) | tuned | | full | MP, MJ5, MCF | 10 |
| E3 | `examples.gaussian(10)` | N(0, 4I) | tuned | | full | MP, MJ5, MCF | 10 |
| E4 | `examples.banana()` | exact draws | tuned | | full | MP, MJ5, MCF | 10 |
| F | `examples.funnel(10)` | its default | tuned | default | pilot at a cap of 10,000, reported only | P, J5, CF | 3 |

- **Logistic regression** (C5, E2): the first design of `m10_mala_logistic.py`: 200
  outcomes, $x_i = (1, z_i)$, $z_i \sim N(0, I_5)$, coefficients (-0.5, 1, -1, 0.5, 0, 2),
  data from `np.random.default_rng(1)`, prior $N(0, 5^2 I_6)$, start
  $N(\hat\theta, 100 H^{-1})$ with BFGS's fit and inverse-Hessian approximation from 0.
  It is the log-concave target where the automatic MALA fails (19 of 100 pilot pairs
  unmet at 3,000, `m10_mala_logistic.out`), the logistic target #91 and #89 name, and the
  one where #91's evidence put CF's median cost closest to the threshold (+17%). Its
  reference values are computed independently, by self-normalized importance sampling
  from a multivariate t with 5 degrees of freedom located at the posterior mode with
  scale matrix the inverse of the exact Hessian there, 10^6 draws from
  `np.random.default_rng(9799)`, with the delta-method standard error; if the effective
  sample size is below 10^5, the draws are increased tenfold.
- **Rotated Gaussian** (D1): $\Sigma = Q\,\mathrm{diag}(\lambda)\,Q^\top$, $\lambda$
  geometric from 1 to 10^-4, $Q$ the Q factor of a 20 x 20 standard normal matrix from
  `np.random.default_rng(91)`. #91 planned d = 100; see section 12.
- **Banana in d = 20** (D2): $x_0 \sim N(1, 1/2)$, $x_j \mid x_0 \sim N(x_0^2, 1/20)$,
  $j = 1, \dots, 19$, $\log\pi = -(1 - x_0)^2 - 10\sum_j (x_j - x_0^2)^2$, started from
  exact draws, as #91's test lists it.
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
| A | 9100 to 9129 |
| B | 9200 to 9229 |
| C1, C2, C3, C4, C5 | 9300 to 9319, 9320 to 9329, 9340 to 9359, 9360 to 9379, 9380 to 9399 |
| D1, D2, D3 | 9400 to 9409, 9420 to 9424, 9440 to 9449 |
| E1, E2, E3, E4 | 9500 to 9504, 9520 to 9529, 9540 to 9549, 9560 to 9569 |
| F | 9600 to 9602 |
| check_coupling | tuning 9700, checks 9701; logistic reference 9799 |

Where G keeps the step at a seed for which P has no record (a resonant cell of part A
beyond P's 5 seeds, or part D3 beyond P's 3), P's record at that seed is computed and
used for both.

**How many, and why.** The numbers come from the spreads between seeds in i137's stores
(`spreads.out`, `ineffpower.out`).

- *Parts A and B, 30 seeds.* The resonance shows as a fraction of seeds: one failing seed
  leaves 10 to 99 of its 100 pairs unmet (`i137_baseline.out`), so the criterion on
  unmet pairs (section 7, U) fails an arm with probability $1 - (1 - p)^{30}$ when a
  fraction $p$ of its tunings fail: 0.26 at $p$ = 0.01, 0.79 at 0.05, 0.96 at 0.10. P's
  failing fractions at the package's cap were 0.09 to 0.49 (#137's validation, 100
  seeds), so 5 seeds of P in the resonant cells are a reference, not a test.
- *Part C, the median cost.* The per-seed log median meeting time has a standard
  deviation between seeds of 0.10 (P) and 0.08 (J5) on the banana with `n_steps=2`,
  0.06 to 0.09 on the Gaussians, and up to 0.38 (J5 on the mixture, whose medians are
  bimodal). With 20 seeds per arm the criterion M (threshold 1.20 on the ratio of
  geometric means) fails an arm on the banana with probability below 0.001, 0.001, 0.5 and
  0.997 at true ratios 1.0, 1.1, 1.2 and 1.3; with the mixture's spreads, 0.04, 0.21,
  0.5, 0.78 and 0.93 at 1.0 to 1.4. The old test's cost arm on the banana failed an arm
  only above 3.19 times the package's mean inefficiency (`i137_nongaussian.out`).
- *Part C, the inefficiency.* The per-seed inefficiency of the banana is heavy-tailed
  (log standard deviation 2.6 for P with `n_steps=2`, set by one seed of five). The criterion V,
  simulated by resampling i137's per-seed values, fails an arm whose true ratio is 1.0,
  1.3, 2 and 4 with probability 0.018, 0.075, 0.34 and 0.84 at 20 seeds with P's banana
  spread; 0.0, 0.62, 1.0, 1.0 with J5's mixture spread; 0.0, 0.99, 1.0, 1.0 with P's mixture
  spread.
  On the banana it can resolve a factor of about 4, not 20%: that is what 20 seeds of a
  heavy-tailed estimator allow, and M is the banana's cost criterion at 20%.
- *C2 cell (banana, default `n_steps`) at 10 seeds:* P's records there cost about 50 to
  90 s each (section 11); 10 seeds give M a standard error of about 0.10 with the
  spreads of the banana with `n_steps=5` (0.28 and 0.11), for which M fails an arm at
  ratios 1.0, 1.3 and 1.5 with probability 0.03, 0.80 and 0.99.
- *Parts D, E, F:* sized by cost; D2 and D3 test U on 500 to 1,000 pairs, so that one
  unmet pair in 100 is distinguishable from none.

## 6. Measurements and records

Each record is one JSON line in a store per part (in a folder git ignores, as
`notes/experiments/_rows/`), keyed by part, cell, arm and seed, carrying the setup line
of `i85_setup.setup()` (commit, processor, thread variables, Python, NumPy and SciPy
versions). Outputs are printed from the stores, and raise if the records mix commits
(as `i137_common.header`). A record holds:

- the nominal step, `n_steps`, jitter width, target acceptance; for G its
  $\varepsilon_0$, action and $\min_i r$ before and after; for the Gaussians of parts A,
  B and D1 the exact whitened frequencies and the largest factor $\lvert\cos n\theta\rvert$
  (for jittered arms its geometric mean over the jitter, as
  `i137_common.jitter_factor`), and for B the pilot's predicted frequencies;
- gradient and log-density evaluations of tuning, of the pilot and of the replicates,
  as the sampler's `Target` counts them;
- the 100 pilot meeting times (inf where unmet at the cap), and from them the unmet
  count, the quantiles 0.5, 0.9, 0.99, `k` or the error of `choose_k`, and
  $g$ = pilot gradient evaluations / sum of the pilot meeting times (unmet pairs at the
  cap), the gradients per pair-iteration;
- for a full run: `k`, `m`, replicates met, each function's estimate, standard error and
  z-score against its exact or reference value, and the inefficiency
  $\mathrm{SE}^2 \times$ (gradient evaluations of the replicates);
- the record's processor seconds; `same_as` where it is a copy of P's record.

The meeting cost of pair $i$ is $c_i = \tau_i g$. The record also gives the number of pairs
whose cost exceeds $2 \times 10^5$ gradients, #91's original budget.

## 7. Criteria

Each criterion is stated for one arm and one cell, against P on the same cell.

- **U, unmet pairs.** The pooled fraction of pilot pairs unmet at the package's cap is at
  most 1% (at most 30 of 3,000 in parts A and B; 20 of 2,000; 10 of 1,000; 5 of 500).
  This is Josh's threshold (#89). The record also says at how many seeds `k` could not be
  chosen, which i137's criterion used (at most 2 of 30).
- **M, median cost.** On a cell where P leaves no pair unmet at any seed: the ratio
  $\rho_M = \exp(\overline{\ln m_A} - \overline{\ln m_P})$, with $m$ a seed's median
  meeting cost $\mathrm{median}_i(\tau_i)\,g$ and the bar a mean over seeds. The arm fails
  when $\rho_M > 1.20$ (Josh's threshold, #89). A point estimate against the threshold is
  used because its standard error is small on these cells (section 5); the probability
  of each outcome at the predicted ratio is stated in section 9.
- **V, cost per effective replicate** (full runs of part C). The ratio $\rho_V$ of the
  medians over seeds of the per-seed inefficiency, arm over P; the arm fails when the 5%
  point of its bootstrap distribution (seeds resampled within each arm, 10,000
  resamples, seed 9798) exceeds 1.20: the data then show a rise above 20% at one-sided
  95%. The median and the bootstrap are chosen because the banana's inefficiency is
  heavy-tailed; the price is the power stated in section 5.
- **Z, unbiasedness** (every full run). For each function, the pooled estimate is the
  mean over seeds of the seeds' estimates, its standard error the root of the sum of
  their squared standard errors divided by the number of seeds, and
  $z$ = (pooled estimate - exact value) / that error (for the logistic regression the
  reference's standard error is added in quadrature). The arm fails when some
  $\lvert z\rvert > 3.5$, a Bonferroni level for the about 130 z-scores of the
  registered full runs (2 to 6 per cell and arm); under normality the family-wise false
  rate is then about 0.06. If P's own $\lvert z\rvert$ exceeds 3.5 on a cell, the
  cell's reference or its error estimates are suspect: the cell's Z verdicts are void
  for every arm and the cell is reported.
- **Funnel (part F):** no criterion; reported.

The bootstrap of V uses 10,000 resamples; its power in section 5 was simulated with
2,000 per experiment and 400 experiments per entry.

**Which criteria apply.**

| Arm | U on | M on | V on | Z on |
|---|---|---|---|---|
| G, J2, J3, J5 | A, B (their purpose), C, D1; D2, D3 reported | every cell of A, B, C, D1 where P has no unmet pair | C | C |
| CF | every non-funnel cell it runs (A, B, C, D) | every cell of A, B, C, D1 where P has no unmet pair | C | C, D2, D3 |
| MJ5, MCF | E1, E2 | E3, E4 (and E2 if MP has no unmet pair) | none | E |

G, J2, J3 and J5 are remedies for a resonance: on D2 and D3, whose failures are of
curvature, they are reported, not judged.

## 8. Decision rules

For each arm, in this order:

1. **Refused** if it fails Z anywhere, or fails `check_coupling`, or fails U on any cell
   of its purpose (A and B for the jitters; A for G, whose part B decides its scope
   below; D2 and D3 for CF; E1 and E2 for the MALA arms).
2. **Default candidate** if it passes U, M and V on every cell where the table above
   applies them. A default candidate becomes the default only after Josh has
   seen the result (#89).
3. **Opt-in** otherwise: it holds U on the cells of its purpose and fails some other
   criterion of rule 2.

Between the remedies of #137:

- Among the jitter widths that are default candidates, the widest is recommended: it
  sweeps the most phase, which is the margin against resonances this test does not
  contain, and its cost is already within 20%. If none is a default candidate, the
  narrowest that passes U on A and B is the opt-in recommended.
- G passes its purpose with U on all 12 cells of A. If it also holds U on the 3 cells of
  B, it covers a given covariance too; if it fails B, it is offered only for a given
  `n_steps` with the tuned covariance, and the record says which seeds of B failed. G changes no kernel, so where
  it passes it is preferred to a jitter for that case.

**What refutes CF as the default** (the result that answers #89's question 1 with no):
any non-funnel cell with more than 1% of its pilot pairs unmet; or $\rho_M > 1.20$ on any
cell where M applies; or V's bound above 1.20 on any cell of C; or a failure of Z. **What
refutes it as an opt-in for curvature:** more than 1% unmet on D2 or D3, or a failure of
Z.

## 9. Predictions

Each prediction is an inequality, scored held or failed. The reasoning: (i) the factor
$\lvert\cos n\theta\rvert$ at the exact frequencies, before and after G, over the 100
stored or probe tunings of each Gaussian cell (`post.out`); (ii) the geometric-mean factor
over the jitter at P's tuned steps (`jpred.out`); (iii) a meeting-time model,
$\tau \approx 1/\gamma + 6/(-\ln f)$, $f$ the slowest mode's factor, whose constant 6
was set so that it gives P's median at `examples.gaussian(2)` within 10% (28.2 against
25.8; `model.out`); (iv) i137's measured outcomes for J5 (remedy c there) and for the old
guard's steps, which G's equal where it moves (checked on the 30 stored pilot seeds of
the three resonant cells: identical steps at 89 of 90, the 90th kept by G).

**Part A** (median meeting time: median over seeds of the seed's median).

| Cell | P | G | J2 | J3 | J5 | CF |
|---|---|---|---|---|---|---|
| d = 2, n = 2 | `k` fails at 0 to 3 of 5 | moves at 30 of 30; U holds; median 20 to 40 | U holds; median 40 to 100 | U holds; median 25 to 50 | U holds; median 25 to 40 | U holds |
| d = 10, n = 3 | `k` fails at 1 to 4 of 5 | moves at 30; U holds; median 35 to 60 | U holds; median 60 to 140 | U holds; median 35 to 75 | U holds; median 30 to 45 | U holds |
| d = 10, n = 10 | `k` fails at 0 to 2 of 5 | moves at 26 or more; U fails with probability about 0.5 | U holds; median 25 to 60 | U holds; median 25 to 60 | U holds; median 35 to 55 | U holds |
| other 9 cells | `k` chosen at 30 of 30 | keeps at every seed of d = 2 with n = default, 1, 3 and of d = 10 with n = default, 1, 2, 5; moves at 0 to 4 of 30 with d = 2, n = 5 and 0 to 3 with n = 10 | U holds; $\rho_M \le 1.10$ | U holds; $\rho_M \le 1.15$ | U holds; $\rho_M \le 1.20$ at 8 or 9 of the 9 cells | U holds; $\rho_M \ge 2$ at the default `n_steps`, d = 2 and 10 |

Reasons. P: #137's validation at the package's cap gave failing fractions 0.21, 0.49 and
0.09. G: over 100 tunings each, after the guard the exact largest factor exceeds 0.999 at
0 (d = 2, n = 2), 0 (d = 10, n = 3) and 4 (d = 10, n = 10: the four tunings G keeps because
the predicted phase at $w = 1$ is at least 0.451 from every $j\pi$, while an exact mode,
between 0.71 and 1.29, is on a resonance) of 100, against 38, 73 and 29 for P; 2 of 100 exceed 0.9998. At $p$ = 0.02 to 0.04 per
seed, at least one failing seed in 30 has probability 0.45 to 0.71, and one failing seed
leaves more than 30 pairs unmet. Its moves elsewhere: 5 and 2 of 100 tunings at d = 2,
n = 5 and 10 (`evalguard.out`). Jitters: the worst geometric-mean factor at P's steps is
at most 0.93 (J2), 0.84 (J3), 0.61 (J5) in every cell (`jpred.out`), which the model turns
into the medians above; J5's are i137's 31.2, 37.0, 42.8 within 30%. $\rho_M$ for J5:
i137's ratios of medians were 1.07, 1.07, 1.16, 1.08, 1.19 at the cells of default and
small `n_steps` and below 1 at the others; J3 and J2 are predicted to scale the excess as
$j^2$. CF: dual averaging toward 0.99 returned $\varepsilon$ = 0.31 and 0.21 with 4 and 5
leapfrog steps at d = 2 and 10, against P's 1.13 and 0.79 with 1 and 2 (`probe_tune.out`);
the model gives $\rho_M$ = 3.4 and 3.1 (`model.out`).

**Part B.**

| `n_steps` | P | G | J2 | J3 | J5 | CF |
|---|---|---|---|---|---|---|
| default | `k` fails at 1 to 5 of 10 | falls back at 28 or more of 30; `k` fails at 1 to 6 of 30; U fails | U holds; median 60 to 140 | U holds; median 35 to 75 | U holds; median 35 to 55 | U holds |
| 3 | `k` at 30 of 30; median 400 to 1,100 | moves at 30; U holds; median 60 to 110 | U holds | U holds | U holds; median 30 to 50 | U holds; $\rho_M < 1$ |
| 10 | `k` at 30 of 30; median 800 to 2,500 | moves at 30; U holds; median 55 to 100 | U holds | U holds | U holds; median 35 to 55 | U holds; $\rho_M < 1$ |

Reasons. On seeds 0 to 29 retuned at `4cb3acd`, P's tunings have an exact factor above 0.999 at 7 and
above 0.9998 at 2 (default); i137 had 3 of 10 failing at the cap. G finds no step for its
band at any default tuning (30 of 30 fall back) and leaves 4 of 30 above 0.999 and 3 above
0.9998; with given `n_steps` it moves every tuning to an exact factor of at most 0.908
(`evalunequal.out`, `post.out`). The jitters' worst factors are 0.93, 0.84, 0.61
(`jpred.out`). i137's medians: P 720 and 1,381; old guard 79 and 72; J5 43.0, 39.2, 44.8.

**Part C** (G keeps the step at every seed of C: the predicted phase is 1.7 or more from
$\pi$ on every stored banana and mixture tuning (`phases.out`) and 1.26 or more on the
probe tunings of the default `n_steps` of the banana, the mixture and the logistic
regression, so its records are P's and it passes by identity).

| Cell | J2 | J3 | J5 | CF |
|---|---|---|---|---|
| C1 banana, n = 2 | $\rho_M \le 1.10$; V holds | $\rho_M \le 1.12$; V holds | $\rho_M$ in 1.08 to 1.30 (i137: 1.18), M fails with probability about 0.3; V holds | $\rho_M > 1.20$ |
| C2 banana, default | U holds | U holds | U holds; $\rho_M < 1.20$ | $\rho_M > 1.20$ |
| C3 mixture, n = 2 | $\rho_M < 1$; V holds | $\rho_M < 1$; V holds | $\rho_M < 1$; $\rho_V$ in 1.15 to 1.45 (i137: 1.29), V fails with probability about 0.6 | V fails ($\rho_V \ge 1.5$) |
| C4 mixture, default | V holds | V holds | $\rho_V > 1.1$ | V fails |
| C5 logistic | $\rho_M \le 1.10$ | $\rho_M \le 1.15$ | $\rho_M \le 1.25$ | $\rho_M \ge 1.8$ |

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
  (`probe_rot.out`). Every arm holds U; P's median meeting time is between 1,000 and
  10,000 iterations, set by the slowest mode (the model's $6/(-\ln\cos(w_{\min}T))$, with
  the probes' $w_{\min}$ and $T = n\varepsilon$ of 1.16 and 1.36, gives 4,500 and 1,200);
  every jitter's median is within a factor 2 of P's in iterations; CF's $\rho_M \ge 2$. G keeps the step (it reads $w = 1$), so its records are P's.
- D2 (banana, d = 20): CF holds U. No prediction is made for P and J5 on U: no
  measurement of either exists on a target with curvature in many directions.
- D3 (banana from U[-5, 5]^2): P leaves at least 50 of 100 pairs unmet at each seed (88
  of 100 in `m9`, 475 of 500 in #91's evidence); J5 fails U (#91's screen: jitter 0.5 at
  acceptance 0.9 left 25 of 100); CF holds U, with `k` below 1,000 at every seed (#91: 0
  of 500 unmet, 99% quantile of meeting cost 8,073 gradients with the random-walk meeting
  step); G's records are P's.
- CF passes Z on D2 and D3.

**Part E (MALA).** MP fails U on E1 and E2 (`benchmarks/results/acb12f9.json`;
`m10_mala_logistic.out`). MJ5 holds U on E2 and fails it on E1. MCF holds U on E2; no
prediction for MCF on E1. MCF's $\rho_M \ge 2$ on E3. MJ5's $\rho_M \le 1.25$ on E3.
Reasons: on E2 the failing chains start where the largest eigenvalue of $\Sigma H$
exceeds $4/\varepsilon^2$ at the tuned step 1.24, and steps of 0.7 met every pair
(`m10_mala_logistic.out`); J5 draws a step below 0.75 times the nominal one with
probability 0.25 per iteration.

**Part F.** Every arm leaves more than 1% of its pairs unmet at a cap of 10,000; CF's
tuning returns 40 or more leapfrog steps at some seed (42 to 180 at probe seeds,
`probe_tune.out`).

**The verdicts this implies.** CF: opt-in, not a default (it fails M on the Gaussians
and the logistic regression). G: refused or limited to a given `n_steps` (it fails U on
B's default and possibly on A's d = 10, n = 10). J3 or J2: default candidates; J5:
opt-in or default candidate depending on M on the banana and V on the mixture.

## 10. What was fixed after seeing which data

| Choice | Set from | Held out |
|---|---|---|
| G's band 1.3 | i137's d = 10 tunings at seeds 0 to 99 (inherited from i137) | every registered seed; B, C, D |
| G's bound $\delta = \arccos 0.9$ | the old guard's bound, a design value; kept after comparing $\delta$ = 0.16, 0.20, 0.30, 0.451, 0.60, 0.75 on the stored d = 2 and the probe d = 10 tunings at seeds 0 to 99 (`evalguard.out`) | every registered seed |
| G's trigger without a band, and no $j = 0$ guard | the review of #148 (the band and the $j = 0$ end moved steps on no resonance) and the same comparison | every registered seed |
| G's fallback | the probe tunings of B at seeds 0 to 29 (`evalunequal.out`): with it 4 of 30 stay above 0.999, against 7 if the step were kept | every registered seed of B |
| The rotated Gaussian's d = 20 | probe tunings at d = 10, 20, 50 (seeds 900, 901) and 100 (seed 900) (`probe_rot.out`, `probe100.out`) | its registered seeds |
| Seeds per cell | the spreads of i137's stores (section 5) | |
| Cost estimates | the stores' seconds and the probe tunings (section 11) | |

The jitter widths (0.2, 0.3, 0.5) are the ruling's; CF's 0.99 and 0.5 are #91's; the
thresholds 1% and 20% are Josh's (#89); 3.5 is a Bonferroni level. None was set from data.

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
predicted frequencies cannot tell these 7 apart from resonant tunings.

## 11. Budget

Estimated processor time on one thread, from the stores' seconds (`costs.out`) and the
probe tunings, with the package's cap of 100,000:

| Part | Estimate | How |
|---|---|---|
| A | 61 min | P: 119 s for the 270 pilots of the 9 cells without resonance (i137 at cap 10^4, all met, so equal at 10^5) + 95 s of tunings + about 950 s for 15 pilots of the resonant cells (one failing pilot at 10^5 took 43 s at d = 2; d = 10 pilots at 10^4 took up to 20 s, times 10); G: 126 s of tunings + 50 s of moved pilots + about 400 s for P's records where G keeps; J5: 167 s of pilots (i137) + 126 s; J2, J3: 1.6 times J5's pilots each; CF: 4 times J5's pilots + 160 s |
| B | 29 min | P: 158 s (`n_steps` 3, 10) + 320 s (default, 10 seeds, failing ones run 9 x 10^4 more iterations); G: 140 s + about 400 s for 2 to 6 failing seeds; J5 84 s; J2, J3 160 s each; CF 230 s |
| C | 86 min | per run at R = 100, about 0.6 times i137's R = 200 seconds: banana n = 2 890 s; banana default 1,120 s (P's records like i137's n = 5, 89 s at R = 200); mixture n = 2 1,050 s; mixture default 1,575 s; logistic 520 s |
| D | 58 min | D1 940 s (slowest pairs near 10^4 iterations); D2 1,500 s (P's pilot at up to 150 s if unmet); D3 1,070 s (P 3 x 120 s, as `acb12f9`'s 119 s) |
| E | 32 min | MP on E1 3 x 60 s, on E2 10 x 49 s (`acb12f9`, `m10`); the rest about 1,200 s |
| F | 15 min | CF up to 10^4 iterations of up to 180 leapfrog steps |
| Total | 4.7 h | |

**Order and cuts.** The runner records each record's seconds and runs in this order:
A, B, C1, C3, C5, D3, D2, D1 (tier 1), then C2, C4 (tier 2), then E (tier 3), then F
(tier 4). A cell is run whole or not at all. When the sum of recorded seconds reaches 6
hours, no new cell starts, and the cells not run are reported as cut. Cutting F loses
the funnel, which #89 asks for, reported separately; cutting E loses MALA, so #91's MALA
grid stays untested; cutting C2 or C4 leaves the default `n_steps` costed on the
logistic regression and the Gaussians alone. Tier 1 is never cut: if it alone exceeds 6
hours, it is finished and the overrun reported.

**Cut to fit, against #91's plan:** one acceptance target and one width for CF instead of
the grid {0.97, 0.99, 0.995} x {0.3, 0.5}; 100 pairs per seed at the package's cap
instead of 500 at 2 x 10^5 gradients (the record still counts pairs over that budget);
100 replicates per run instead of 3 runs of 2,000; d = 20 instead of 100 for the
ill-conditioned Gaussian, and no identity-mass version of it; P at 3 to 10 seeds in
cells where it is known to fail.

## 12. Limits

- **Coverage.** One to twenty dimensions, one funnel, one mixture, no hierarchical
  target. A pass says nothing about d = 100, about frequencies spanning more than the
  factor of 4 of part B (except D1's tuned covariance), or about a given `n_steps`
  beyond 10.
- **The guard sees only predicted frequencies.** With the package's tuned covariance
  they are all 1, so G cannot act on a target whose tuned covariance does not whiten it
  (D1) and cannot tell an exact mode near a resonance from one beside it when $n h$ is
  large (section 10). Its test is the Gaussian table and B; a pass does not extend to
  other covariance errors.
- **The cost criteria compare medians of meeting costs and of inefficiencies.** The tail
  of the meeting times, which sets `k` and so the length of a replicate, enters only
  through V; on the banana V resolves a factor of about 4 (section 5).
- **The default decision for the jitter** rests on cost cells where the jitter's excess
  over P is near the threshold (J5 at 1.16 to 1.19 in i137); a width's verdict there can
  turn on the seeds, with the probabilities stated in section 9.
- **One commit, one machine:** the records are comparable with each other, not with
  i137's, which ran at `b0b8f71` at a cap of 10,000 for the Gaussians.
- **CF is #91's recipe with the package's meeting step.** #91's evidence used
  $\gamma = 0.2$ and a MALA meeting step; a failure of CF on cost here does not refute
  that configuration.
- **Unbiasedness** is checked through pooled z-scores of first and second moments; a bias
  smaller than a few pooled standard errors is not detectable.

## 13. Choices this registration made where the issues left them open

Each can be overruled before the run.

1. **CF is one recipe**, acceptance 0.99 and jitter 0.5, not #91's grid of six; the grid
   would choose its best member after seeing the outcome. Alternative: the grid, with a
   rule for choosing among its members fixed here.
2. **CF keeps the package's meeting step** ($\gamma$ = 0.05, scale 1e-3), as #91's grid
   states. Alternative: #91's evidence configuration ($\gamma$ = 0.2, a MALA meeting step).
3. **Dual averaging tunes the nominal step on the jittered kernel**, as i137 did; #91 says
   only that the target is reached by dual averaging. Alternative: tune without jitter,
   then jitter.
4. **The guard's $\delta$ = 0.451**, its band 1.3 for the move only, no $j = 0$ guard, and
   the fallback of section 2. Alternative: $\delta$ = 0.16, which keeps every stored
   tuning of the four cells and is predicted to fail U at d = 10, n = 3.
5. **No combination arm** (section 2). Alternative: G followed by J2.
6. **The package's pilot cap and 100 pairs per seed**, as the assignment asks, instead of
   #91's 500 pairs at 2 x 10^5 gradients.
7. **The cost statistic is the median meeting cost in gradients** (M), as #89 and #91
   phrase it, with V added for the cost per effective replicate. Alternative: V alone.
8. **The banana in d = 20 starts from exact draws**, so that it tests curvature, not the
   approach from a wide start (D3 tests that). Alternative: U[-5, 5]^20.
9. **The ill-conditioned Gaussian is d = 20** with the tuned covariance only. Alternatives:
   d = 100 (about 75 s per seed and arm for P's pilot alone, by the probe's frequencies),
   or an identity-mass version.
10. **MALA arms MJ5 and MCF** on four cells; G does not apply to MALA.
11. **The recommended jitter width** is the widest default candidate. Alternative: the
    cheapest.
12. **Seeds 9100 onward**, unused by any earlier experiment of the package.

**Files of the design probes** (to be committed beside this file): `retune.py`,
`unequal_tune.py`, `probe_tune.py`, `probe100.py`, `probe_rot.py`, `targets.py`,
`guard.py`, `evalguard.py`, `evalunequal.py`, `post.py`, `trig.py`, `phases.py`,
`jpred.py`, `model.py`, `spreads.py`, `ineffpower.py`, `costs.py`, and their outputs
`tunings_d10.jsonl`, `tunings_unequal.jsonl`, `probe_tune.out`, `evalguard.out`,
`evalunequal.out`, `post.out`, `trig.out`, `jpred.out`, `spreads.out`, `ineffpower.out`,
`costs.out`, `phases.out`, `probe_rot.out`, `probe100.out`, `model.out`, `setup.out`.
