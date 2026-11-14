# Second review of the pre-registration x91, after its revision (2026-11-14)

**Verdict: sound to post after the listed changes.** Every finding of the first review
is resolved as section 14 claims. The revision
introduces two defects that can change a verdict (N1, N2) and four smaller ones (N3 to
N6). All six are fixed by sentences; none needs a new arm, cell or probe. Nothing was
run for this pass: it reads the revised text and the new probe outputs (`kpower.out`,
`heldout.out`, `scan_heldout.out`, `heldout_review.out`, `model.out`, the new
`guard.py`). Line numbers are those of the revised `PREREGISTRATION.md`.

## What was checked and holds

- B1: K, L and T exist; section 5's counts and powers are the recomputed ones;
  `kpower.out` is right (K fails at 0.036, 0.12, 0.45, 0.82; T at 0.013, 0.35, 0.85).
- B2: a full run is defined (lines 162 to 165); V is void below half the seeds.
- B3: C2 and C4 are in tier 1; rule 4 covers the cells that can still be cut.
- I1: P.2 and CF.2 on four cells, judged against each other, deciding no default; the
  wording of section 8 now matches section 12. `model.out` gives the stated ratios.
- I2: the fallback keeps the step (`guard.py`), B's cells decide G's scope one by one,
  and A', B' were chosen from P's factors alone (`scan_heldout.out`); the cells I had
  suggested do not resonate (`heldout_review.out`), so replacing them is right.
- I3 to I6 and the minors: as section 14 says. The Z confirmation (same sign, above 2
  on fresh seeds) and the difference score are sound; T's nine seed blocks are disjoint
  from every other block.

## New findings that can change a verdict

### N1. Whether L or M applies in a resonant cell depends on five seeds of P (lines 330, 338, 169 to 170)

M applies "where P leaves no pair unmet at any seed" and L "where P has an unmet pair".
In the resonant cells of A and A', P runs at 5 seeds, plus those where G keeps the
step. At #137's rates (0.21, 0.49, 0.09 of tunings fail at the cap) P shows no failing
seed in 5 with probability 0.31, 0.03 and 0.62. In that case the cell is scored by M
against a P whose medians are in the thousands, which every arm passes, and L, the
criterion added for these cells, is switched off. At d = 10, `n_steps=10` this is the
likelier outcome. Smallest change: L applies on the nine cells of T by name, whatever
P shows there, and M on every other cell where P has no unmet pair.

### N2. "Unmet" now has two meanings, and M, L and the full run do not say which (lines 163, 309 to 310, 317 to 322, 330, 338)

U is judged at the cap and at 2 x 10^5 gradients, and line 309 defines "unmet at the
gradient budget". Lines 163, 330 and 338 say only "unmet". The readings differ where
meeting is slow but certain: on D1 the model puts P's median at 1,200 to 4,500
iterations of about 3.8 gradients, and on B with `n_steps` 3 and 10 P's median is 400
to 2,500 iterations of 5.7 and 19 gradients, so some of P's pairs can pass 2 x 10^5
gradients with none unmet at the cap. Under one reading M applies on those cells and
under the other it does not. Smallest change: say "unmet at the cap" in the three
places. Then also say what U at the gradient budget means for an arm on a cell where
P itself fails it (D1 is the likely case): judged, or reported.

## New findings that need a sentence

### N3. T refuses G on a proxy, and its threshold was set with G's rate in view (lines 326 to 329, 391 to 393, 589 to 591)

- Rule 1 says G is refused if it fails U or K "with T on their cells"; section 9 reads
  that as "T alone refuses" (line 565). Say so in rule 1.
- T is predicted to fail where K is not (d = 10, `n_steps=10` and A' with 3: T fails
  with 0.85, K with 0.12). G would then be refused with no seed that raised. The link
  from "factor above 0.9998" to "k cannot be chosen" comes from #137's validation on
  `gaussian(2)` and `(10)`. Add a check inside the experiment: over the registered
  pilot records of the T cells, tabulate factor above 0.9998 against `k` not chosen,
  and state the agreement below which T is reported, not judged.
- Line 589 says G's 1% was set "without data of any arm". The 100 probe tunings after
  the guard (2 of 100 at d = 10, `n_steps=10`; 2 and 11 of 100 on A') were known when
  it was set, so T's outcome for G was predictable at that moment. Section 10 should
  say that, and that A' is held out from G's rule but not from the prediction of its
  result.

### N4. A full run is still not bounded (lines 630 to 631)

The bound on `k` bounds `m`, not the meeting time: `run(100)` with `max_iter=None`
runs every replicate until it meets. "No run can grow without bound" is not true as
written. Either drop the sentence, or fix a rule now: a record whose replicates are
not all met after a stated number of iterations is abandoned and recorded as having
no replicates.

### N5. A seed without replicates is seen by no criterion (lines 162 to 165, 323 to 325, 345 to 353)

A seed of a full-run cell with one unmet pilot pair, or with `k` above 20,000 and none
unmet, holds K (`k` is chosen), barely moves U and M, and is dropped from V and Z.
Dropping an arm's slowest seeds from V favors that arm. Smallest change: in full-run
cells a seed without replicates counts as a seed that fails K.

### N6. J2's prediction on L sits on the threshold of the registration's own model (lines 448 to 449, 491, 504)

With the model of section 9 and the median factors of `jpred.out`, J2 against J5 gives
(20 + 6/0.083)/(20 + 6/0.546) = 3.0 at d = 10, `n_steps=3` and 2.8 at B's default;
the predicted medians (60 to 140 against 30 to 45) allow up to 4.7. L's factor is 3.
"L holds" for J2 on those cells is a coin flip, not a prediction; state its
probability or withdraw it, as was done for the jitters on D1. The consequence is
small: J2 matters only if J3 and J5 are not default candidates.
