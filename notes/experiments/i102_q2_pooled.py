"""Both seed sets pooled: does any probe setting within 50% of the pilot separate the runs?

Question. After the setting chosen on the main seeds failed on the confirm seeds, is there
a fixed setting of R1 (statistic, m, n) and a threshold c that would have passed on all
880 false-alarm runs and 124 wrong runs together? This is chosen after seeing the data:
a setting found here needs fresh seeds. Also shown, as chosen after seeing the data too:
R1 silenced when the existing R-hat warning already fires ("R1 unless R-hat").

Design. Reads q2_rows_main.npy and q2_rows_confirm.npy, which i102_q2_rules.py writes to
the current folder. For each setting within 50% of the pilot: the smallest statistic
over the wrong runs of the criterion cells, the largest over the false-alarm runs, and the
detection at c = (largest false-alarm statistic) + 1, rounded down; then the same with
the runs silenced by R-hat left out of the false alarms and of the detections.

Result (i102_q2_pooled.out). Without the silencing, no setting detects all 124 wrong
runs at that threshold; the best, "end" or "end_dens" with m = 32 and n = n_steps/16
(cost 0.5), detects 122, with the largest false-alarm statistic 14.2 against a smallest
wrong-run statistic of 14.4. The false-alarm tails come from examples.pumps() from its
default start (up to 68.9) and the 50-dimensional Gaussian of condition number 100 from
a tight start (up to 16.9). Silenced by R-hat, "end" with m = 8 and n = n_steps/8 (cost
0.25) detects 124 of 124 at c = 10, with the largest false-alarm statistic 9.3 against a
smallest wrong-run statistic of 13.8, a thin margin. That rule went to fresh seeds in
i102_q3_confirm.py. The silenced rule's detection of 124 of 124 also shows that the
existing R-hat warning fired on none of the 124 wrong runs.

Runs in seconds.

Run: python notes/experiments/i102_q2_pooled.py
"""

import numpy as np
from i102_q2_analysis import CRIT, DIVS, MS, STATS, ci, load, r1, r1_cost

rows = load("main") + load("confirm")
det = [r for r in rows if r["cell"] in CRIT and r["wrong"] is True]
fa = [r for r in rows if r["kind"] == "fa" and r["wrong"] is False]
print(f"pooled: {len(det)} wrong runs in the criterion cells, {len(fa)} false-alarm runs")
print(
    "stat m n | cost/pilot | min over wrong | max over FA (cell, seed) | detection at c = floor(max FA)+1 |"
    " same for 'R1 unless R-hat': max FA (cell), detection"
)
for stat in STATS:
    for m in MS:
        for div in DIVS:
            cost = max(r1_cost(r, m, div)[0] for r in rows)
            if cost > 0.5:
                continue
            dv = [r1(r, stat, m, div) for r in det]
            fv = [r1(r, stat, m, div) for r in fa]
            i = int(np.argmax(fv))
            c = np.floor(fv[i]) + 1
            fv2 = [r1(r, stat, m, div) if not r["disagree"] else -np.inf for r in fa]
            j = int(np.argmax(fv2))
            c2 = np.floor(fv2[j]) + 1
            d2 = sum(r1(r, stat, m, div) > c2 and not r["disagree"] for r in det)
            print(
                f"{stat:9s} {m:2d} /{div:<2d} | {cost:5.3f} | {min(dv):7.1f} | {fv[i]:6.1f} ({fa[i]['cell']}, {fa[i]['seed']}) | "
                f"{ci(sum(v > c for v in dv), len(det)):18s} | {fv2[j]:6.1f} ({fa[j]['cell']}) {ci(d2, len(det))}"
            )
