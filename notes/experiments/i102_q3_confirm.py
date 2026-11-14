"""Confirmation, on fresh seeds, of one coverage rule fixed in advance (issue #102).

Question. Does the rule chosen in i102_q2_pooled.py pass the criterion of
i102_q2_rules.py on fresh seeds? The rule, fixed and published before this run: the R1
"end" statistic (the largest log weight log pi - log q at the last state of m = 8
random-walk probe chains of n = n_steps/8 steps, minus the pilot's largest log weight);
it fires when the statistic exceeds c = 10, and is silent when the existing R-hat warning
fires on the pilot (largest R-hat > sqrt 2).

Criterion (fixed before this run). Detection of at least 90% of the wrong runs (|z| > 5)
on the criterion cells (the issue's family, d in {10, 32}, s in {6, 12}); no false alarm
on the right runs of the 22 false-alarm cells (a run there whose estimate is wrong is
listed separately); cost at most 50% of the pilot's evaluations.

Design. i102_q2_rules.one for each (cell, seed), unchanged: the default run with 400
replicates (thinned when its automatic k would store more than 0.5 GB), the pilot, the
probes. Seeds 40 to 59 for every detection cell (the criterion cells for the verdict, the
others for information), 300 to 319 for the 22 false-alarm cells and the funnel (for
information). At most two jobs at a time; a job still running after 540 s is stopped and
listed.

Result (i102_q3_confirm.out). The criterion as written passes on these seeds: 67 of 67
wrong criterion runs detected (95% interval [0.95, 1.00]); no false alarm in 440 right
runs ([0.00, 0.01]); cost 0.25 of the pilot. The smallest statistic on a wrong run is
15.4 and the largest on a right run with R-hat silent 5.1; the silencing carries weight,
since the 14 right runs it silences, all on examples.pumps() from its default start,
reach 21.1. Four jobs ran past 540 s, two of them criterion runs (d = 10, s = 12, seeds
44 and 47) whose truth is unknown; the rule fires on both. The existing R-hat warning
fired on none of the 67 wrong runs.

Narrowed by the adversarial review, which ran this job function unchanged on fresh seeds
(i102_review_runner.py and i102_review_full_list.py; their outputs are named below):
  * It recounts these figures from the saved rows (i102_review_recount.out, where the
    95% interval of 0 in 440 is [0, 0.0083]) and reproduces eight saved rows exactly
    (i102_review_repro.out).
  * The rule is not free of false alarms on its own cells: on examples.pumps() from its
    default start it fires on 8 of 1,200 fresh runs (2 of 200 in
    i102_review_pumps_rule.out, 6 of 1,000 in i102_review_pumps_rule2.out), whose R-hat,
    1.269 to 1.411, sits just under sqrt 2; their default runs all have |z| < 3
    (i102_review_pumps_full1.out, i102_review_pumps_full2.out). At that rate, 20 runs
    rarely show one.
  * It fires on right runs of single-mode Gaussians of condition number 10^4 from a tight
    start: 3 of 20 at d = 20 (statistics 10.4 to 15.0, i102_review_fa_full1.out) and 9 of
    10 at d = 50 (statistics up to 40.1, i102_review_g50b_rule.out, whose default runs in
    i102_review_g50b_full.out are all right, with standard errors up to 130). At
    condition number 100, the cells' hardest covered case, it fires on 0 of 100
    (i102_review_g50_rule.out).
  * Its detection depends on the width of the initial distribution. On the criterion's
    target at s = 6: from N(0, 400 I) it detects 4 of 5 wrong runs at d = 10 and 10 of 10
    at d = 32; from N(0, 1600 I), 0 of 6 at d = 10 and 1 of 8 at d = 32; from
    N(0, 10000 I), 0 of 1 at d = 10
    (i102_review_family2.out, i102_review_family3.out). The probes are random walks of a
    fixed scale, and from a wider start they have not reached a mode after n_steps/8 steps.
  * At smaller separations and dimensions of the same family (d = 10 with s = 4 and 5,
    d = 5 and 6 with s = 6) it detects the 6 wrong runs and fires on 10 of the 30 right
    runs, 4 + 1 + 2 + 3 of 10 + 5 + 6 + 9 (i102_review_family1.out).
  * Heavier tails (Student-t with 2.5 to 4 degrees of freedom), correlations of 0.99 and
    0.999 and a skewed (log-gamma) target gave no fire over 20 seeds each
    (i102_review_fa_rule1.out).
  * The 540 s cap was not in the published rule and criterion, and the threshold of 10
    and the silencing were chosen after the first rule failed: this run is a holdout for
    these cells and this start width only.

Ran against commit c961089: about an hour on two workers, plus the four jobs stopped at
540 s. Rerun for issue #85 with the same rows and the same four jobs stopped: about two
hours of job time on one core, one job per process, at commit 32ae70b; the saved rows were
then read and the tables printed at the commit that the first line of its output records,
whose package source is that of 32ae70b.

Run: python notes/experiments/i102_q3_confirm.py
It writes one file per job to notes/experiments/_rows/rows_q3/ and all rows to
notes/experiments/_rows/q3_rows.npy, which i102_review_recount.py and i102_review_repro.py
read there; git ignores the folder.
"""

import os
import pickle
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from i85_setup import setup
from i102_common import ROWS
from i102_q2_analysis import CRIT, ci, r1, r1_cost
from i102_q2_rules import CELLS, one, path_of, saved

STAT, M, DIV, C = "end", 8, 8, 10.0
FOLDER = ROWS / "rows_q3"
CAP = 540  # seconds: a job still running then is stopped and recorded as timed out


def _alarm(signum, frame):
    raise TimeoutError


def capped(args):
    folder, job = args
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(CAP)
    try:
        return saved(args)
    except TimeoutError:
        row = {
            "cell": job[0],
            "seed": job[1],
            "kind": CELLS[job[0]][0],
            "timed_out": True,
            "secs": CAP,
            "rss_mb": 0,
        }
        with open(path_of(folder, job), "wb") as f:
            pickle.dump(row, f)
        return row
    finally:
        signal.alarm(0)


def fires(r):
    return bool(r1(r, STAT, M, DIV) > C and not r["disagree"])


def main():
    t0 = time.time()
    print(setup())
    jobs = [
        (name, (40 if CELLS[name][0] == "detect" else 300) + i)
        for name in CELLS
        for i in range(20)
    ]
    jobs.sort(key=lambda j: -CELLS[j[0]][3])
    os.makedirs(FOLDER, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(path_of(FOLDER, j))]
    with Pool(2, maxtasksperchild=10) as pool:
        for i, row in enumerate(pool.imap_unordered(capped, [(FOLDER, j) for j in todo])):
            print(
                f"done {i + 1}/{len(todo)} {row['cell']} {row['seed']} {row['secs']:.0f}s rss {row['rss_mb']:.0f}MB",
                file=sys.stderr,
                flush=True,
            )
    order = list(CELLS)
    rows = sorted(
        (pickle.loads(Path(path_of(FOLDER, j)).read_bytes()) for j in jobs),
        key=lambda r: (order.index(r["cell"]), r["seed"]),
    )
    np.save(ROWS / "q3_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
    timed_out = [r for r in rows if r.get("timed_out")]
    print(
        f"jobs stopped at {CAP} s (rule not evaluated): {[(r['cell'], r['seed']) for r in timed_out] or 'none'}"
    )
    for r in timed_out:  # the rule alone (pilot and probes), without the default run
        probe_only = one((r["cell"], r["seed"], True))
        print(
            f"  {r['cell']} seed {r['seed']}: truth unknown; R-hat warn {int(probe_only['disagree'])}, "
            f"statistic {r1(probe_only, STAT, M, DIV):.1f}, rule fires {int(fires(probe_only))}"
        )
    rows = [r for r in rows if not r.get("timed_out")]

    print(
        "raw table: cell | seed | n_steps pilot-evals run-evals k tau_max thin | estimate se z wrong(E=raised) |"
        " R-hat-warn | statistic (end, m=8, n=n_steps/8) | fires | probe cost/pilot"
    )
    for r in rows:
        w = "E" if r["wrong"] is None else int(r["wrong"])
        print(
            f"{r['cell']:30s} | {r['seed']:3d} | {r['n_steps']:6d} {r['pilot_evals']:6d} {r['run_evals']:8d} "
            f"{r['k']:5d} {r['tau_max']:8.0f} {r['thin']:3d} | {r['value']:9.4f} {r['se']:8.4f} {r['z']:10.2f} {w} | "
            f"{int(r['disagree'])} | {r1(r, STAT, M, DIV):8.1f} | {int(fires(r))} | {r1_cost(r, M, DIV)[0]:.3f}"
        )

    det = [r for r in rows if r["cell"] in CRIT and r["wrong"] is True]
    crit_right = [r for r in rows if r["cell"] in CRIT and r["wrong"] is False]
    fa = [r for r in rows if r["kind"] == "fa" and r["wrong"] is False]
    fa_wrong = [r for r in rows if r["kind"] == "fa" and r["wrong"] is True]
    fa_raised = [r for r in rows if r["kind"] == "fa" and r["wrong"] is None]
    n_det, n_fa = sum(map(fires, det)), sum(map(fires, fa))
    cost = max(r1_cost(r, M, DIV)[0] for r in rows)
    print(
        f"\nVERDICT INPUTS (rule: {STAT}, m={M}, n=n_steps/{DIV}, c={C:g}, silent when R-hat > sqrt 2)"
    )
    print(f"  detection on criterion cells: {ci(n_det, len(det))}   (criterion >= 0.90)")
    print(f"  false alarms on covered runs: {ci(n_fa, len(fa))}   (criterion 0)")
    print(f"  cost: at most {cost:.3f} of the pilot's evaluations   (criterion <= 0.5)")
    passed = len(det) > 0 and n_det / len(det) >= 0.9 and n_fa == 0 and cost <= 0.5
    print(f"  PASS: {passed}")
    # Margin, on the statistic the rule thresholds (runs silenced by R-hat are listed apart).
    wrong_stats = [r1(r, STAT, M, DIV) for r in det if not r["disagree"]]
    cov_stats = [(r1(r, STAT, M, DIV), r["cell"], r["seed"]) for r in fa if not r["disagree"]]
    top = max(cov_stats)
    print(
        f"  smallest statistic on a wrong criterion run (R-hat silent): {min(wrong_stats):.1f};"
        f" largest on a covered run with R-hat silent: {top[0]:.1f} ({top[1]}, {top[2]});"
        f" margin {min(wrong_stats) - top[0]:.1f}"
    )
    print(
        f"  covered runs silenced by R-hat: {sum(r['disagree'] for r in fa)}; their largest statistic "
        f"{max([r1(r, STAT, M, DIV) for r in fa if r['disagree']], default=float('nan')):.1f}"
    )
    print(f"  wrong criterion runs silenced by R-hat: {sum(r['disagree'] for r in det)}")
    print("\nruns on which the rule and the truth disagree:")
    for r in det:
        if not fires(r):
            print(
                f"  missed wrong run: {r['cell']} seed {r['seed']} statistic {r1(r, STAT, M, DIV):.1f} R-hat warn {int(r['disagree'])}"
            )
    for r in fa:
        if fires(r):
            print(
                f"  false alarm: {r['cell']} seed {r['seed']} statistic {r1(r, STAT, M, DIV):.1f}"
            )
    for r in crit_right:
        if fires(r):
            print(
                f"  fires on a right criterion run (not scored): {r['cell']} seed {r['seed']} statistic {r1(r, STAT, M, DIV):.1f}"
            )
    print(
        f"  covered-cell runs with a wrong estimate (scored neither way): "
        f"{[(r['cell'], r['seed'], round(r['z'], 1), int(fires(r))) for r in fa_wrong] or 'none'}"
    )
    print(
        f"  covered-cell runs that raised: {[(r['cell'], r['seed']) for r in fa_raised] or 'none'}"
    )

    print(
        "\nper cell: wrong / raised / runs | fires on wrong | fires on right | R-hat warn | thinned | median cost/pilot, cost/run"
    )
    for cell in dict.fromkeys(r["cell"] for r in rows):
        c_rows = [r for r in rows if r["cell"] == cell]
        wrong = [r for r in c_rows if r["wrong"] is True]
        right = [r for r in c_rows if r["wrong"] is False]
        tag = "" if cell in CRIT or CELLS[cell][0] == "fa" else "  (information only)"
        cp, cr = np.median([r1_cost(r, M, DIV) for r in c_rows], axis=0)
        print(
            f"  {cell:30s} {len(wrong):2d}/{sum(r['wrong'] is None for r in c_rows):2d}/{len(c_rows)} | "
            f"{ci(sum(map(fires, wrong)), len(wrong)):18s} | {ci(sum(map(fires, right)), len(right)):18s} | "
            f"{sum(r['disagree'] for r in c_rows):2d} | {sum(r['thin'] > 1 for r in c_rows):2d} | {cp:.3f} {cr:.4f}{tag}"
        )
    print(
        f"\nsum of job seconds {sum(r['secs'] for r in rows):.0f}; wall time {time.time() - t0:.0f} s (shared machine)"
    )


if __name__ == "__main__":
    main()
