"""Detection, false alarms and cost of each candidate rule, from i102_q2_rules.py's rows.

Question and design: see i102_q2_rules.py. Detection is scored on runs whose estimate was
in fact wrong (|z| > 5); the criterion's detection set is the issue's family at d = 10
and 32 (s = 6 and 12). False alarms are scored on the runs of the false-alarm cells whose
estimate was right; runs there with a wrong estimate are listed separately. Cost is the
probe's log-density evaluations over the pilot's 4 (n_steps + 1), and over the default
run's total (pilot, 100 pilot pairs and 400 replicates). Intervals: Clopper-Pearson 95%.

Result (i102_q2_analysis_main.out). R0, the issue's rule, and the existing R-hat warning
detect 0 of 62 wrong runs; the R-hat warning fires instead on 17 of the 440 right runs of
the false-alarm cells. R1 settings within 25% of the pilot detect up to 62 of 62, but
their false-alarm statistics reach the wrong runs' range on examples.pumps() from its
default start (up to 11.1 for "end", m = 8, n = n_steps/8) and on the 50-dimensional
Gaussian of condition number 100 from a tight start (up to 14.2 at n = n_steps/16).
i102_q2_analysis_main_fixed.out evaluates "end", m = 8, n = n_steps/8, c = 12 on the main
seeds (62 of 62 detected, no false alarm in 440), and i102_q2_analysis_confirm.out the
same setting on the confirm seeds: 62 of 62 detected, 2 false alarms in 440, both on
pumps from its default start, so it fails the criterion.

Reads the rows q2_rows_<tag>.npy that i102_q2_rules.py writes to the current folder;
runs in seconds.

Run, for i102_q2_analysis_main.out, _main_fixed.out and _confirm.out in turn:
  python notes/experiments/i102_q2_analysis.py main
  python notes/experiments/i102_q2_analysis.py main end 8 8 12
  python notes/experiments/i102_q2_analysis.py confirm end 8 8 12
The last four arguments evaluate one fixed setting: statistic, m, divisor of n_steps, c.
"""

import sys

import numpy as np
from scipy.stats import binomtest

MS = (4, 8, 16, 32)
DIVS = (64, 32, 16, 8, 4, 2)
STATS = ("end", "half", "end_dens", "half_dens")
CRIT = ("bimodal d=10 s=6", "bimodal d=10 s=12", "bimodal d=32 s=6", "bimodal d=32 s=12")


def ci(k, n):
    if n == 0:
        return "n=0"
    lo, hi = binomtest(k, n).proportion_ci()
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def load(tag):
    return list(np.load(f"q2_rows_{tag}.npy", allow_pickle=True))


def r1(r, stat, m, div):
    return r["stats"][MS.index(m), DIVS.index(div), STATS.index(stat)]


def r1_cost(r, m, div):
    return m * (r["n_steps"] // div) / r["pilot_evals"], m * (r["n_steps"] // div) / r["run_evals"]


def report_setting(rows, stat, m, div, c):
    print(f"\nsetting R1 {stat}, m={m}, n=n_steps/{div}, c={c}: per cell")
    print(
        "  cell | runs wrong | fires on wrong | fires on right | probe chains on q's other side at n (mean) | cost/pilot cost/run"
    )
    for cell in dict.fromkeys(r["cell"] for r in rows):
        c_rows = [r for r in rows if r["cell"] == cell]
        wrong = [r for r in c_rows if r["wrong"] is True]
        right = [r for r in c_rows if r["wrong"] is False]
        fw = sum(r1(r, stat, m, div) > c for r in wrong)
        fr = sum(r1(r, stat, m, div) > c for r in right)
        other = np.mean([r["other"][MS.index(m), DIVS.index(div)] for r in c_rows])
        cp, cr = np.median([r1_cost(r, m, div) for r in c_rows], axis=0)
        print(
            f"  {cell:30s} | {len(wrong):2d}/{len(c_rows)} | {ci(fw, len(wrong)):18s} | {ci(fr, len(right)):18s} |"
            f" {other:5.2f} | {cp:.3f} {cr:.4f}"
        )


def main(tag, fixed=None):
    rows = load(tag)
    det = [r for r in rows if r["cell"] in CRIT and r["wrong"] is True]
    det_all = [r for r in rows if r["kind"] == "detect" and r["wrong"] is True]
    fa = [r for r in rows if r["kind"] == "fa" and r["wrong"] is False]
    fa_wrong = [r for r in rows if r["kind"] == "fa" and r["wrong"] is True]
    print(
        f"tag {tag}: {len(rows)} runs; criterion detection set: {len(det)} wrong runs of "
        f"{sum(r['cell'] in CRIT for r in rows)}; all detection cells: {len(det_all)} wrong of "
        f"{sum(r['kind'] == 'detect' for r in rows)}; false-alarm set: {len(fa)} right runs of "
        f"{sum(r['kind'] == 'fa' for r in rows)}"
    )
    print(
        "false-alarm cells with a wrong estimate (excluded from the false-alarm set):",
        sorted({(r["cell"], r["seed"], round(r["z"], 1)) for r in fa_wrong}) or "none",
    )
    print(
        "\nper cell: runs wrong, |z| median, pilot ends on q-side +1 (right mode) count, existing R-hat warning fires,"
        " median pilot evals, median run evals, median k, max meeting time"
    )
    for cell in dict.fromkeys(r["cell"] for r in rows):
        c_rows = [r for r in rows if r["cell"] == cell]
        print(
            f"  {cell:30s} wrong {sum(r['wrong'] is True for r in c_rows):2d}/{len(c_rows)} (raised {sum(r['wrong'] is None for r in c_rows)}), |z| median "
            f"{np.nanmedian([abs(r['z']) for r in c_rows]):9.1f}, q right {sum(r['q_side'] > 0 for r in c_rows):2d}, "
            f"R-hat warn {sum(r['disagree'] for r in c_rows):2d}, pilot {np.median([r['pilot_evals'] for r in c_rows]):7.0f}, "
            f"run {np.median([r['run_evals'] for r in c_rows]):9.0f}, k {np.median([r['k'] for r in c_rows]):5.0f}, thinned {sum(r['thin'] > 1 for r in c_rows)}, "
            f"tau max {max(r['tau_max'] for r in c_rows):6.0f}"
        )
    if fixed:
        stat, m, div, c = fixed
        f_det = sum(r1(r, stat, m, div) > c for r in det)
        f_fa = sum(r1(r, stat, m, div) > c for r in fa)
        cost = max(r1_cost(r, m, div)[0] for r in rows)
        print(
            f"\nFIXED SETTING R1 {stat} m={m} n=n_steps/{div} c={c}: detection {ci(f_det, len(det))},"
            f" false alarms {ci(f_fa, len(fa))}, cost at most {cost:.3f} of the pilot"
        )
        report_setting(rows, stat, m, div, c)
        return

    print(
        "\nexisting R-hat warning: detection",
        ci(sum(r["disagree"] for r in det), len(det)),
        "; false alarms",
        ci(sum(r["disagree"] for r in fa), len(fa)),
    )
    for name, cost in (("r0a", 100), ("r0b", 2000), ("r0a_dens", 100), ("r0b_dens", 2000)):
        top_fa = max(r[name] for r in fa)
        print(
            f"R0 {name}: largest statistic on wrong runs {max(r[name] for r in det):.1f} (fires at c=0: "
            f"{ci(sum(r[name] > 0 for r in det), len(det))}); largest on false-alarm set {top_fa:.1f}; "
            f"cost {cost} evaluations = {np.median([cost / r['pilot_evals'] for r in det]):.3f} of the pilot (median)"
        )

    print(
        "\nR1 settings: stat m n | cost/pilot (max over runs) | largest statistic on the false-alarm set (cell) |"
        " detection at c = that + 1 | detection at c = 2, 5, 10 | false alarms at c = 2, 5, 10"
    )
    best = []
    for stat in STATS:
        for m in MS:
            for div in DIVS:
                cost = max(r1_cost(r, m, div)[0] for r in rows)
                vals = [r1(r, stat, m, div) for r in fa]
                i = int(np.argmax(vals))
                top = vals[i]
                c_star = np.floor(top) + 1 if np.isfinite(top) else 0.0
                d_star = sum(r1(r, stat, m, div) > c_star for r in det)
                dets = [sum(r1(r, stat, m, div) > c for r in det) for c in (2, 5, 10)]
                fas = [sum(r1(r, stat, m, div) > c for r in fa) for c in (2, 5, 10)]
                print(
                    f"  {stat:9s} {m:2d} /{div:<2d} | {cost:6.3f} | {top:7.1f} ({fa[i]['cell']}, {fa[i]['seed']}) | "
                    f"{ci(d_star, len(det)):18s} | {dets} | {fas}"
                )
                best.append((cost <= 0.5, d_star, -cost, stat, m, div, c_star))
    best.sort(reverse=True)
    print(
        "\nbest R1 settings within 50% of the pilot, by detection at c = (largest false-alarm statistic) + 1:"
    )
    for _ok, d_star, neg_cost, stat, m, div, c_star in [b for b in best if b[0]][:8]:
        print(
            f"  {stat} m={m} n=n_steps/{div} c={c_star:g}: detection {ci(d_star, len(det))}, cost {-neg_cost:.3f}"
        )
    # Per-cell tables for a few informative settings.
    for stat, m, div, c in ((("half_dens", 16, 16, 5)), ("half", 16, 16, 5), ("end", 8, 8, 5)):
        report_setting(rows, stat, m, div, c)
    # Where the false alarms are, for the statistics without the density condition.
    print(
        "\nlargest R1 statistic per false-alarm cell (over its right runs), at m=16, n=n_steps/16, each stat:"
    )
    for cell in dict.fromkeys(r["cell"] for r in fa):
        c_rows = [r for r in fa if r["cell"] == cell]
        print(
            f"  {cell:30s}",
            "  ".join(f"{s}={max(r1(r, s, 16, 16) for r in c_rows):7.1f}" for s in STATS),
        )
    extra = [r for r in rows if r["kind"] == "extra"]
    if extra:
        print(
            "\nextra (funnel, outside the criterion): wrong",
            sum(r["wrong"] is True for r in extra),
            "of",
            len(extra),
            "; R1 half_dens m=16 n/16 largest",
            round(max(r1(r, "half_dens", 16, 16) for r in extra), 1),
        )


if __name__ == "__main__":
    tag = sys.argv[1]
    fixed = None
    if len(sys.argv) > 2:
        fixed = (sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    main(tag, fixed)
