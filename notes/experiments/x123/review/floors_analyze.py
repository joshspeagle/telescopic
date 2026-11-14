"""Review of x123: can a floor of another form separate group B from the harmful families?

Question: RESULT.md reports that every relative floor rho |g o v| that silences part of
group B also loses warnings on harmful families. Does a floor on another statistic of a
line, or a rule that asks several lines to agree, separate them?

Design: the lines that floors_collect.py wrote (fresh seeds 520000 to 541299), and the
registered classes of harm_summary.out. Eight statistics of a line that warns under (N):
the gap over the tolerance, over h |log_prob|, over |g o v|, over |g.v|, over the
difference quotient, over the larger of the last two, over h (nats over u, the statistic
of (H)), and over h sqrt|log_prob|. Four rules: the statistic must exceed the threshold
on any line, on two lines, on half the lines, on all lines. For each B family that (N)
warns on, the threshold is the one that silences 95% of its checks, and the cost is the
number of harmful families, among those that (N) detects in at least 99% of checks,
whose detection falls below 0.975. The last table gives, for each pair of a B family and
a harmful family, the best Youden index over every statistic, rule and threshold.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/floors_analyze.py floors.pkl > floors_analyze.out

Result: see floors_analyze.out and REVIEW.md, the checks that found nothing. Silencing
the 8 warned B families at once loses at best 27 of 46 harmful families; for 6 of the 8,
some harmful family with a tight start is smaller on every statistic (best index 0.00).
Only the Jacobi solver at 1e-5 separates: a floor of 8.4e-5 nats over u loses none.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

with open(sys.argv[1], "rb") as file:
    data = pickle.load(file)
cls = {}
with open(Path(__file__).resolve().parent.parent / "harm_summary.out") as summary:
    for line in summary:
        if (
            line.startswith("RESULT | ")
            and "by affine" not in line
            and "validation" not in line
            and not line.startswith("RESULT | group")
        ):
            p = [x.strip() for x in line.split("|")]
            cls[(p[1], p[2])] = p[6]
cls[("C", "rotated cond 1e4 d=2000, matched: sign along the widest axis")] = "harmful"
B = [k for k in data if k[0] == "B"]
Hf = [k for k in data if k[0] in "CF" and cls[k] == "harmful"]
tiny = 1e-300


def stats(c):
    gap, tol, _, _, _, L, norm, h, D, Gv = c.T
    warn = gap > tol
    s = {
        "gap/tolerance (a larger S)": gap / np.maximum(tol, tiny),
        "gap/(h |log_prob|)": gap / np.maximum(h * L, tiny),
        "gap/|g o v| (registered rho)": gap / np.maximum(norm, tiny),
        "gap/|g.v|": gap / np.maximum(np.abs(Gv), tiny),
        "gap/|difference quotient|": gap / np.maximum(np.abs(D), tiny),
        "gap/max(|g.v|,|D|)": gap / np.maximum(np.maximum(np.abs(Gv), np.abs(D)), tiny),
        "gap/h (nats over u, the floor of H)": gap / h,
        "gap/(h sqrt|log_prob|)": gap / np.maximum(h * np.sqrt(L), tiny),
    }
    return warn, s


AGG = {"any line": 0, "two lines": 1, "half the lines": 3, "all lines": -1}


def score(c, name, k):
    warn, s = stats(c)
    v = np.where(warn, s[name], -np.inf)
    v = np.sort(v)[::-1]
    if len(v) == 0:
        return -np.inf
    return v[k] if (k >= 0 and k < len(v)) else v[-1]


names = list(stats(data[B[0]][0])[1])
S = {}
for k in data:
    for nm in names:
        for an, ak in AGG.items():
            S[(k, nm, an)] = np.array([score(c, nm, ak) for c in data[k]])
baseN = {k: np.mean(S[(k, names[0], "any line")] > -np.inf) for k in data}
print("fresh seeds, (N) rate: B", {k[1][:28]: baseN[k] for k in B})
print(
    "harmful families used:",
    len(Hf),
    "of which (N) rate >= 0.99:",
    sum(baseN[k] >= 0.99 for k in Hf),
)
Hd = [k for k in Hf if baseN[k] >= 0.99]
Bw = [k for k in B if baseN[k] > 0.5]
print(
    "\nFor each B family that (N) warns on: the floor (statistic, rule) that silences it in >= 95% of checks at the least cost."
)
print(
    f"cost = harmful families (of {len(Hd)} that (N) detects) whose detection falls below 0.975, and the lowest detection"
)
for b in Bw:
    best = None
    for nm in names:
        for an in AGG:
            sb = S[(b, nm, an)]
            thr = np.quantile(sb[np.isfinite(sb)], 0.95) if np.isfinite(sb).any() else -np.inf
            det = {f: np.mean(S[(f, nm, an)] > thr) for f in Hd}
            lost = [f for f in Hd if det[f] < 0.975]
            row = (len(lost), -min(det.values()), nm, an, thr, min(det, key=det.get))
            if best is None or row[:2] < best[:2]:
                best = row
    print(
        f"B {b[1][:52]:<52} best: {best[2]} / {best[3]}, threshold {best[4]:.3g}: loses {best[0]} families, lowest detection {-best[1]:.2f} on '{best[5][1][:50]}'"
    )
print(
    "\nSilencing ALL of those B families at once (>= 95% of checks of each): cost per statistic and rule"
)
for nm in names:
    for an in AGG:
        thr = max(np.quantile(S[(b, nm, an)][np.isfinite(S[(b, nm, an)])], 0.95) for b in Bw)
        det = {f: np.mean(S[(f, nm, an)] > thr) for f in Hd}
        lost = sorted((det[f], f[1][:40]) for f in Hd if det[f] < 0.975)
        print(
            f"  {nm:<36} {an:<14} threshold {thr:9.3g}: loses {len(lost):>2} of {len(Hd)}; zero detection on {sum(1 for d_, _ in lost if d_ == 0)}; lowest {lost[:2] if lost else '-'}"
        )
print(
    "\nSilencing all B families except the two worst-separated (grad + 0.02; logistic from sd 10):"
)
Bsub = [b for b in Bw if "+ 0.02" not in b[1] and "start sd 10" not in b[1]]
for nm in names:
    for an in ("any line", "two lines", "all lines"):
        thr = max(np.quantile(S[(b, nm, an)][np.isfinite(S[(b, nm, an)])], 0.95) for b in Bsub)
        det = {f: np.mean(S[(f, nm, an)] > thr) for f in Hd}
        lost = sorted((det[f], f[1][:40]) for f in Hd if det[f] < 0.975)
        print(
            f"  {nm:<36} {an:<14} threshold {thr:9.3g}: loses {len(lost):>2} of {len(Hd)}; zero detection on {sum(1 for d_, _ in lost if d_ == 0)}"
        )
# pairwise: for each B family and each harmful family, best Youden J over thresholds, max over statistics/rules
print(
    "\nPairwise: the harmful family hardest to separate from each B family (best J = silence(B) + detect(harmful) - 1 over every statistic, rule and threshold)"
)
for b in Bw:
    worst = (2, None, None)
    for f in Hd:
        bestJ = (-1, None)
        for nm in names:
            for an in AGG:
                sb, sf = S[(b, nm, an)], S[(f, nm, an)]
                cand = np.unique(np.concatenate([sb[np.isfinite(sb)], sf[np.isfinite(sf)]]))
                J = max((np.mean(sb <= t) + np.mean(sf > t) - 1) for t in cand) if len(cand) else 0
                if bestJ[0] < J:
                    bestJ = (J, (nm, an))
        if bestJ[0] < worst[0]:
            worst = (bestJ[0], f, bestJ[1])
    print(f"B {b[1][:50]:<50} hardest: '{worst[1][1][:58]}' best J {worst[0]:.2f} with {worst[2]}")
# medians of the per-check maxima, to show the scales
print("\nMedian over checks of the largest line value (lines that warn under N):")
print(f"{'family':<62}" + "".join(f"{n.split(' ')[0][:14]:>16}" for n in names))
for k in Bw + [
    f
    for f in Hd
    if any(
        s in f[1]
        for s in (
            "(0.01, 2",
            "(0.1, 2",
            "d=10000, start sd 1",
            "0.01^2 I",
            "sd 10000",
            "d=5, start sd 2: sign",
            "equicorr",
            "set to 0",
        )
    )
]:
    print(
        f"{k[0]} {k[1][:59]:<60}"
        + "".join(f"{np.median(S[(k, n, 'any line')]):>16.3g}" for n in names)
    )
