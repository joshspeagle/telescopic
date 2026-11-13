"""The adversarial review: recount the confirmation's verdict from its saved rows.

Checks that q3_rows.npy and the per-job files of rows_q3/, which i102_q3_confirm.py writes,
agree field by field; recomputes from them the detections, the false alarms, the cost and
each run's z; lists the seeds, the statistics of the wrong runs and the largest ones of
the right runs; and gives Clopper-Pearson intervals.

Result (i102_review_recount.out): 67 of 67 wrong criterion runs detected, no false alarm in
440 right runs (95% interval [0, 0.0083]), cost at most 0.25 of the pilot, no mismatch
in z; without the R-hat silencing, 2 of the 440 right runs would fire. This output was
captured when the experiment was recorded, from the confirmation's rows; it does not use
the package.

Run: python notes/experiments/i102_review_recount.py <folder where i102_q3_confirm.py ran>
"""

import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

E = sys.argv[1]
MS = (4, 8, 16, 32)
DIVS = (64, 32, 16, 8, 4, 2)
CRIT = {f"bimodal d={d} s={s}" for d in (10, 32) for s in (6, 12)}
rows = list(np.load(os.path.join(E, "q3_rows.npy"), allow_pickle=True))
pk = [
    pickle.loads(Path(p).read_bytes())
    for p in sorted(glob.glob(os.path.join(E, "rows_q3", "*.pkl")))
]
print("npy rows", len(rows), "pkl rows", len(pk))
key = lambda r: (r["cell"], r["seed"])
d1 = {key(r): r for r in rows}
d2 = {key(r): r for r in pk}
assert set(d1) == set(d2)
for k, a in d1.items():
    b = d2[k]
    assert set(a) == set(b), k
    for f in a:
        x, y = a[f], b[f]
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True), (k, f)
        elif isinstance(x, float) and np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y, (k, f, x, y)
print("npy == pkl for every row and field")
to = [r for r in rows if r.get("timed_out")]
print("timed out:", [key(r) for r in to])
rows = [r for r in rows if not r.get("timed_out")]
stat = lambda r: r["stats"][MS.index(8), DIVS.index(8), 0]
fires = lambda r: bool(stat(r) > 10 and not r["disagree"])
seeds = {}
for r in rows + to:
    seeds.setdefault(r["kind"], set()).add(r["seed"])
print({k: (min(v), max(v), len(v)) for k, v in seeds.items()})
det = [r for r in rows if r["cell"] in CRIT and r["wrong"] is True]
crit_all = [r for r in rows if r["cell"] in CRIT]
fa_all = [r for r in rows if r["kind"] == "fa"]
fa = [r for r in fa_all if r["wrong"] is False]
print(
    "criterion runs",
    len(crit_all),
    "wrong",
    len(det),
    "right",
    sum(r["wrong"] is False for r in crit_all),
    "raised",
    sum(r["wrong"] is None for r in crit_all),
)
print("detected", sum(map(fires, det)), "/", len(det))
print(
    "fa cells",
    len({r["cell"] for r in fa_all}),
    "fa runs",
    len(fa_all),
    "right",
    len(fa),
    "wrong",
    sum(r["wrong"] is True for r in fa_all),
    "raised",
    sum(r["wrong"] is None for r in fa_all),
)
print("false alarms", sum(map(fires, fa)), "/", len(fa))
print(
    "fa fires ignoring silencing",
    sum(stat(r) > 10 for r in fa),
    "silenced",
    sum(r["disagree"] for r in fa),
)
print(
    "cost max",
    max(8 * (r["n_steps"] // 8) / r["pilot_evals"] for r in rows),
    "with starts",
    max(8 * (r["n_steps"] // 8 + 1) / r["pilot_evals"] for r in rows),
)
print(
    "pilot_evals == 4(n_steps+1):", all(r["pilot_evals"] == 4 * (r["n_steps"] + 1) for r in rows)
)
# independent recheck of wrong: |z|>5 from value/se for detect cells (truth weights)
truth = dict.fromkeys(CRIT, 0.7)
bad = 0
for r in crit_all:
    if r["wrong"] is None:
        continue
    z = (r["value"] - 0.7) / r["se"] if r["se"] > 0 else (0 if r["value"] == 0.7 else np.inf)
    if (abs(z) > 5) != r["wrong"] or (not np.isclose(z, r["z"]) and np.isfinite(z)):
        bad += 1
        print("mismatch", key(r), z, r["z"], r["wrong"])
print("criterion z recomputed, mismatches:", bad)
print(
    "criterion |z| of wrong: min",
    min(abs(r["z"]) for r in det),
    " |z| of right: max",
    max([abs(r["z"]) for r in crit_all if r["wrong"] is False]),
)
print(
    "criterion z in (3,8):",
    sorted(
        (r["cell"], r["seed"], round(r["z"], 2), fires(r))
        for r in crit_all
        if r["wrong"] is not None and 3 < abs(r["z"]) < 8
    ),
)
print(
    "fa |z| max:",
    max(abs(r["z"]) for r in fa_all),
    "; fa |z|>3:",
    sorted((r["cell"], r["seed"], round(r["z"], 2)) for r in fa_all if abs(r["z"]) > 3),
)
print(
    "criterion wrong values:",
    np.unique(np.round([r["value"] for r in det], 2), return_counts=True),
)
print(
    "criterion right: ",
    sorted(
        (
            r["cell"],
            r["seed"],
            round(r["value"], 3),
            round(r["se"], 4),
            round(r["z"], 2),
            int(r["disagree"]),
            round(float(stat(r)), 1),
            r["k"],
            r["thin"],
        )
        for r in crit_all
        if r["wrong"] is False
    ),
)
print("stat on wrong crit: sorted", np.round(sorted(stat(r) for r in det), 1))
s_fa = sorted(
    ((float(stat(r)), r["cell"], r["seed"], int(r["disagree"])) for r in fa), reverse=True
)
print("top fa stats:", [(round(a, 1), b, c, d) for a, b, c, d in s_fa[:25]])
print(
    "thinned among det:",
    sum(r["thin"] > 1 for r in det),
    "among fa:",
    sum(r["thin"] > 1 for r in fa),
)
print("warn texts fa:", {r["warn_text"] for r in fa_all})


# Clopper-Pearson
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


print("CP 67/67", cp(67, 67), "one-sided 95% lower", 0.05 ** (1 / 67))
print("CP 0/440", cp(0, 440), "one-sided 95% upper", 1 - 0.05 ** (1 / 440))
print("CP 69/69", cp(69, 69), "67/69", cp(67, 69))
# other detection cells
for c in dict.fromkeys(r["cell"] for r in rows):
    cr = [r for r in rows if r["cell"] == c]
    if cr[0]["kind"] != "fa":
        w = [r for r in cr if r["wrong"] is True]
        print(
            f"{c:30s} runs {len(cr)} wrong {len(w)} raised {sum(r['wrong'] is None for r in cr)} fires on wrong {sum(map(fires, w))} fires on right {sum(fires(r) for r in cr if r['wrong'] is False)} fires on raised {sum(fires(r) for r in cr if r['wrong'] is None)}"
        )
