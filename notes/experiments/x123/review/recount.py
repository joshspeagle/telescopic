"""Review of x123: an independent recount of the registered outputs, and the harm classes by other rules.

Question: do the counts of RESULT.md follow from the registered outputs, does every
registered command have one output whose header names the registration commit and the
command, and which verdicts depend on the rule that classes harm?

Design: reads PREREGISTRATION.md and the outputs of the folder above (it runs nothing and
reads no seed). It parses the commands of section 10 and the header of every output; the
rows of run_checks_*.out, for the counts of criteria 1 to 4 against the predictions of
the table of section 5; and the ROW lines of harm_proxy_*.out and harm_package_*.out,
from which it computes Lambda, its standard error and the registered class again and
compares them with harm_summary.out. It then classes every family by four other rules:
a t interval with 4 degrees of freedom in place of the factor 2; a t interval on the
logarithms of the 5 per-seed ratios; for the proxy, the t interval divided by its
largest and smallest validation ratios (1.63 and 1.14) against the package's bars; and
all 5 per-seed ratios beyond the bar. Criteria 2 and 4 are evaluated under each rule.

Run: python notes/experiments/x123/review/recount.py > recount.out

Result: see recount.out and REVIEW.md, objections 1, 7 and 8 and the checks that found
nothing. Every count of RESULT.md reproduces and harm_summary.out has no mismatch in 67
rows. predictions.out and equivalence.out, made before the registration commit, print
commit 32ae70b. Under the other rules 2 or 1 families of criterion 2 become unclassed;
(N) still warns on every check of the families that remain, and (H) still misses 10 or
11 harmful families. The last lines count the harmful families of C and F on which each
design warns on every check, on none and in between, and those classed by the proxy.
"""

import re
from pathlib import Path

import numpy as np
from scipy import stats

X = Path(__file__).resolve().parent.parent
# --- headers
for f in sorted(X.glob("*.out")):
    L = f.read_text().splitlines()
    if not L[0].startswith("commit 7b2c992;"):
        print("HEADER not 7b2c992:", f.name, L[0][:16])
# --- registered commands vs outputs
reg = (X / "PREREGISTRATION.md").read_text()
cmds = re.findall(
    r"^\s*\d+\. python (notes/experiments/x123/\S+\.py.*?) > (\S+)$", reg, re.MULTILINE
)
print("registered commands:", len(cmds))
seen = set()
for cmd, out in cmds:
    seen.add(out)
    p = X / out
    if not p.exists():
        print("MISSING", out)
        continue
    got = p.read_text().splitlines()[1]
    want = "command: python " + cmd.replace('"', "")
    if out == "harm_summary.out":
        ok = got.startswith(
            "command: python notes/experiments/x123/harm.py summarize harm_proxy_00.out"
        )
        n = len(got.split()) - 4
        print("summary command files:", n)
    else:
        ok = got == want
    if not ok:
        print("COMMAND MISMATCH", out, "|", got, "|", want)
extra = sorted({f.name for f in X.glob("*.out")} - seen)
print("outputs not from a registered phase-2 command:", extra)
# --- run_checks
rows = {}
for g, files in {"A": ["A1", "A2", "A3"], "B": ["B"], "C": ["C1", "C2"], "F": ["F"]}.items():
    for fn in files:
        for line in (X / f"run_checks_{fn}.out").read_text().splitlines():
            m = re.match(
                r"\[\s*(\d+)\] (.*?)\s+(\d+)\.\.(\d+) \| N\s+(\d+)/(\d+) \| H\s+(\d+)/\d+ \|\s+(\d+)\s+(\d+)\s+(\d+) \| (\S+)/(\S+) (\S+) \| (\d+) (\d+) \| (\S+) \| (\d+)",
                line,
            )
            if m:
                idx = int(m[1])
                rows[(g, m[2].strip())] = {
                    "idx": idx,
                    "s0": int(m[3]),
                    "s1": int(m[4]),
                    "N": int(m[5]),
                    "n": int(m[6]),
                    "H": int(m[7]),
                    "r6": int(m[8]),
                    "r4": int(m[9]),
                    "r2": int(m[10]),
                    "med": float(m[11]),
                    "mx": float(m[12]),
                    "mxH": float(m[13]),
                    "unus": int(m[14]),
                    "skip": int(m[15]),
                    "ev": m[16],
                    "err": int(m[17]),
                }
for g, (base, n, count) in {
    "A": (300000, 300, 109),
    "B": (335000, 300, 9),
    "C": (340000, 400, 46),
    "F": (360000, 1000, 7),
}.items():
    R = [v for (gg, _), v in rows.items() if gg == g]
    assert len(R) == count, (g, len(R))
    for v in R:
        assert v["s0"] == base + n * v["idx"], v
        assert v["s1"] == v["s0"] + n - 1, v
        assert v["n"] == n, v
    print(
        g,
        "families",
        len(R),
        "checks",
        sum(v["n"] for v in R),
        "N",
        sum(v["N"] for v in R),
        "H",
        sum(v["H"] for v in R),
        "errors",
        sum(v["err"] for v in R),
        "unusable",
        sum(v["unus"] for v in R),
        "skipped",
        sum(v["skip"] for v in R),
        "r6/r4/r2",
        sum(v["r6"] for v in R),
        sum(v["r4"] for v in R),
        sum(v["r2"] for v in R),
    )
A = [(k[1], v) for k, v in rows.items() if k[0] == "A"]
print(
    "A max N per family",
    max(v["N"] for _, v in A),
    "max H",
    max(v["H"] for _, v in A),
    "top ratios:",
    sorted(((v["mx"], n) for n, v in A), reverse=True)[:5],
)
# --- registered families table names match
tab = re.findall(r"^\| (\d+) \| (.*?) \| (\d+) to (\d+) \|$", reg, re.MULTILINE)
print("family rows in registration:", len(tab))
# --- predictions
pred = {}


def num(s):
    s = s.strip()
    if s == "-":
        return None
    if s.startswith("1-"):
        return 1 - float(s[2:])
    return float(s)


for m in re.finditer(
    r"^\| ([BCF]) \| (.*?) \| (\w+) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \|$", reg, re.MULTILINE
):
    pred[(m[1], m[2].strip())] = {
        "model": m[3],
        "Nl": num(m[4]),
        "Nc": num(m[5]),
        "Hl": num(m[6]),
        "Hc": num(m[7]),
    }
print("predictions:", len(pred))
# --- harm
harm = {}
for f in sorted([*X.glob("harm_proxy_*.out"), *X.glob("harm_package_*.out")]):
    for line in f.read_text().splitlines():
        if line.startswith("ROW |"):
            p = [x.strip() for x in line.split("|")]
            harm.setdefault((p[1], p[2], p[3]), {})[int(p[4])] = [float(x) for x in p[5:12]]
print("harm families:", len(harm))


def classify(Lam, lo, hi, bar):
    return "harmful" if lo >= bar else ("harmless" if hi <= 1.5 else "unclassed")


out = {}
t4 = stats.t.ppf(0.975, 4)
for (g, name, samp), by in harm.items():
    r = np.array([by[s] for s in sorted(by)])
    assert len(r) == 5, (name, len(r))
    ratios = r[:, 0] / r[:, 1]
    Lam = r[:, 0].mean() / r[:, 1].mean()
    se = ratios.std(ddof=1) / np.sqrt(5)
    bar = 3.0 if samp.startswith("proxy") else 2.0
    c_reg = classify(Lam, Lam - 2 * se, Lam + 2 * se, bar)
    c_t = classify(Lam, Lam - t4 * se, Lam + t4 * se, bar)
    # log-ratio t interval
    lr = np.log(ratios)
    m, s = lr.mean(), lr.std(ddof=1) / np.sqrt(5)
    lo, hi = np.exp(m - t4 * s), np.exp(m + t4 * s)
    c_log = classify(Lam, lo, hi, bar)
    # proxy deflated by 1.63 against bar 2
    c_defl = (
        classify(Lam, (Lam - t4 * se) / 1.63, (Lam + t4 * se) / 1.14, 2.0)
        if samp.startswith("proxy")
        else c_t
    )
    # min per-seed ratio (all 5 seeds above bar): sign-test like
    c_min = (
        "harmful" if ratios.min() >= bar else ("harmless" if ratios.max() <= 1.5 else "unclassed")
    )
    out[(g, name, samp)] = {
        "Lam": Lam,
        "se": se,
        "reg": c_reg,
        "t": c_t,
        "log": c_log,
        "defl": c_defl,
        "mn": c_min,
        "unmet": int(r[:, 2].sum()),
        "unmet0": int(r[:, 3].sum()),
        "ratios": ratios,
    }
# compare with harm_summary
summ = {}
for line in (X / "harm_summary.out").read_text().splitlines():
    if line.startswith("RESULT | ") and "by affine" not in line and "group" not in line[:16]:
        p = [x.strip() for x in line.split("|")]
        summ[(p[1], p[2], p[3])] = (float(p[4]), float(p[5]), p[6])
bad = 0
for k, v in out.items():
    s = summ[k]
    if abs(s[0] - v["Lam"]) > 6e-4 or abs(s[1] - v["se"]) > 6e-4 or s[2] != v["reg"]:
        bad += 1
        print("SUMMARY MISMATCH", k, s, v["Lam"], v["se"], v["reg"])
print("summary rows", len(summ), "mismatches", bad)
print(
    "\nharm classes: family | sampler | Lambda (se) | registered | t-interval(4 df) | log-ratio t | proxy deflated 1.63, bar 2 | all-5-seeds | unmet | per-seed ratios"
)
for k, v in out.items():
    flag = "" if len({v["reg"], v["t"], v["log"], v["defl"], v["mn"]}) == 1 else "  <-- differs"
    print(
        f"{k[0]} | {k[1][:62]:<62} | {k[2][:7]:<7} | {v['Lam']:8.3f} ({v['se']:7.3f}) | {v['reg']:<9} | {v['t']:<9} | {v['log']:<9} | {v['defl']:<9} | {v['mn']:<9} | {v['unmet']:>3} | "
        + " ".join(f"{x:.2f}" for x in v["ratios"])
        + flag
    )


# --- criteria
def harmclass(g, name, key="reg"):
    c = [
        v[key] for k, v in out.items() if k[0] == g and k[1] == name and k[2] != "proxy validation"
    ]
    if name.startswith("rotated cond 1e4 d=2000"):
        c = [
            v[key]
            for k, v in out.items()
            if k[1] == "N(0,I) d=2000, start sd 2: sign of coordinate 0" and k[2] == "proxy"
        ]
    assert len(c) == 1, (g, name, c)
    return c[0]


for key in ("reg", "t", "log", "defl", "mn"):
    for design, pc, cnt in (("N", "Nc", "N"), ("H", "Hc", "H")):
        c2 = []
        c4 = []
        mid = []
        for (g, name), v in rows.items():
            if g not in "CF":
                continue
            cl = harmclass(g, name, key)
            p = pred[(g, name)][pc]
            if cl != "harmful":
                continue
            if p >= 0.999:
                c2.append((name, v[cnt], v["n"], v[cnt] >= (390 if v["n"] == 400 else 975)))
            elif p < 0.001:
                c4.append((name, v[cnt], v["n"], v[cnt] <= 10))
            else:
                mid.append((g, name, p, v[cnt] / v["n"]))
        print(
            f"\nclasses={key} design {design}: criterion 2 families {len(c2)} (C {sum(1 for x in c2 if x[2] == 400)}, F {sum(1 for x in c2 if x[2] == 1000)}), all pass {all(x[3] for x in c2)}, min rate {min(x[1] / x[2] for x in c2):.4f}; predicted misses on harmful {len(c4)} confirm {all(x[3] for x in c4)} counts {[x[1] for x in c4]}"
        )
        print(
            "   harmful families with predicted rate in (0.001, 0.999), outside criteria 2 and 4:",
            [(g, n[:45], round(p, 4), round(r, 3)) for g, n, p, r in mid],
        )
# criterion 3
print()
for (g, name), v in rows.items():
    if g == "F":
        p = pred[(g, name)]
        print(
            f"F {name[:60]:<60} N {v['N'] / 1000:.3f} pred {p['Nc']:.4f} diff {v['N'] / 1000 - p['Nc']:+.3f} | H {v['H'] / 1000:.3f} pred {p['Hc']:.4f} diff {v['H'] / 1000 - p['Hc']:+.3f}"
        )
# B/C z-scores in mid range
for (g, name), v in rows.items():
    if g in "BC":
        for d_, pc in (("N", "Nc"), ("H", "Hc")):
            p = pred[(g, name)][pc]
            if 0.001 < p < 0.999:
                z = (v[d_] - v["n"] * p) / np.sqrt(v["n"] * p * (1 - p))
                print(f"{g} {name[:55]:<55} {d_} {v[d_]}/{v['n']} pred {p:.4f} z {z:+.2f}")
# proxy validation ratios
print()
for k, v in out.items():
    if k[2] == "proxy validation":
        pk = next(vv for kk, vv in out.items() if kk[1] == k[1] and kk[2] == "package")
        print(
            "validation",
            k[1],
            f"proxy {v['Lam']:.3f} package {pk['Lam']:.3f} ratio {v['Lam'] / pk['Lam']:.3f}",
        )
# the harmful families of C and F by their registered counts, and those classed by the proxy
print()
harmful = [k for k in rows if k[0] in "CF" and harmclass(*k) == "harmful"]
print("harmful families of C and F:", len(harmful))
for design in ("N", "H"):
    every = [k for k in harmful if rows[k][design] == rows[k]["n"]]
    none = [k for k in harmful if rows[k][design] == 0]
    between = [round(rows[k][design] / rows[k]["n"], 3) for k in harmful if k not in every + none]
    print(
        f"design {design}: warns on every check of {len(every)} "
        f"(C {sum(k[0] == 'C' for k in every)}, F {sum(k[0] == 'F' for k in every)}), "
        f"on no check of {len(none)}, in between on {len(between)} at rates {between}"
    )
print(
    "families of C and F classed by the proxy:",
    sum(1 for k in out if k[0] in "CF" and k[2] == "proxy"),
)
