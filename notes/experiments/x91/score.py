"""The scorer of x91: the criteria of PREREGISTRATION.md section 7 and the rules of section 8.

    python notes/experiments/x91/score.py [--offset N]

It reads the stores that `run.py` writes (`notes/experiments/_rows/x91/`, or
`x91_offset<N>/`), raises if their records come from more than one commit, and prints
each cell's measurements, every criterion for every arm where section 7's table applies
it, and the verdicts of section 8. It reads the outcome of `check_coupling` from the last
line of `check_exactness.out`. `confirmations` tells `run.py --confirm` which fresh seeds
a z-score beyond 3.5 needs (section 7, Z).
"""

import argparse
import functools
import math
import re
from pathlib import Path

import numpy as np
from cells import CELL, CELLS, SEEDS, logistic_reference, target

HERE = Path(__file__).parent
ROWS = HERE.parent / "_rows"
U_LEVEL, BUDGET_GRADIENTS = 0.01, 2e5
M_LEVEL, V_LEVEL, L_FACTOR = 1.20, 1.20, 3.0
Z_LEVEL, Z_CONFIRM = 3.5, 2.0
T_FACTOR, T_MAX = 0.9998, 3
JITTERS = ("J2", "J3", "J5")
VERDICT_ARMS = ("G", *JITTERS, "CF", "MJ5", "MCF")
PURPOSE = {
    "G": ("A", "A'"),
    **dict.fromkeys(JITTERS, ("A", "A'", "B", "B'")),
    "CF": ("D2", "D3"),
    "MJ5": ("E1", "E2"),
    "MCF": ("E1", "E2"),
}


def block(cell):
    # "D2", "C5", "E1": the cell's own name in tables that name cells of C, D and E singly.
    return cell.name.split()[0] if cell.part in "CDE" else cell.part


def comparator(arm):
    return {"CF.2": "P.2"}.get(arm, "MP" if arm.startswith("M") else "P")


class Records:
    """The records of the stores, by cell, arm and seed."""

    def __init__(self, records, offset=0):
        self.all, self.offset = records, offset
        commits = {
            re.search(r"commit (\S+) of the package", r["setup"])[1]
            for r in records.values()
            if "setup" in r
        }
        if len(commits) > 1:
            raise ValueError(f"the stores mix records of commits {sorted(commits)}")
        self.commit = commits.pop() if commits else None
        self.setup = next((r["setup"] for r in records.values() if "setup" in r), None)

    def listed(self, cell, arm, confirm=False):
        # The arm's own seeds: the first of the cell's block, or of the confirmation seeds.
        n = cell.arms[arm]
        first = SEEDS["confirmation"] if confirm else None
        listed = range(first, first + n) if confirm else cell.seeds[:n]
        return [s + self.offset for s in listed]

    def seeds(self, cell, arm, confirm=False):
        # P's set adds every seed where its record was computed for G's copy: "P's record
        # at that seed is computed and used for both" (section 5).
        seeds = self.listed(cell, arm, confirm)
        if arm == "P":
            extra = {
                r["seed"]
                for r in self.all.values()
                if r.get("for_copy")
                and r["cell"] == cell.name
                and r["arm"] == "P"
                and (r["part"] == "Z") == confirm
            }
            seeds += sorted(extra - set(seeds))
        return seeds

    def get(self, cell, arm, confirm=False):
        # The arm's records at its seeds, in their order; missing ones left out.
        prefix = "Z|" if confirm else ""
        keys = (f"{prefix}{cell.name}|{arm}|{s}" for s in self.seeds(cell, arm, confirm))
        return [self.all[k] for k in keys if k in self.all]

    def complete(self, cell, arm, confirm=False):
        prefix = "Z|" if confirm else ""
        return all(
            f"{prefix}{cell.name}|{arm}|{s}" in self.all for s in self.listed(cell, arm, confirm)
        )

    def cut(self, cell):
        return f"{cell.name}|cut" in self.all


# Per record.
def k_failed(record, cell):
    # A seed where k cannot be chosen; in a full run, also one without replicates (N5).
    if record["k"] is None:
        return True
    return (
        cell.run == "full"
        and record["arm"] not in cell.pilot_arms
        and record.get("replicates") != "ran"
    )


def unmet_budget(record):
    tau = np.array(record["tau"])
    return int(np.sum(np.isinf(tau) | (tau * record["g"] > BUDGET_GRADIENTS)))


def median_cost(record):
    return record["median"] * record["g"]


# Criteria for one arm on one cell.
def u_fractions(records):
    pairs = 100 * len(records)
    return (
        sum(r["unmet"] for r in records) / pairs,
        sum(unmet_budget(r) for r in records) / pairs,
    )


def u_holds(rec, cell, arm):
    """(holds, at the cap, at the budget, whether the budget is judged)."""
    cap, budget = u_fractions(rec.get(cell, arm))
    reference = comparator(arm)
    judged = True
    if reference in cell.arms and rec.get(cell, reference):
        p_cap, p_budget = u_fractions(rec.get(cell, reference))
        # P slow at the budget but meeting at the cap: the budget is reported only (N2).
        judged = not (p_cap <= U_LEVEL < p_budget)
    holds = cap <= U_LEVEL and (budget <= U_LEVEL or not judged)
    return holds, cap, budget, judged


def k_failures(rec, cell, arm):
    return sum(k_failed(r, cell) for r in rec.get(cell, arm))


def log_mean_cost(records):
    with np.errstate(divide="ignore"):
        return float(np.mean(np.log([median_cost(r) for r in records])))


def rho_m(rec, cell, arm):
    return math.exp(
        log_mean_cost(rec.get(cell, arm)) - log_mean_cost(rec.get(cell, comparator(arm)))
    )


def p_unmet(rec, cell, arm):
    # Whether the arm's comparator left a pair unmet at the cap at any of its seeds.
    return any(r["unmet"] for r in rec.get(cell, comparator(arm)))


def tail(records, cell):
    # T_A: the geometric mean of k g over the seeds where k was chosen.
    values = [r["k"] * r["g"] for r in records if r["k"] is not None]
    return float(np.exp(np.mean(np.log(values)))) if values else math.inf


def l_outcome(rec, cell, arm):
    """(holds, T_A, the smallest T_A among the arms on the cell that hold K)."""
    # The arms at a meeting weight of 0.2 decide no default (section 8): they set no
    # threshold for the others.
    arms = [a for a in cell.arms if a == arm or a not in ("P.2", "CF.2")]
    holding = {a: tail(rec.get(cell, a), cell) for a in arms if k_failures(rec, cell, a) <= 1}
    if arm not in holding:
        return None, math.nan, min(holding.values(), default=math.nan)
    best = min(holding.values())
    return holding[arm] <= L_FACTOR * best, holding[arm], best


def inefficiencies(records):
    return {r["seed"]: r["inefficiency"][0] for r in records if r.get("replicates") == "ran"}


def v_outcome(rec, cell, arm):
    """(holds or None where void, rho_V, its bootstrap 5% point, seeds used)."""
    a, p = inefficiencies(rec.get(cell, arm)), inefficiencies(rec.get(cell, "P"))
    half = len(cell.seeds) / 2
    if len(a) < half or len(p) < half:
        return None, math.nan, math.nan, min(len(a), len(p))
    common = sorted(set(a) & set(p))
    x, y = np.array([a[s] for s in common]), np.array([p[s] for s in common])
    rho = float(np.median(x) / np.median(y))
    rng = np.random.default_rng(SEEDS["bootstrap"])
    n = len(common)
    ratios = np.median(x[rng.integers(n, size=(10_000, n))], axis=1) / np.median(
        y[rng.integers(n, size=(10_000, n))], axis=1
    )
    low = float(np.quantile(ratios, 0.05))
    return low <= V_LEVEL, rho, low, n


# Z.
@functools.cache
def reference(offset):
    return logistic_reference(SEEDS["logistic_reference"] + offset)


def truth(cell, offset):
    # The exact values and their standard error: 0, or the logistic reference's.
    exact = target(cell.target)[4]
    if exact is not None:
        return np.array(exact), 0.0
    return np.array(reference(offset)["mean"]), np.array(reference(offset)["se"])


def pooled(records):
    """(pooled estimate, its standard error, number of seeds) over seeds with replicates."""
    ran = [r for r in records if r.get("replicates") == "ran"]
    if not ran:
        return None
    value = np.mean([r["value"] for r in ran], axis=0)
    se = np.sqrt(np.sum(np.square([r["se"] for r in ran]), axis=0)) / len(ran)
    return value, se, len(ran)


def z_scores(rec, cell, arm, confirm=False, against_p=False):
    """Pooled z of each function: against the exact value, or of the difference with P."""
    mine = pooled(rec.get(cell, arm, confirm))
    if mine is None:
        return None
    if against_p:
        theirs = pooled(rec.get(cell, comparator(arm), confirm))
        if theirs is None:
            return None
        return (mine[0] - theirs[0]) / np.sqrt(mine[1] ** 2 + theirs[1] ** 2)
    exact, exact_se = truth(cell, rec.offset)
    return (mine[0] - exact) / np.sqrt(mine[1] ** 2 + exact_se**2)


def z_outcome(rec, cell, arm, against_p=False):
    """("holds", "fails", "unconfirmed" while its fresh seeds are missing, or None), z."""
    z = z_scores(rec, cell, arm, against_p=against_p)
    if z is None:
        return None, None
    flagged = np.abs(z) > Z_LEVEL
    if not flagged.any():
        return "holds", z
    if not rec.complete(cell, arm, True) or (
        against_p and not rec.complete(cell, comparator(arm), True)
    ):
        return "unconfirmed", z
    fresh = z_scores(rec, cell, arm, confirm=True, against_p=against_p)
    confirmed = flagged & (np.sign(fresh) == np.sign(z)) & (np.abs(fresh) > Z_CONFIRM)
    return ("fails" if confirmed.any() else "holds"), z


def p_biased(rec, cell):
    # Whether P's own z beyond 3.5 is confirmed on a full cell, so that the arms there are
    # judged by their difference with P.
    reference = "MP" if cell.part == "E" else "P"
    return z_outcome(rec, cell, reference)[0] == "fails"


def confirmations(records, offset=0):
    """(cell, arm, number of fresh seeds) still to run to confirm a z beyond 3.5."""
    rec = Records(records, offset)
    todo = []
    for cell in CELLS:
        if cell.run != "full":
            continue
        reference = "MP" if cell.part == "E" else "P"
        arms = [a for a in cell.arms if a not in cell.pilot_arms and rec.complete(cell, a)]
        if reference not in arms:
            continue
        status = z_outcome(rec, cell, reference)[0]
        if status == "unconfirmed":
            todo.append((cell.name, reference, cell.arms[reference]))
            continue
        against_p = status == "fails"
        for arm in arms:
            if arm != reference and z_outcome(rec, cell, arm, against_p)[0] == "unconfirmed":
                todo.append((cell.name, arm, cell.arms[arm]))
    return todo


# Which criteria apply (section 7's table).
def applies(arm, cell, criterion, rec):
    b = block(cell)
    other_cells = cell.part in ("A", "A'", "C") or b == "D1"
    if cell.part == "T" or cell.part == "F":
        return False
    if arm in ("G", *JITTERS):
        if criterion in ("U", "K"):
            return cell.part in ("A", "A'", "B", "B'", "C") or b == "D1"
        if criterion == "M":
            return other_cells and not cell.of_T and not p_unmet(rec, cell, arm)
        if criterion == "L":
            return cell.of_T or (other_cells and p_unmet(rec, cell, arm))
        if criterion == "V":
            return cell.part == "C"
    if arm == "CF":
        cf_cells = cell.part in ("A", "C") or b == "D1"
        if criterion in ("U", "K"):
            return True
        if criterion == "M":
            return cf_cells and not cell.of_T and not p_unmet(rec, cell, arm)
        if criterion == "L":
            return (cell.of_T and cell.part in ("A", "B")) or (
                cf_cells and p_unmet(rec, cell, arm)
            )
        if criterion == "V":
            return cell.part == "C"
    if arm == "CF.2":
        return criterion == "M" and b in ("A", "C5", "D1") and not p_unmet(rec, cell, arm)
    if arm in ("MJ5", "MCF"):
        if criterion in ("U", "K"):
            return True
        if criterion == "M":
            # E3 and E4 as section 7's table names them; E2 only if MP has no pair unmet.
            return b in ("E3", "E4") or (b == "E2" and not p_unmet(rec, cell, arm))
    return False


def evaluate(rec, check_failed):
    """Every criterion of every arm on every cell: {arm: [(cell, criterion, holds, text)]}."""
    out = {arm: [] for arm in (*VERDICT_ARMS, "CF.2")}
    for cell in CELLS:
        if cell.part in ("T", "F") or rec.cut(cell):
            continue
        biased = cell.run == "full" and p_biased(rec, cell)
        for arm, row in out.items():
            if arm not in cell.arms or not rec.get(cell, arm):
                continue
            u = u_holds(rec, cell, arm)
            k = k_failures(rec, cell, arm)
            for name, holds, text in (
                ("U", u[0], f"unmet {u[1]:.4f} at the cap, {u[2]:.4f} at the budget"
                 + ("" if u[3] else " (reported)")),
                ("K", k <= 1, f"{k} of {len(rec.get(cell, arm))} seeds without k"),
            ):  # fmt: skip
                if applies(arm, cell, name, rec):
                    row.append((cell, name, holds, text))
                elif arm in ("G", *JITTERS) and block(cell) in ("D2", "D3"):
                    row.append((cell, name + " (reported)", None, text))
            if applies(arm, cell, "M", rec) and rec.get(cell, comparator(arm)):
                rho = rho_m(rec, cell, arm)
                row.append((cell, "M", rho <= M_LEVEL, f"rho_M {rho:.3f}"))
            if applies(arm, cell, "L", rec):
                holds, mine, best = l_outcome(rec, cell, arm)
                row.append((cell, "L", holds, f"T {mine:.0f}, smallest {best:.0f}"))
            if applies(arm, cell, "V", rec) and rec.get(cell, "P"):
                holds, rho, low, n = v_outcome(rec, cell, arm)
                text = f"rho_V {rho:.3f}, 5% point {low:.3f}, {n} seeds"
                row.append((cell, "V", holds, text if holds is not None else f"void, {n} seeds"))
            if cell.run == "full" and arm not in cell.pilot_arms:
                status, z = z_outcome(rec, cell, arm, biased)
                if status:
                    kind = "Z (difference with P)" if biased else "Z"
                    text = f"{status}; z {np.round(z, 2).tolist()}"
                    holds = {"holds": True, "fails": False, "unconfirmed": None}[status]
                    row.append((cell, kind, holds, text))
    for arm in ("J2", "J3", "J5", "CF", "MJ5", "MCF"):
        if arm in check_failed:
            out[arm].append((None, "check_coupling", False, "failed its repeat"))
    return out


def t_test(rec):
    """G's tuning test per cell of T, and whether the proxy check makes it judged."""
    tests = {}
    for cell in CELLS:
        if cell.part == "T":
            factors = [r["factor"] for r in rec.get(cell, "G")]
            above = sum(f > T_FACTOR for f in factors)
            tests[cell.name] = (above <= T_MAX, above, len(factors))
    # The proxy check (section 8, rule 1) on the registered pilot records of P and G on the
    # nine cells: G's copies are P's records and are counted once, as P's.
    pool = [
        r
        for r in rec.all.values()
        if r.get("arm") in ("P", "G")
        and r.get("part") != "Z"
        and "factor" in r
        and CELL[r["cell"]].of_T
        and CELL[r["cell"]].part != "T"
        and not r.get("same_as")
    ]
    above = [r for r in pool if r["factor"] > T_FACTOR]
    failed = sum(r["k"] is None for r in above)
    judged = len(above) >= 5 and failed >= len(above) / 2
    return tests, judged, len(above), failed, len(pool)


def verdicts(rec, rows, t, check_failed):
    tests, t_judged = t[0], t[1]
    result = {}
    for arm in VERDICT_ARMS:
        row = rows[arm]
        failed = [(c, n) for c, n, h, _ in row if h is False]
        purpose_cells = [c for c in CELLS if block(c) in PURPOSE[arm] or c.part in PURPOSE[arm]]
        refused = any(n.startswith("Z") for _, n in failed) or arm in check_failed
        refused |= any(c is not None and c in purpose_cells and n in ("U", "K") for c, n in failed)
        if arm == "G" and t_judged:
            refused |= any(
                not tests[c.name][0]
                for c in CELLS
                if c.part == "T" and c.name.split(" ", 1)[1].startswith(("A ", "A' "))
            )
        undecided = any(rec.cut(c) and arm in c.arms for c in CELLS if c.part == "E")
        if refused:
            result[arm] = "refused"
        elif undecided:
            result[arm] = "undecided (a cell was cut)"
        elif not failed:
            result[arm] = "default candidate"
        else:
            result[arm] = "opt-in"
    return result


def check_exactness_failures():
    path = HERE / "check_exactness.out"
    if not path.exists():
        return None
    last = path.read_text().strip().splitlines()[-1]
    match = re.match(r"kernels that failed their repeat: (.*)$", last)
    if not match:
        raise ValueError(f"check_exactness.out ends with {last!r}, not its verdict line")
    return set() if match[1] == "none" else set(match[1].split(", "))


def describe(rec, cell):
    lines = []
    for arm in cell.arms:
        rs = rec.get(cell, arm)
        if not rs:
            lines.append(f"  {arm:5} no records")
            continue
        text = f"  {arm:5} {len(rs):3}/{cell.arms[arm]:3} seeds"
        if cell.run == "tune":
            moved = sum(r["guard"]["action"] == "moved" for r in rs)
            fallback = sum(r["guard"]["action"] == "fallback" for r in rs)
            above = sum(r["factor"] > T_FACTOR for r in rs)
            text += f", moved {moved}, fallback {fallback}, factor > 0.9998 at {above}"
            lines.append(text)
            continue
        cap, budget = u_fractions(rs)
        medians = [r["median"] for r in rs]
        text += (
            f", eps median {np.median([r['eps'] for r in rs]):.4f}, unmet {cap:.4f} (cap) "
            f"{budget:.4f} (budget), k failed {k_failures(rec, cell, arm)}, median of medians "
            f"{np.median(medians):.1f}, T {tail(rs, cell):.0f}"
        )
        if any("factor" in r for r in rs):
            text += f", largest factor max {max(r.get('factor', 0) for r in rs):.6f}"
        if arm == "G":
            actions = [r["guard"]["action"] for r in rs]
            text += ", " + ", ".join(f"{a} {actions.count(a)}" for a in sorted(set(actions)))
        if cell.run == "full":
            ran = sum(r.get("replicates") == "ran" for r in rs)
            text += f", replicates at {ran} seeds"
        lines.append(text)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    from run import Stores

    folder = ROWS / ("x91" if not args.offset else f"x91_offset{args.offset}")
    rec = Records(Stores(folder).records, args.offset)
    print(rec.setup, "\nrecords of commit", rec.commit, "printed with the setup above")
    seconds = [r.get("seconds", 0.0) for r in rec.all.values()]
    tier1 = sum(
        r.get("seconds", 0.0)
        for r in rec.all.values()
        if "cell" in r and CELL[r["cell"]].tier == 1
    )
    print(f"processor seconds recorded: {sum(seconds):.0f} in all, {tier1:.0f} in tier 1")
    incomplete = [
        c.name for c in CELLS if not rec.cut(c) and not all(rec.complete(c, a) for a in c.arms)
    ]
    cut = [c.name for c in CELLS if rec.cut(c)]
    print(f"cut cells: {cut or 'none'}; incomplete cells: {incomplete or 'none'}")
    if incomplete:
        print("THE VERDICTS BELOW ARE PROVISIONAL: some cells are incomplete")
    check_failed = check_exactness_failures()
    print(
        "check_coupling: "
        + (
            "no check_exactness.out"
            if check_failed is None
            else f"failed {check_failed or 'none'}"
        )
    )
    print("\nCells")
    for cell in CELLS:
        print(cell.name + (" (cut)" if rec.cut(cell) else ""))
        if not rec.cut(cell):
            print("\n".join(describe(rec, cell)))
    if rec.get(CELL["C5 logistic n=default"], "P"):
        ref = reference(args.offset)
        print(
            f"\nlogistic reference: mean {np.round(ref['mean'], 4).tolist()}, se "
            f"{np.round(ref['se'], 5).tolist()}, ESS {ref['ess']:.0f} of {ref['draws']} draws"
        )
    t = t_test(rec)
    print("\nG's tuning test (at most 3 of 300 above 0.9998 after the guard)")
    for name, (holds, above, n) in t[0].items():
        print(f"  {name}: {above} of {n} above; {'holds' if holds else 'fails'}")
    print(
        f"  proxy check: {t[2]} of {t[4]} pilot records of P and G above 0.9998, k not chosen "
        f"at {t[3]}: T is {'judged' if t[1] else 'reported, not judged'}"
    )
    rows = evaluate(rec, check_failed or set())
    print("\nCriteria (holds, fails, or reported)")
    for arm, row in rows.items():
        print(arm)
        for cell, name, holds, text in row:
            mark = {True: "holds", False: "FAILS", None: "reported"}[holds]
            print(f"  {cell.name if cell else '-':28} {name:22} {mark:8} {text}")
    result = verdicts(rec, rows, t, check_failed or set())
    print("\nVerdicts (section 8)")
    unconfirmed = [
        f"{arm} on {c.name}"
        for arm, row in rows.items()
        for c, n, h, text in row
        if n.startswith("Z") and text.startswith("unconfirmed")
    ]
    if unconfirmed:
        print(
            "THE VERDICTS BELOW ARE PROVISIONAL: z beyond 3.5 awaits its fresh seeds "
            f"(run.py --confirm): {', '.join(unconfirmed)}"
        )
    if check_failed is None:
        print("THE VERDICTS BELOW ARE PROVISIONAL: no check_exactness.out")
    if incomplete:
        print("THE VERDICTS BELOW ARE PROVISIONAL: some cells are incomplete")
    print(
        "M of MJ5 and MCF: on E3 and E4 whatever MP shows there, on E2 only where MP left "
        "no pair unmet at the cap (section 7's table)"
    )
    for arm, verdict in result.items():
        print(f"  {arm}: {verdict}")
    print("\nBetween the remedies")
    candidates = [a for a in JITTERS if result[a] == "default candidate"]
    if candidates:
        print(f"  jitter: the widest default candidate, {candidates[-1]}, is recommended")
    else:
        holding = [
            a
            for a in JITTERS
            if result[a] != "refused"
            and not any(
                h is False and n in ("U", "K") and c.part in PURPOSE[a] for c, n, h, _ in rows[a]
            )
        ]
        print(f"  jitter: no default candidate; opt-in recommended: {holding[:1] or 'none'}")
    g_rows = rows["G"]

    def g_holds(parts):
        criteria = [h for c, n, h, _ in g_rows if c and c.part in parts and n in ("U", "K", "L")]
        return all(h is not False for h in criteria)

    g_purpose = g_holds(("A", "A'")) and (
        not t[1] or all(t[0][c][0] for c in t[0] if c.split(" ", 1)[1].startswith(("A ", "A' ")))
    )
    print(f"  G passes its purpose (U, K, L on A and A', and T there): {g_purpose}")

    def g_scope(names):
        return all(
            h is not False
            for c, n, h, _ in g_rows
            if c and c.name in names and n in ("U", "K", "L")
        ) and (not t[1] or all(t[0][f"T{i} {nm}"][0] for i, nm in _t_index(names)))

    for name in (
        "B unequal10 n=3",
        "B unequal10 n=10",
        "B unequal10 n=default",
        "B' unequal8 n=default",
    ):
        print(f"  G holds U, K, L and the tuning test on {name}: {g_scope((name,))}")
    for label, names in (
        ("given n_steps (B's 3 and 10)", ("B unequal10 n=3", "B unequal10 n=10")),
        (
            "default n_steps (B's default and B')",
            ("B unequal10 n=default", "B' unequal8 n=default"),
        ),
    ):
        print(f"  G covers a given covariance with {label}: {g_scope(names)}")
    cf = rows["CF"]
    refuted_default = any(h is False for _, _, h, _ in cf)
    refuted_curvature = any(
        h is False and (n.startswith("Z") or (c and block(c) in ("D2", "D3") and n in ("U", "K")))
        for c, n, h, _ in cf
    )
    print(f"  CF refuted as the default at the package's meeting step: {refuted_default}")
    print(f"  CF refuted as an opt-in for curvature: {refuted_curvature}")
    cf2 = [h for _, n, h, _ in rows["CF.2"] if n == "M"]
    print(
        f"  CF.2 against P.2 on M: {sum(h is True for h in cf2)} of {len(cf2)} cells pass"
        + (
            " (all four: the cost comes from the meeting weight)"
            if len(cf2) == 4 and all(cf2)
            else ""
        )
    )


def _t_index(names):
    for cell in CELLS:
        if cell.part == "T":
            index, base = cell.name.split(" ", 1)
            if base in names:
                yield int(index[1:]), base


if __name__ == "__main__":
    main()
