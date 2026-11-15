"""Scores the budget-mode runs under the pre-registered rules (PREREGISTRATION.md, 5 and 7).

Run: PYTHONPATH=src python notes/experiments/i79_budget/score.py > notes/experiments/i79_budget/score.out
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from budget import BUDGETS, CASES, LANES

RUNS = HERE / "runs"
COMPONENTS = ("x", "x^2")
BOOTSTRAP = 2000


def load(name, arm, b):
    with np.load(RUNS / f"{name}-{arm}-b{b}.npz") as data:
        return dict(data)


def interval_cover(estimate, sd, size, truth):
    se = sd / np.sqrt(size)
    return np.abs(estimate - truth) <= stats.t.ppf(0.975, size - 1) * se


def bias(estimate, truth):
    # beta = (mean - truth) / sd, its Monte Carlo error 1/sqrt(n), and z = beta sqrt(n).
    n = len(estimate)
    beta = (estimate.mean(axis=0) - truth) / estimate.std(axis=0, ddof=1)
    return beta, 1 / np.sqrt(n), beta * np.sqrt(n)


def inefficiency_ratio(budget_estimate, budget_cost, fixed_estimate, fixed_cost, rng):
    def ratio(eb, cb, ef, cf):
        return eb.var(axis=0, ddof=1) * cb.mean() / (ef.var(axis=0, ddof=1) * cf.mean())

    point = ratio(budget_estimate, budget_cost, fixed_estimate, fixed_cost)
    draws = []
    nb, nf = len(budget_cost), len(fixed_cost)
    for _ in range(BOOTSTRAP):
        i, j = rng.integers(nb, size=nb), rng.integers(nf, size=nf)
        draws.append(ratio(budget_estimate[i], budget_cost[i], fixed_estimate[j], fixed_cost[j]))
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return point, low, high


def main():
    plans = json.loads((RUNS / "plans.json").read_text())
    rng = np.random.default_rng(79)
    rules = {"U": [], "U_control": [], "C": [], "E": [], "P": []}
    for name, case in CASES.items():
        truth = case.truth
        mean_cost = plans[name]["mean_cost"]
        print(
            f"== {name}: h = (x_{case.coordinate}, x_{case.coordinate}^2), truth {truth}, "
            f"E[C] = {mean_cost:.2f} from the pilot, G = {LANES}, n = {case.runs} runs per cell"
        )
        # Reference E[tau]: every replicate of the fixed-R arm of this target.
        fixed = {b: load(name, "fixed", b) for b in BUDGETS}
        total = sum(fixed[b]["tau_sum"].sum() for b in BUDGETS)
        count = sum(int(fixed[b]["size"]) * case.runs for b in BUDGETS)
        tau_ref = total / count
        var_ref = (
            sum(
                (int(fixed[b]["size"]) * case.runs) ** 2
                * (fixed[b]["tau_sum"] / fixed[b]["size"]).var(ddof=1)
                / case.runs
                for b in BUDGETS
            )
            / count**2
        )
        print(
            f"   reference E[tau] = {tau_ref:.3f} +- {np.sqrt(var_ref):.3f} "
            f"({count} fixed-R replicates)"
        )
        for b in BUDGETS:
            cell, ref = load(name, "budget", b), fixed[b]
            budget = int(cell["budget"])
            size = int(ref["size"])
            lanes = cell["estimate"]  # (n, G, 2)
            est_b = lanes.mean(axis=1)
            cover_b = interval_cover(est_b, lanes.std(axis=1, ddof=1), LANES, truth).mean(axis=0)
            cost_b = cell["charged"].sum(axis=1)
            est_f = ref["estimate"]
            cover_f = interval_cover(est_f, ref["sd"], size, truth).mean(axis=0)
            cost_f = ref["cost"]
            n = case.runs
            se_cover = np.sqrt(0.95 * 0.05 / n)
            beta_b, mc, z_b = bias(est_b, truth)
            beta_f, _, z_f = bias(est_f, truth)
            rho, low, high = inefficiency_ratio(est_b, cost_b, est_f, cost_f, rng)
            completed = cell["completed"]
            print(
                f"-- b = {b}: B = {budget} kernel steps; fixed R = {size}; mean cost per run "
                f"{cost_b.mean():.0f} (budget) against {cost_f.mean():.0f} (fixed R)"
            )
            print(
                f"   completed per lane: mean {completed.mean():.2f}, lanes with none "
                f"{(completed == 0).mean():.3f}; makespan, largest lane cost / B: mean "
                f"{(cell['charged'].max(axis=1) / budget).mean():.2f}, largest "
                f"{(cell['charged'].max(axis=1) / budget).max():.2f}"
            )
            for c, component in enumerate(COMPONENTS):
                print(
                    f"   {component:>3}: bias/sd budget {beta_b[c]:+.3f} +- {mc:.3f} "
                    f"(z {z_b[c]:+.2f}), fixed {beta_f[c]:+.3f} (z {z_f[c]:+.2f}); coverage "
                    f"budget {cover_b[c]:.3f}, fixed {cover_f[c]:.3f} (+- {se_cover:.3f} each); "
                    f"rho {rho[c]:.3f} [{low[c]:.3f}, {high[c]:.3f}]"
                )
                rules["U"].append(abs(z_b[c]) <= 3.5)
                rules["U_control"].append(abs(z_f[c]) <= 3.5)
                rules["C"].append(cover_b[c] >= cover_f[c] - 3 * np.sqrt(2) * se_cover)
                if b == 10:
                    rules["E"].append(rho[c] <= 1.25)
            # Positive control: the pooled average of the replicates completed within B.
            done = completed.sum(axis=1)
            ok = done > 0
            naive = cell["completed_sum"].sum(axis=1)[ok] / done[ok, None]
            beta_n, _, z_n = bias(naive, truth)
            rules["P"].extend(np.abs(z_n) >= 5)
            print(
                f"   P, pooled completed average ({(~ok).sum()} runs with none dropped): "
                + "; ".join(
                    f"{component} bias/sd {beta_n[c]:+.3f} (z {z_n[c]:+.1f})"
                    for c, component in enumerate(COMPONENTS)
                )
            )
            # Post hoc, not registered: rho split into what the counts alone give,
            # E[1/N~] E[charged] / E[C], which holds if H were independent of the lane's
            # costs, and the rest, Var(H~) / (Var(H) E[1/N~]).
            kept_count = np.maximum(completed, 1)
            per_replicate_cost = (cost_f / size).mean()
            from_counts = (1 / kept_count).mean() * cell["charged"].mean() / per_replicate_cost
            var_h = (ref["sd"] ** 2).mean(axis=0)
            rest = lanes.reshape(-1, 2).var(axis=0, ddof=1) / (var_h * (1 / kept_count).mean())
            print(
                f"   post hoc: rho from the counts alone {from_counts:.3f}; Var(H~) / "
                f"(Var(H) E[1/N~]) "
                + ", ".join(f"{component} {rest[c]:.3f}" for c, component in enumerate(COMPONENTS))
            )
            # D: E[tau] pooled over kept replicates, and lane-weighted.
            some = completed > 0
            kept = np.maximum(completed, 1)
            tau_kept = np.where(some, cell["completed_tau"], cell["first_tau"])
            pooled = tau_kept.sum(axis=1) / kept.sum(axis=1)
            weighted = cell["lane_tau"].mean(axis=1)
            for label, values in (("pooled (a)", pooled), ("lane-weighted (b)", weighted)):
                se = values.std(ddof=1) / np.sqrt(n)
                z = (values.mean() - tau_ref) / np.sqrt(se**2 + var_ref)
                print(
                    f"   D, E[tau] {label}: {values.mean():.3f} +- {se:.3f}, "
                    f"minus reference {values.mean() - tau_ref:+.3f} (z {z:+.1f})"
                )
    print("== rules")
    # P needs |z| >= 5 in at least one cell; the others need every test to pass.
    verdicts = {key: any(v) if key == "P" else all(v) for key, v in rules.items()}
    for key, values in rules.items():
        print(
            f"   {key}: {sum(values)} of {len(values)} pass -> "
            f"{'holds' if verdicts[key] else 'fails'}"
        )
    if not verdicts["U_control"]:
        print("   the fixed-R control fails U in some cell: those cells are invalid")
    if verdicts["U"] and verdicts["C"] and verdicts["E"]:
        verdict = "build budget mode on part B's plans (#79 option 3)"
    elif not (verdicts["U"] and verdicts["C"]):
        verdict = "do not build it as proposed"
    else:
        verdict = "unbiased, but more than 25% dearer than fixed R at b = 10: do not build now"
    print(
        f"   verdict: {verdict}"
        + ("" if verdicts["P"] else "; the positive control did not show the bias at stake")
    )


if __name__ == "__main__":
    main()
