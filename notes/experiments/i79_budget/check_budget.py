"""Checks of the budget prototype's own contracts (PREREGISTRATION.md, section 2).

1. The lane rule on hand-made costs.
2. Exact unbiasedness of the lane estimate on a discrete (H, C) law, by enumeration in
   rational arithmetic, and the bias there of the average of the completed replicates
   alone, 0 when none completed (GH90's biased estimator, the positive control).
3. budget_cell equals a driver that runs each lane's replicates one at a time
   through engine.run and censors at max_iter the replicate that cannot finish within the
   lane's remaining budget, abandoning it: the same completed counts, and lane estimates
   equal to 1e-12 (the driver sums each replicate's atoms in another order).
4. pilot_costs agrees with lanes of budget_cell run on consecutive blocks.

Run: PYTHONPATH=src python notes/experiments/i79_budget/check_budget.py
"""

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from budget import CASES, budget_cell, lane_rule, pilot_costs, replicates

from telescopic import engine

# 1. The lane rule.
assert lane_rule([5], 4) == (1, 0, 5.0)  # none completed: the first is kept, overrunning
assert lane_rule([2, 3], 4) == (1, 1, 4.0)  # one completed, the second abandoned
assert lane_rule([2, 2], 4) == (2, 2, 4.0)  # a tie at B completes
assert lane_rule([4], 4) == (1, 1, 4.0)
assert lane_rule([1, 1, 1, 9], 4) == (3, 3, 4.0)
print("1. lane rule: 5 hand-made cases pass")

# 2. Exact enumeration. H and C are dependent: cheap replicates have small H.
LAW = [  # (H, C, probability)
    (Fraction(0), 1, Fraction(3, 10)),
    (Fraction(1), 2, Fraction(3, 10)),
    (Fraction(5), 3, Fraction(2, 10)),
    (Fraction(-2), 6, Fraction(2, 10)),
]
mean = sum(h * p for h, _, p in LAW)


def expectations(budget):
    # E of the lane estimate and of the completed-only average (0 when none completed),
    # summed over every sequence of replicates up to the one that passes the budget.
    lane, pooled = Fraction(0), Fraction(0)

    def walk(prefix, probability):
        nonlocal lane, pooled
        spent = sum(c for _, c in prefix)
        if prefix and spent >= budget:
            costs = [c for _, c in prefix]
            kept, completed, _ = lane_rule(costs, budget)
            lane += probability * sum(h for h, _ in prefix[:kept]) / kept
            if completed:
                pooled += probability * sum(h for h, _ in prefix[:completed]) / completed
            return
        for h, c, p in LAW:
            walk([*prefix, (h, c)], probability * p)

    walk([], Fraction(1))
    return lane, pooled


for budget in range(1, 9):
    lane, pooled = expectations(budget)
    assert lane == mean, (budget, lane, mean)
    print(
        f"2. B = {budget}: E[lane estimate] = {lane} = E[H] exactly; "
        f"completed-only average {pooled} (bias {float(pooled - mean):+.4f})"
    )
assert any(expectations(b)[1] != mean for b in range(1, 9))

# 3. A lane driver that abandons at the budget, against budget_cell.
case = CASES["gaussian"]
rng = np.random.SeedSequence(7979)
runs, lanes = 3, 4
for b in (20, 41, 123):
    cell = budget_cell(case, b, runs, rng, lanes=lanes)
    rounds = b // case.m + 1
    target = case.target()
    for i, g in itertools.product(range(runs), range(lanes)):
        spent, kept_values, completed = 0, [], 0
        for j in range(rounds):
            remaining = b - spent
            index = i * lanes * rounds + j * lanes + g
            if j == 0:
                cap = None  # the first replicate always runs to its end
            else:
                # The largest tau whose cost max(tau, m) + tau - L fits the remainder.
                tau = max(
                    remaining - case.m + case.lag
                    if remaining - case.m + case.lag <= case.m
                    else (remaining + case.lag) // 2,
                    -1,
                )
                if tau < case.lag:
                    break  # nothing can finish: abandoned before it starts
                cap = max(tau + 1, max(case.lag, case.m) + 1)
            run = engine.run(
                target,
                case.kernel,
                case.initial,
                [index],
                k=case.k,
                m=case.m,
                lag=case.lag,
                max_iter=cap,
                rng=rng,
                lanes=1,
            )
            met = np.isfinite(run.meeting_time[0])
            if j and (not met or run.cost[0] > remaining):
                break  # abandoned at the budget
            values = case.h(run.atoms).T @ run.weights
            spent += int(run.cost[0])
            if spent <= b:
                completed += 1
                kept_values.append(values)
            elif j == 0:
                kept_values.append(values)  # none completed: keep the overrun
                break
            if spent >= b:
                break
        estimate = np.mean(kept_values, axis=0)
        assert completed == cell["completed"][i, g], (b, i, g)
        np.testing.assert_allclose(estimate, cell["estimate"][i, g], rtol=1e-12, atol=1e-12)
    print(
        f"3. B = {b}: {runs * lanes} lanes run one replicate at a time with abandonment "
        f"match budget_cell (completed per lane {cell['completed'].ravel().tolist()})"
    )

# 4. pilot_costs against lanes run on consecutive blocks of the same costs.
_, costs, _ = replicates(case, case.target(), np.arange(2000), np.random.SeedSequence(97))
for b in (20, 41, 123):
    charged, n = pilot_costs(costs, b)
    start, by_hand = 0, []
    for _ in range(n):
        end = start
        while costs[start : end + 1].sum() < b:
            end += 1
        by_hand.append(lane_rule(costs[start : end + 1], b)[2])
        start = end + 1
    assert np.isclose(charged, np.mean(by_hand))
    assert n > 10
    print(f"4. B = {b}: pilot mean charged cost {charged:.2f} over {n} lanes agrees")
print("all checks pass")
