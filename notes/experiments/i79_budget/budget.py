"""Prototype of budget mode (#79): GH91's per-lane estimator, a lane as the processor.

Lane g of G runs the replicate indices g, g + G, g + 2G, ... in sequence and stops at a
budget B of kernel steps: it keeps the replicates completed within B, abandons the one in
progress, and runs its first to completion if none completed (PREREGISTRATION.md,
section 1). Every replicate comes from telescopic.engine.run, unchanged.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from telescopic import Target, engine, examples
from telescopic.initial import Normal
from telescopic.kernels import HMC, RandomWalk
from telescopic.measure import SignedMeasure

LANES = 32  # G, the budget lanes
BUDGETS = (1, 3, 10)  # B in units of the pilot's mean cost
PILOT = 20_000
CHUNK = 50_000  # replicates per engine call, to bound the memory of the atoms


@dataclass(frozen=True)
class Case:
    example: Any
    kernel: Any
    initial: Any
    k: int
    m: int
    lag: int
    coordinate: int
    runs: int
    gradient: bool = False

    @property
    def truth(self):
        j = self.coordinate
        return np.array([self.example.mean[j], self.example.cov[j, j] + self.example.mean[j] ** 2])

    def h(self, x):
        y = x[:, self.coordinate]
        return np.stack([y, y**2], axis=1)

    def target(self):
        grad = self.example.grad if self.gradient else None
        return Target(self.example.log_prob, grad=grad, vectorize=True)


GAUSSIAN = examples.gaussian(5, 0.5)
BANANA, MIXTURE = examples.banana(), examples.mixture()
CASES = {
    "gaussian": Case(
        GAUSSIAN, RandomWalk(GAUSSIAN.cov), Normal(np.full(5, 3.0), np.eye(5)), 2, 15, 3, 0, 1000
    ),
    "banana": Case(
        BANANA,
        HMC(0.02, cov=np.eye(2)),
        Normal([3.0, 9.0], 0.01 * np.eye(2)),
        10,
        100,
        10,
        1,
        500,
        gradient=True,
    ),
    "mixture": Case(
        MIXTURE, RandomWalk(np.eye(1), scale=2.0), MIXTURE.initial, 50, 200, 1, 0, 1000
    ),
}
ARMS = ("pilot", "budget", "fixed")


def seed(name, arm, b=0):
    return np.random.SeedSequence(79, spawn_key=(list(CASES).index(name), ARMS.index(arm), b))


def replicates(case, target, indices, rng):
    """H (n, 2), cost (n,) and meeting time (n,) of the replicates indices, in their order."""
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    values, cost, tau = np.empty((len(indices), 2)), np.empty(len(indices)), np.empty(len(indices))
    for start in range(0, len(indices), CHUNK):
        chunk = order[start : start + CHUNK]
        run = engine.run(
            target,
            case.kernel,
            case.initial,
            indices[chunk],
            k=case.k,
            m=case.m,
            lag=case.lag,
            rng=rng,
            lanes=min(256, len(chunk)),
        )
        assert np.array_equal(run.index, indices[chunk])  # sorted, all met
        position = np.searchsorted(run.index, run.owner)
        measure = SignedMeasure(run.atoms, run.weights, position, len(run.index))
        values[chunk] = measure.per_replicate(case.h)
        cost[chunk] = run.cost
        tau[chunk] = run.meeting_time
    return values, cost, tau


def lane_rule(costs, budget):
    """(kept, completed, charged) of a lane whose replicates cost costs, in order.

    completed is N = max{n : S_n <= B}; kept is max(1, N); charged is B, or the first
    replicate's cost when N = 0. costs must reach past B, or end exactly at it.
    """
    completed = int(np.searchsorted(np.cumsum(costs), budget, side="right"))
    if completed == 0:
        return 1, 0, float(costs[0])
    return completed, completed, float(budget)


def budget_cell(case, budget, runs, rng, lanes=LANES):
    """Lane-level records of runs independent budget runs of lanes lanes each.

    Run i, lane g, round j uses the index i G J + j G + g. Round j runs, in one engine
    call, the j-th replicate of every lane still inside its budget. Each replicate runs to
    its meeting, but an abandoned one only contributes its cost, to say it was abandoned.
    """
    target = case.target()
    rounds = budget // max(case.m, 1) + 1  # J: every cost is at least max(m, 1)
    total = runs * lanes
    lane = np.arange(total)
    spent = np.zeros(total)
    completed = np.zeros(total, dtype=np.int64)
    completed_sum = np.zeros((total, 2))
    completed_tau = np.zeros(total)
    first = np.zeros((total, 2))
    first_cost, first_tau = np.zeros(total), np.zeros(total)
    active = np.ones(total, dtype=bool)
    computed = 0
    for j in range(rounds):
        now = lane[active]
        if not len(now):
            break
        indices = (now // lanes) * lanes * rounds + j * lanes + now % lanes
        values, cost, tau = replicates(case, target, indices, rng)
        computed += len(now)
        if j == 0:
            first[now], first_cost[now], first_tau[now] = values, cost, tau
        spent[now] += cost
        inside = spent[now] <= budget
        done = now[inside]
        completed[done] += 1
        completed_sum[done] += values[inside]
        completed_tau[done] += tau[inside]
        active[now] = spent[now] < budget
    assert not active.any()
    some = completed > 0
    kept = np.maximum(completed, 1)
    estimate = np.where(some[:, None], completed_sum / kept[:, None], first)
    lane_tau = np.where(some, completed_tau / kept, first_tau)
    charged = np.where(some, float(budget), first_cost)
    shape = (runs, lanes)
    return {
        "estimate": estimate.reshape(*shape, 2),
        "completed": completed.reshape(shape),
        "completed_sum": completed_sum.reshape(*shape, 2),
        "completed_tau": completed_tau.reshape(shape),
        "first_tau": first_tau.reshape(shape),
        "lane_tau": lane_tau.reshape(shape),
        "charged": charged.reshape(shape),
        "budget": np.array(budget),
        "computed": np.array(computed),
    }


def fixed_cell(case, size, runs, rng):
    """Per-run records of runs fixed-R runs of size replicates, indices i R to (i + 1) R - 1."""
    target = case.target()
    values, cost, tau = replicates(case, target, np.arange(runs * size), rng)
    values = values.reshape(runs, size, 2)
    return {
        "estimate": values.mean(axis=1),
        "sd": values.std(axis=1, ddof=1),
        "cost": cost.reshape(runs, size).sum(axis=1),
        "tau_sum": tau.reshape(runs, size).sum(axis=1),
        "size": np.array(size),
        "computed": np.array(runs * size),
    }


def pilot_costs(costs, budget):
    """Mean charged cost of a lane, from costs allotted to lanes in consecutive blocks.

    A lane's block ends at its first replicate whose cost takes it past the budget (or
    exactly to it); each block boundary is a stopping time of the i.i.d. sequence, so the
    blocks are i.i.d. The incomplete last block is dropped.
    """
    charged, start, total = [], 0, np.cumsum(costs)
    while True:
        before = total[start - 1] if start else 0.0
        end = int(np.searchsorted(total, before + budget, side="left"))  # S >= B here
        if end >= len(costs):
            break
        charged.append(lane_rule(costs[start : end + 1], budget)[2])
        start = end + 1
    return float(np.mean(charged)), len(charged)
