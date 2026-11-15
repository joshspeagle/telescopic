"""The targets and cells of x91 (PREREGISTRATION.md, section 4), and the seeds of section 5.

A cell is a target with its settings, the arms that run on it with their numbers of seeds,
and its block of seeds, read from `seeds.json`. `CELLS` lists them in the order of
section 11: A, A', B, B', T, C1 to C5, D3, D2, D1 (tier 1), E1 to E4 (tier 2), F (tier 3).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from i137_common import gaussian_target
from scipy import special, stats
from targets import ExactBanana, banana20, logistic, rotated

from telescopic import examples

SEEDS = json.loads((Path(__file__).parent / "seeds.json").read_text())
CAP = 100_000  # the package's pilot cap
K_MAX = 20_000  # a full run runs only with k at most this (section 4)
REPLICATE_CAP = 10**6  # max_iter of a full run's replicates (section 4)
PAIRS, REPLICATES = 100, 100
HMC_A = ("P", "G", "J2", "J3", "J5", "CF")
MALA_E = ("MP", "MJ5", "MCF")


def _banana_functions(x):
    return x[:, :2]


def _moments(x):
    return np.column_stack([x[:, 0], x[:, 0] ** 2])


def _identity(x):
    return x


@functools.cache
def target(name):
    """(example, initial, given cov or None, test functions, exact values or None) of a target.

    The exact values of the logistic regression are None: its reference comes from
    `logistic_reference`.
    """
    if name.startswith("gaussian"):  # "gaussian<d>"
        ex = examples.gaussian(int(name[8:]))
        return ex, ex.initial, None, _moments, (0.0, 1.0)
    if name.startswith("unequal"):  # N(0, diag(s^2)), s from 1 to 1/4 (d = 10) or 1/5 (d = 8)
        d = int(name[7:])
        ex = gaussian_target(d, np.geomspace(1, {10: 0.25, 8: 0.2}[d], d))
        return ex, ex.initial, np.eye(d), None, None
    if name == "banana_exact":
        return examples.banana(), ExactBanana(2), None, _banana_functions, (1.0, 1.5)
    if name == "banana_U":
        ex = examples.banana()
        return ex, ex.initial, None, _banana_functions, (1.0, 1.5)
    if name == "banana20":
        return banana20(), ExactBanana(20), None, _banana_functions, (1.0, 1.5)
    if name == "mixture":
        ex = examples.mixture()
        return ex, ex.initial, None, _moments, (0.0, 17.0)
    if name == "logistic":
        ex = logistic()[0]
        return ex, ex.initial, None, _identity, None
    if name == "rotated20":
        ex = rotated(d=20)
        return ex, ex.initial, None, None, None
    if name == "funnel10":
        ex = examples.funnel(10)
        return ex, ex.initial, None, None, None
    raise KeyError(name)


def logistic_reference(seed=SEEDS["logistic_reference"], draws=10**6):
    """The logistic regression's posterior means by self-normalized importance sampling.

    From a multivariate t with 5 degrees of freedom at the posterior mode, with scale the
    inverse of the exact Hessian of -log pi there, `draws` draws from
    `np.random.default_rng(seed)`; tenfold more while the effective sample size is below
    10^5 (section 4). The standard error is the delta method's,
    sqrt(sum w^2 (x - mean)^2) / sum w.

    Returns
    -------
    dict with mean (6,), se (6,), ess and draws
    """
    ex, X, _, fit = logistic()
    mode = fit.x
    for _ in range(20):  # Newton's method from BFGS's fit, to the mode within rounding
        p = special.expit(X @ mode)
        hessian = X.T @ ((p * (1 - p))[:, None] * X) + np.eye(6) / 25
        mode = mode + np.linalg.solve(hessian, ex.grad(mode))
    p = special.expit(X @ mode)
    hessian = X.T @ ((p * (1 - p))[:, None] * X) + np.eye(6) / 25
    proposal = stats.multivariate_t(mode, np.linalg.inv(hessian), df=5)
    while True:
        x = proposal.rvs(size=draws, random_state=np.random.default_rng(seed))
        log_w = np.concatenate(
            [ex.log_prob(c) - proposal.logpdf(c) for c in np.array_split(x, draws // 10**5)]
        )
        w = np.exp(log_w - log_w.max())
        mean = w @ x / w.sum()
        se = np.sqrt((w**2) @ (x - mean) ** 2) / w.sum()
        ess = w.sum() ** 2 / (w**2).sum()
        if ess >= 1e5:
            return {"mean": mean.tolist(), "se": se.tolist(), "ess": float(ess), "draws": draws}
        draws *= 10


def seeds(block):
    first, last = SEEDS[block]
    return tuple(range(first, last + 1))


@dataclass(frozen=True)
class Cell:
    part: str  # A, A', B, B', T, C, D, E or F
    name: str  # unique, as "A gaussian2 n=2"
    target: str
    n_steps: int | None
    run: str  # "pilot", "full" or "tune"
    arms: dict  # arm -> number of seeds, the first of the block
    seeds: tuple
    cap: int = CAP
    pilot_arms: tuple = ()  # arms that run the pilot only in a full cell (C5's P.2, CF.2)
    of_T: bool = False  # one of the nine cells of T, where L applies by name

    @property
    def tier(self):
        return {"E": 2, "F": 3}.get(self.part, 1)


def _n(n_steps):
    return "default" if n_steps is None else str(n_steps)


def _cells():
    cells = []
    resonant_A = {(2, 2), (10, 3), (10, 10)}
    for d in (2, 10):
        for n in (None, 1, 2, 3, 5, 10):
            arms = dict.fromkeys(HMC_A, 30)
            if (d, n) in resonant_A:
                arms["P"] = 5
            if n is None:
                arms |= {"P.2": 30, "CF.2": 30}
            cells.append(
                Cell("A", f"A gaussian{d} n={_n(n)}", f"gaussian{d}", n, "pilot", arms, seeds("A"),
                     of_T=(d, n) in resonant_A)
            )  # fmt: skip
    for n in (3, 4, 6):
        arms = {"P": 5, "G": 30, "J2": 30, "J3": 30, "J5": 30}
        cells.append(
            Cell("A'", f"A' gaussian5 n={n}", "gaussian5", n, "pilot", arms, seeds("A'"),
                 of_T=n in (3, 6))
        )  # fmt: skip
    for n in (None, 3, 10):
        arms = dict.fromkeys(HMC_A, 30)
        cells.append(
            Cell(
                "B", f"B unequal10 n={_n(n)}", "unequal10", n, "pilot", arms, seeds("B"), of_T=True
            )
        )
    arms = dict.fromkeys(("P", "G", "J2", "J3", "J5"), 30)
    cells.append(Cell("B'", "B' unequal8 n=default", "unequal8", None, "pilot", arms, seeds("B'"),
                      of_T=True))  # fmt: skip
    # T: G's tuning at 300 seeds of each of the nine cells, c = 0 to 8 in this order.
    order = [
        "A gaussian2 n=2", "A gaussian10 n=3", "A gaussian10 n=10",
        "B unequal10 n=default", "B unequal10 n=3", "B unequal10 n=10",
        "A' gaussian5 n=3", "A' gaussian5 n=6", "B' unequal8 n=default",
    ]  # fmt: skip
    by_name = {c.name: c for c in cells}
    for c, name in enumerate(order):
        base = by_name[name]
        cells.append(
            Cell("T", f"T{c} {name}", base.target, base.n_steps, "tune", {"G": 300},
                 seeds(f"T{c}"), of_T=True)
        )  # fmt: skip
    for block, tgt, n, count in (
        ("C1", "banana_exact", 2, 20),
        ("C2", "banana_exact", None, 10),
        ("C3", "mixture", 2, 20),
        ("C4", "mixture", None, 20),
    ):
        cells.append(Cell("C", f"{block} {tgt} n={_n(n)}", tgt, n, "full",
                          dict.fromkeys(HMC_A, count), seeds(block)))  # fmt: skip
    cells.append(
        Cell("C", "C5 logistic n=default", "logistic", None, "full",
             dict.fromkeys((*HMC_A, "P.2", "CF.2"), 20), seeds("C5"), pilot_arms=("P.2", "CF.2"))
    )  # fmt: skip
    cells.append(Cell("D", "D3 banana_U n=default", "banana_U", None, "full",
                      {"P": 3, "G": 10, "J5": 5, "CF": 10}, seeds("D3")))  # fmt: skip
    cells.append(Cell("D", "D2 banana20 n=default", "banana20", None, "full",
                      {"P": 5, "G": 5, "J5": 5, "CF": 10}, seeds("D2")))  # fmt: skip
    cells.append(Cell("D", "D1 rotated20 n=default", "rotated20", None, "pilot",
                      dict.fromkeys((*HMC_A, "P.2", "CF.2"), 10), seeds("D1")))  # fmt: skip
    for block, tgt, arms in (
        ("E1", "banana_U", {"MP": 3, "MJ5": 5, "MCF": 5}),
        ("E2", "logistic", dict.fromkeys(MALA_E, 10)),
        ("E3", "gaussian10", dict.fromkeys(MALA_E, 10)),
        ("E4", "banana_exact", dict.fromkeys(MALA_E, 10)),
    ):
        cells.append(Cell("E", f"{block} {tgt}", tgt, None, "full", arms, seeds(block)))
    cells.append(Cell("F", "F funnel10 n=default", "funnel10", None, "pilot",
                      {"P": 3, "J5": 3, "CF": 3}, seeds("F"), cap=10_000))  # fmt: skip
    return cells


CELLS = _cells()
CELL = {c.name: c for c in CELLS}
# The Gaussians of parts A, A', B, B', D1 and T, whose HMC records hold the exact whitened
# frequencies and the largest factor (section 6).
GAUSSIANS = ("gaussian", "unequal", "rotated")
