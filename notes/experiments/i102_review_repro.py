"""The adversarial review: rerun eight jobs of the confirmation and compare with its rows.

Reruns i102_q2_rules.one for eight (cell, seed) jobs of i102_q3_confirm.py and compares
every field with the saved row, then evaluates the rule alone on three jobs that ran past
the confirmation's cap. The reviewer's version also checked, in the experiment's working
folder, that the compiled modules were compiled from the sources as they stand, a check
that has no meaning in the repository and is left out here.

Result (i102_review_repro.out): all eight rows are reproduced exactly, and the rule's
statistic on the three capped jobs is 27.7, 26.7 and 10.8 with R-hat silent, as
i102_q3_confirm.out records. This output was captured when the experiment was recorded,
at commit 08cadff, from rows written at c961089; it takes under a minute.

Run: EXP=<folder where i102_q3_confirm.py ran> python notes/experiments/i102_review_repro.py
"""

import os
import pickle
from pathlib import Path

import numpy as np

EXP = os.environ["EXP"]
import i102_q2_rules as q2_rules

MS = (4, 8, 16, 32)
DIVS = (64, 32, 16, 8, 4, 2)
rng = np.random.default_rng(20261113)
names = list(q2_rules.CELLS)
picks = [
    ("bimodal d=10 s=12", 49),
    ("bimodal d=32 s=6", 53),
    ("bimodal d=32 s=12", 46),
    ("ex pumps tight", 304),
    ("student4 d=10 wide", 311),
    ("gauss d=10 cond=100 tight", 307),
    ("ex banana default", 302),
    ("bimodal d=10 s=6", 51),
]
for name, seed in picks:
    row = q2_rules.one((name, seed))
    saved = pickle.loads(
        Path(q2_rules.path_of(os.path.join(EXP, "rows_q3"), (name, seed))).read_bytes()
    )
    same = all(
        (
            np.array_equal(row[k], saved[k], equal_nan=True)
            if isinstance(row[k], np.ndarray)
            else (
                row[k] == saved[k]
                or (isinstance(row[k], float) and np.isnan(row[k]) and np.isnan(saved[k]))
            )
        )
        for k in row
        if k not in ("secs", "rss_mb")
    )
    print(
        name,
        seed,
        "reproduces the saved row exactly:",
        same,
        "| z",
        row["z"],
        "stat",
        row["stats"][1, 3, 0],
        "disagree",
        row["disagree"],
    )
for name, seed in (("bimodal d=10 s=12", 44), ("bimodal d=10 s=12", 47), ("bimodal d=2 s=6", 44)):
    row = q2_rules.one((name, seed, True))
    print(
        "rule only:",
        name,
        seed,
        "stat",
        round(float(row["stats"][1, 3, 0]), 1),
        "R-hat warn",
        row["disagree"],
    )
