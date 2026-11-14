"""Exploratory: false alarms of (N) and (H) on correct gradients (seeds 200000..209899).

Question: without the 1% floor, does the error model alone keep (N) silent on the first
experiment's 44 correct families, on its 7 out-of-reach families, on the review's 42 probes
(cases.REVIEW_CORRECT) and on targets computed wholly in float32?
Design: 100 seeds per family; group bases 200000 (cases.CORRECT), 204400 (cases.OUT_OF_REACH),
205100 (the group "fa1": cases.REVIEW_CORRECT), 209300 (float32 throughout); family f of a
group at base + 100 f + (0..99). The second argument names the module of the prototype:
prototype (the final one, the default), or prototype_v0 or prototype_v1, the two superseded
versions, which exist only to reproduce fa_fa1_v0.out and fa_fa1_v1.out.
usage: fa_run.py correct|reach|fa1|all32 [prototype|prototype_v0|prototype_v1]
Result: see fa_*.out and DESIGN.md section 6.
"""

import importlib
import sys
import warnings

import cases
import numpy as np

prototype = importlib.import_module(sys.argv[2] if len(sys.argv) > 2 else "prototype")


def groups(name):
    if name == "correct":
        return 200000, {k: (v, True) for k, v in cases.CORRECT.items()}
    if name == "reach":
        return 204400, {k: (v, True) for k, v in cases.OUT_OF_REACH.items()}
    if name == "fa1":
        return 205100, cases.REVIEW_CORRECT
    if name == "all32":
        return 209300, {
            f"float32 throughout, N({m:g}, I) d=5": (cases.all32(m), True)
            for m in (0.0, 10.0, 100.0, 300.0, 1e3, 1e4)
        }


if __name__ == "__main__":
    base, families = groups(sys.argv[1])
    n = 100
    total = {"N": 0, "H": 0}
    print(
        f"group {sys.argv[1]}: family f at seeds {base} + {n} f + (0..{n - 1}); P = {prototype.POINTS}"
    )
    for f, (name, (case, vec)) in enumerate(families.items()):
        out = []
        for key, design in (("N", prototype.N), ("H", prototype.H)):
            w = unusable = skipped = 0
            ratios = []
            for seed in range(base + n * f, base + n * (f + 1)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = design(*cases.resolve(case, seed), seed, vectorize=vec)
                w += v.warned
                ratios.append(v.ratio)
                unusable += v.lines_unusable
                skipped += v.points_skipped
            total[key] += w
            out.append(
                f"{key} {w:>3}/{n} ratio med/max {np.median(ratios):8.2e}/{np.max(ratios):8.2e}"
            )
        print(
            f"[{f:>2}] {name:<64} | {out[0]} | {out[1]} | unusable {unusable} skipped {skipped} evals {v.n_log_prob}+{v.n_grad}",
            flush=True,
        )
    print(f"total warnings: N {total['N']}, H {total['H']} of {n * len(families)}")
