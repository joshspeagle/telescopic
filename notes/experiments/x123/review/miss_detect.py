"""Review of x123: detection of the wrong gradients of cases5.py, and silence on their twins.

Question: how often do (N) and (H) warn on wrong gradients that are right, or orthogonal
to the lines, where the chains start, or that hide behind a large or single-precision
log_prob? And do they stay silent on the correct gradient of the same target and start?

Design: every family of cases5.M, the correct twin and the wrong gradient, on seeds
500000 to 500199 (500000 to 500099 at d = 10000), one call of gradcheck.check per seed.
Columns: the warnings of (N) and of (H) on the wrong gradient, its median ratio of gap to
tolerance, the checks in which no line was compared and the errors; then the warnings on
the correct twin and its largest ratio.

Run: PYTHONPATH=src:notes/experiments/x123 python notes/experiments/x123/review/miss_detect.py > miss_detect.out

Result: see miss_detect.out and REVIEW.md, objections 5 and 6. No correct twin warns.
(N) warns on 0 of 200 from a point start at a mode, with a coordinate held at 0, where
the error lies outside the start or orthogonal to it, behind a constant of 1e10 and from
a start on a boundary; with a float32 log_prob it warns on 196, 37 and 0 on the sign of
one coordinate at d = 200, 1000 and 10000. harm5.py measures the harm of these families.
"""

import cases5
from probe import run

S = range(500_000, 500_200)
for name, (right, wrong) in cases5.M.items():
    seeds = S if "d=10000" not in name else range(500_000, 500_100)
    r = run("RIGHT " + name, right, seeds, quiet=True)
    w = run(name, wrong, seeds, quiet=True)
    print(
        f"{name[:92]:<92} | wrong: N {w['N']:>3}/{w['n']} H {w['H']:>3}/{w['n']} ratio med {w['med']:.2e} nolines {w['nolines']} err {w['err']} | right: N {r['N']}/{r['n']} H {r['H']}/{r['n']} max {r['max']:.2e} {w['last']}",
        flush=True,
    )
