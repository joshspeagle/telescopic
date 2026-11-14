"""What every probe of the review of experiment x123 uses: one row of counts per family.

``run`` calls ``gradcheck.check`` once per seed on a family, as the registered
``run_checks.py`` does, and prints the warnings of (N) and of (H), the median and the
largest ratio of gap to tolerance of (N), the unusable lines, the skipped points, the
checks in which no line was compared, the errors raised by the family inside the check,
and the time. A family is ``(log_prob, grad, initial)`` or a callable of the seed that
returns one. The review's seeds are 500000 and above; no registered seed is read.
"""

import time
import warnings

import gradcheck as G
import numpy as np


def run(name, case, seeds, vectorize=True, quiet=False):
    n = N = H = err = unus = skip = nolines = 0
    ratios = []
    last = ""
    t0 = time.perf_counter()
    for seed in seeds:
        c = case(seed) if callable(case) else case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                v = G.check(*c, seed, vectorize=vectorize)
            except G.GradientCheckError as e:  # the family raised inside the check
                err += 1
                last = repr(e)[:150]
                continue
        n += 1
        N += v.warns(0.0)
        H += v.warns(G.PHI)
        ratios.append(v.ratio(0.0))
        unus += v.lines_unusable
        skip += v.points_skipped
        nolines += len(v.lines) == 0
    r = np.array(ratios) if ratios else np.array([np.nan])
    out = {
        "name": name,
        "n": n,
        "N": N,
        "H": H,
        "err": err,
        "unusable": unus,
        "skipped": skip,
        "nolines": nolines,
        "med": float(np.median(r)),
        "max": float(np.max(r)),
        "last": last,
    }
    if not quiet:
        print(
            f"{name[:78]:<78} | N {N:>4}/{n:<4} H {H:>4}/{n:<4} | ratio med {out['med']:.2e} max {out['max']:.2e} | unusable {unus} skipped {skip} nolines {nolines} | errors {err} {last} | {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    return out
