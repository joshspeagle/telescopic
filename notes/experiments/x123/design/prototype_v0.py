"""The first, superseded version of the design round's prototype of the gradient check (#123).

It is kept only so that ``fa_run.py fa1 prototype_v0`` reproduces ``fa_fa1_v0.out``, which
DESIGN.md cites for what this version got wrong: it pools the rounding estimate in relative
size only, and rounds every point to the narrowest dtype. The final prototype is
``../prototype.py``. This file's notes, as written for this version, follow.

Prototype of the candidate gradient checks of the second design round (#123). Not package code.

One function, ``check``, with the candidates as settings:

(N) noise-limited:  check(..., floor=None)        warns iff |G - D| > SAFETY * (T + R_f + R_g) on a line
(H) harm floor:     check(..., floor=PHI)         warns iff |G - D| > max(that, h * PHI)
(B') the first experiment's directions with (N)'s threshold: check(..., direction="signs")

Points x_p, p = 1..P, and for each two further draws a_p, b_p of the initial distribution, all
from the generator replicate_generator(root_seed(rng), 3, 0), which no run reads. The line at
x_p is x_p + t h u_p, t in {-2, -1, 1, 2}, u_p = a_p - b_p, h = eps_f^(1/3). Everything is per
unit of t, with the points as rounded:

  D  = (f(+1) - f(-1)) / w1                       the difference of log_prob
  G  = grad(x_p) . (x(+1) - x(-1)) / w1           what grad predicts for it
  T  = |(D - G) - (D2 - G2)|                      truncation estimate (D2, G2 at t = +-2)
  F4 = f(-2) - 4 f(-1) + 6 f(0) - 4 f(1) + f(2)   fourth difference
  R_f = (2 / w1) max(r_hat L, |F4| / sqrt(70), ULPS eps_f L),  L = max |f| on the line,
        r_hat^2 = mean over usable lines of (F4 / L)^2 / 70
  R_g = ULPS eps_g sum_i |grad_i| |x(+1) - x(-1)|_i / w1

Where log_prob or grad returns a dtype narrower than float64, every point at which they are
evaluated is first rounded to that dtype.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from telescopic._random import replicate_generator, root_seed
from telescopic.initial import as_initial, draw
from telescopic.target import Target

POINTS = 8
ULPS = 64.0
SAFETY = 10.0
DELTA_STAR = 0.125  # drift error, in proposal standard deviations, not worth a warning
PHI = 2 * np.sqrt(2) * DELTA_STAR  # (H): nats mispredicted over u = a - b
CHECK_NAMESPACE = 3
EPS64 = np.finfo(np.float64).eps
T_STENCIL = np.array([1.0, -1.0, 2.0, -2.0])


@dataclass
class Verdict:
    warned: bool = False
    lines: int = 0
    lines_unusable: int = 0
    points_skipped: int = 0
    ratio: float = (
        0.0  # largest gap / tolerance over the usable lines (inf for gap > 0 = tolerance)
    )
    n_log_prob: int = 0
    n_grad: int = 0
    dtype_f: str = "float64"
    dtype_g: str = "float64"
    detail: list = field(
        default_factory=list
    )  # per usable line: dict of gap, T, R_f, R_g, floor, D, L


class _Recorder:
    def __init__(self, function):
        self.function, self.itemsize = function, 8

    def __call__(self, *a, **k):
        out = self.function(*a, **k)
        dtype = np.asarray(out).dtype
        if dtype.kind == "f":
            self.itemsize = min(self.itemsize, dtype.itemsize)
        return out


def draws_of(initial, seed, points):
    """The 3 P draws (x_p, a_p, b_p) and the signs used by the fallbacks, from the check's stream."""
    initial = as_initial(initial)
    generator = replicate_generator(root_seed(seed), CHECK_NAMESPACE, 0)
    first = draw(initial, generator)
    rest = [draw(initial, generator, len(first)) for _ in range(3 * points - 1)]
    every = np.array([first, *rest]).reshape(points, 3, len(first))
    signs = generator.choice([-1.0, 1.0], size=(points, len(first)))
    return every[:, 0], every[:, 1], every[:, 2], signs


def directions(x, a, b, signs, kind="draws"):
    every = np.concatenate([x, a, b])
    spread = np.ptp(every, axis=0)
    if kind == "draws":
        u = a - b
    else:  # "signs": the first experiment's direction, the spread times random signs
        sd = np.where(spread > 0, every.std(axis=0), 0.0)
        u = sd * signs
    # A coordinate where the two draws are equal: the range of the draws, else |x|, else 1.
    fallback = np.where(spread > 0, spread, np.where(x != 0, np.abs(x), 1.0))
    u = np.where(u != 0, u, signs * fallback)
    # A step must not round away next to a large coordinate.
    smallest = np.sqrt(EPS64) * np.abs(x)
    return np.where(np.abs(u) < smallest, np.copysign(smallest, u), u)


def check(
    log_prob,
    grad,
    initial,
    seed,
    *,
    vectorize=True,
    points=POINTS,
    floor=None,
    direction="draws",
    safety=SAFETY,
    ulps=ULPS,
    step=None,
    args=(),
    kwargs=None,
):
    lp, gr = _Recorder(log_prob), _Recorder(grad)
    target = Target(lp, gr, args=args, kwargs=kwargs or {}, vectorize=vectorize)
    x, a, b, signs = draws_of(initial, seed, points)
    u = directions(x, a, b, signs, direction)
    verdict = Verdict()

    def centers(x):
        out = []
        for row in x:
            zeroed = target.n_grad_zeroed
            f0, g = target.log_prob_and_grad(row[None])
            out.append((f0[0], g[0], target.n_grad_zeroed > zeroed))
        return out

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*returned float")
        base = centers(x)
        narrow = min(lp.itemsize, gr.itemsize)
        if narrow < 8:  # evaluate only at points the narrow dtype represents
            narrow_dtype = np.dtype(f"f{narrow}")
            x = x.astype(narrow_dtype).astype(np.float64)
            base = centers(x)
        eps_f = np.finfo(np.dtype(f"f{lp.itemsize}")).eps
        eps_g = np.finfo(np.dtype(f"f{gr.itemsize}")).eps
        h = (max(eps_f, np.finfo(np.dtype(f"f{narrow}")).eps) ** (1 / 3)) if step is None else step
        verdict.dtype_f, verdict.dtype_g = f"float{8 * lp.itemsize}", f"float{8 * gr.itemsize}"
        rows = []
        for p in range(points):
            f0, g, zeroed = base[p]
            if f0 == -np.inf or zeroed:
                verdict.points_skipped += 1
                continue
            v = h * u[p]
            stencil = x[p] + T_STENCIL[:, None] * v
            if narrow < 8:
                stencil = stencil.astype(narrow_dtype).astype(np.float64)
            values = target.log_prob(stencil)
            moved = stencil - x[p]
            d1, d2 = moved[0] - moved[1], moved[2] - moved[3]
            vv = v @ v
            w1, w2 = d1 @ v / vv, d2 @ v / vv
            verdict.lines += 1
            if not (np.isfinite(values).all() and w1 > 0 and w2 > w1):
                verdict.lines_unusable += 1
                continue
            D, D2 = (values[0] - values[1]) / w1, (values[2] - values[3]) / w2
            G, G2 = g @ d1 / w1, g @ d2 / w2
            F4 = values[2] + values[3] - 4 * (values[0] + values[1]) + 6 * f0
            L = max(np.abs(values).max(), abs(f0))
            rows.append(
                {
                    "p": p,
                    "D": D,
                    "G": G,
                    "gap": abs(G - D),
                    "T": abs((D - G) - (D2 - G2)),
                    "F4": F4,
                    "L": L,
                    "w1": w1,
                    "Rg": ulps * eps_g * (np.abs(g) @ np.abs(d1)) / w1,
                    "h": h,
                }
            )
    if rows:
        rel = np.array([r["F4"] / r["L"] if r["L"] > 0 else 0.0 for r in rows])
        r_hat = float(np.sqrt(np.mean(rel**2) / 70))
        for r in rows:
            r["Rf"] = (2 / r["w1"]) * max(
                r_hat * r["L"], abs(r["F4"]) / np.sqrt(70), ulps * eps_f * r["L"]
            )
            r["tol"] = safety * (r["T"] + r["Rf"] + r["Rg"])
            r["floor"] = 0.0 if floor is None else r["h"] * floor
            bar = max(r["tol"], r["floor"])
            r["warn"] = bool(r["gap"] > bar)  # strict: 0 > 0 is no warning
            ratio = r["gap"] / bar if bar > 0 else (np.inf if r["gap"] > 0 else 0.0)
            verdict.ratio = max(verdict.ratio, float(ratio))
            verdict.warned |= r["warn"]
    verdict.detail = rows
    verdict.n_log_prob, verdict.n_grad = target.n_log_prob, target.n_grad
    return verdict


def N(log_prob, grad, initial, seed, **k):
    return check(log_prob, grad, initial, seed, **k)


def H(log_prob, grad, initial, seed, **k):
    return check(log_prob, grad, initial, seed, floor=PHI, **k)
