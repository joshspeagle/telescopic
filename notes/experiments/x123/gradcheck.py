"""The two gradient checks of the second pre-registered experiment of #123: (N) and (H).

Question: does a check of the user's ``grad`` against central differences of
``log_prob``, run when a sampler is built, stay silent on correct gradients and warn on
harmful wrong ones, at a cost that does not grow with the dimension? See
PREREGISTRATION.md for the criteria and the result.

Design. From the generator ``replicate_generator(root_seed(rng), 3, 0)``, a namespace
that no run of the sampler reads, the check draws ``3 P`` points of the initial
distribution, ``x_p, a_p, b_p`` for ``p = 1..P`` in that order, then ``2 P d`` random
signs (the first ``P d`` for the fallback directions, the last ``P d`` for the probe of
a narrow gradient). The line at ``x_p`` has direction ``u_p = a_p - b_p`` and step
vector ``v_p = h u_p``, ``h = eps_f^(1/3)``, with ``eps_f`` the epsilon of the dtype of
``log_prob``; its points are ``x_p + t v_p``, ``t`` in ``{1, -1, 2, -2}``. Where
``u_pi = 0`` it is a random sign times the range of the ``3 P`` draws in coordinate
``i``, else ``|x_pi|``, else 1; where ``|u_pi| < sqrt(eps_64) |x_pi|`` it is raised to
that size. Per unit of ``t``, with the points as rounded (``m_t`` the rounded
displacement, ``w_1`` and ``w_2`` the rounded widths along ``v``, about 2 and 4):

    D  = (f_1 - f_-1) / w_1,         G  = g(x_p) . (m_1 - m_-1) / w_1,
    D' = (f_2 - f_-2) / w_2,         G' = g(x_p) . (m_2 - m_-2) / w_2,
    gap = |G - D|,   T = |(D - G) - (D' - G')|,
    F4 = f_-2 - 4 f_-1 + 6 f_0 - 4 f_1 + f_2,   L = max |f| on the line,
    R_f = (2 / w_1) max(r L, s, |F4| / sqrt(70), U eps_f L),
          r^2 = mean over usable lines of (F4 / L)^2 / 70,  s^2 = mean of F4^2 / 70,
    R_g = sum_i max(U eps_g |g_i|, q_i) |m_1 - m_-1|_i / w_1,

where ``q_i`` is the movement of ``grad`` when its input is multiplied by
``1 + 4 eps_g s`` (``s`` random signs), measured only where ``grad`` returns a dtype
narrower than float64, and 0 otherwise. A line warns when

    (N)  gap > S (T + R_f + R_g),
    (H)  gap > max(S (T + R_f + R_g), h PHI),   PHI = 2 sqrt(2) / 8 nats,

and the check warns when any usable line does. A line is unusable when a value on it is
``-inf`` or ``w_1 <= 0`` or ``w_2 <= w_1``; a point is skipped when its density is zero
or its gradient is not finite. Where ``log_prob`` returns a dtype narrower than
float64, ``x_p`` and the points of each line are rounded to it before use, and ``x_p``
is evaluated again after rounding.

The check holds one line at a time: its memory is a fixed number of vectors of length
``d`` and its cost is ``5 P`` evaluations of ``log_prob`` and ``P`` of ``grad`` at any
dimension (``P`` more of each for a narrow ``log_prob``, ``P`` more for a narrow
``grad``). It draws the ``3 P`` points twice, once for the range of the fallback and
once for the lines, with no evaluation in the first pass. It builds a ``Target`` of its
own, so the sampler's counters and its once-only precision warning are untouched. An
error raised by the user's functions inside the check is raised again as
``GradientCheckError``, which names the check and ``check_grad=False``.

Every line keeps its gap, its tolerance ``S (T + R_f + R_g)``, its step ``h`` and the
norm ``|g o (m_1 - m_-1)| / w_1``, so that a relative floor ``rho`` on that norm is
evaluated from the same check (``Verdict.warns``).

This file is the design round's prototype (``prototype.py``) rewritten to hold one line
at a time; on seeds 220000 to 229999 it gives the prototype's gaps, tolerances and
decisions line for line (``equivalence.py``).
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from telescopic._random import replicate_generator, root_seed
from telescopic.initial import as_initial, draw
from telescopic.target import Target

POINTS = 8  # P: lines, one per point
ULPS = 64.0  # U: units in the last place below which rounding is not resolved
SAFETY = 10.0  # S: a line warns above S times its estimated error
DELTA_STAR = 0.125  # (H): a drift error below an eighth of a proposal standard deviation
PHI = 2 * np.sqrt(2) * DELTA_STAR  # (H): its floor in nats over u = a - b
CHECK_NAMESPACE = 3  # the replicate streams of runs use namespaces 0, 1 and 2
EPS64 = np.finfo(np.float64).eps
T_STENCIL = np.array([1.0, -1.0, 2.0, -2.0])


class GradientCheckError(RuntimeError):
    """An error raised by ``log_prob`` or ``grad`` inside the gradient check."""


@dataclass
class Line:
    """The comparison on one usable line, per unit of ``t``, in nats."""

    point: int
    gap: float
    tolerance: float  # S (T + R_f + R_g)
    step: float  # h
    norm: float  # |g o (m_1 - m_-1)| / w_1
    truncation: float
    rounding_f: float
    rounding_g: float
    largest: float  # L


@dataclass
class Verdict:
    """What one check saw: its lines, and whether a design warns on them."""

    lines: list[Line] = field(default_factory=list)
    lines_unusable: int = 0
    points_skipped: int = 0
    n_log_prob: int = 0
    n_grad: int = 0
    dtype_f: str = "float64"
    dtype_g: str = "float64"
    floor: float = 0.0  # the floor in nats over u of the design that made this verdict

    @property
    def warned(self) -> bool:
        """Whether the design that made this verdict warns."""
        return self.warns(self.floor)

    def warns(self, floor: float = 0.0, relative: float = 0.0) -> bool:
        """Whether a line's gap exceeds its tolerance, ``h floor`` and ``relative norm``."""
        return any(
            line.gap > max(line.tolerance, line.step * floor, relative * line.norm)
            for line in self.lines
        )

    def ratio(self, floor: float = 0.0) -> float:
        """The largest gap over its bar; ``inf`` for a gap above a bar of 0."""
        largest = 0.0
        for line in self.lines:
            bar = max(line.tolerance, line.step * floor)
            if bar > 0:
                largest = max(largest, line.gap / bar)
            elif line.gap > 0:
                largest = np.inf
        return largest


class _Recorder:
    # Records the narrowest float dtype that a user function returned since reset.
    def __init__(self, function: Any) -> None:
        self.function, self.itemsize = function, 8

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.function(*args, **kwargs)
        dtype = np.asarray(out).dtype
        if dtype.kind == "f":
            self.itemsize = min(self.itemsize, dtype.itemsize)
        return out

    def reset(self) -> None:
        self.itemsize = 8


def _guarded(method: Any, x: np.ndarray) -> Any:
    try:
        return method(x)
    except Exception as error:
        raise GradientCheckError(
            "the gradient check, which compares grad with differences of log_prob at draws "
            f"of the initial distribution when the sampler is built, raised: {error}; pass "
            "check_grad=False to skip it"
        ) from error


def _signs(generator: np.random.Generator, d: int) -> np.ndarray:
    return generator.choice([-1.0, 1.0], size=d)


def _streams(initial: Any, root: Any) -> tuple[int, np.ndarray | None, Any, Any]:
    # First pass over the check's stream: the range of the 3 P draws in each coordinate
    # (None when no line needs the fallback, a_p = b_p in no coordinate), and two
    # generators placed at the fallback signs and at the probe signs.
    generator = replicate_generator(root, CHECK_NAMESPACE, 0)
    first = draw(initial, generator)
    d = len(first)
    low, high = first, first.copy()
    needed = False
    for k in range(1, 3 * POINTS):
        point = draw(initial, generator, d)
        if k % 3 == 1:
            a = point
        elif k % 3 == 2:
            needed = needed or bool((a == point).any())
        np.minimum(low, point, out=low)
        np.maximum(high, point, out=high)
        del point
    del a
    high -= low
    del low
    fallback = generator
    probe = copy.deepcopy(generator)
    for _ in range(POINTS):
        _signs(probe, d)
    return d, high if needed else None, fallback, probe


def _direction(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, spread: np.ndarray | None, signs: np.ndarray
) -> np.ndarray:
    u = a - b
    zero = u == 0
    if spread is not None and zero.any():
        fallback = np.where(spread > 0, spread, np.where(x != 0, np.abs(x), 1.0))
        u = np.where(zero, signs * fallback, u)
    smallest = np.sqrt(EPS64) * np.abs(x)
    return np.where(np.abs(u) < smallest, np.copysign(smallest, u), u)


def _sweep(target, lp, gr, initial, root, itemsize_f, itemsize_g):
    # One pass over the P lines, with the dtypes assumed narrowest; returns the lines'
    # numbers and the narrowest dtypes seen at the points, which the caller compares.
    d, spread, fallback_signs, probe_signs = _streams(initial, root)
    generator = replicate_generator(root, CHECK_NAMESPACE, 0)
    rows, unusable, skipped = [], 0, 0
    seen_f = seen_g = 8
    eps_f = h = eps_g = None
    for p in range(POINTS):
        x = draw(initial, generator, None if p == 0 else d)
        a = draw(initial, generator, d)
        b = draw(initial, generator, d)
        u = _direction(x, a, b, spread, _signs(fallback_signs, d))
        del a, b
        probe_sign = _signs(probe_signs, d)

        lp.reset(), gr.reset()
        zeroed = target.n_grad_zeroed
        f0, g = _guarded(target.log_prob_and_grad, x[None])
        seen_f = min(seen_f, lp.itemsize)
        if p == 0:
            itemsize_f = min(itemsize_f, lp.itemsize)
        dtype_f = np.dtype(f"f{itemsize_f}")
        if itemsize_f < 8:  # evaluate only at points that log_prob's dtype represents
            x = x.astype(dtype_f).astype(np.float64)
            zeroed = target.n_grad_zeroed
            f0, g = _guarded(target.log_prob_and_grad, x[None])
        f0, g, zeroed = f0[0], g[0], target.n_grad_zeroed > zeroed
        seen_g = min(seen_g, gr.itemsize)
        if p == 0:
            itemsize_g = min(itemsize_g, gr.itemsize)
            eps_f = np.finfo(dtype_f).eps
            eps_g = np.finfo(np.dtype(f"f{itemsize_g}")).eps
            h = eps_f ** (1 / 3)

        probe = 0.0
        if itemsize_g < 8 and np.isfinite(f0) and not zeroed:  # grad's conditioning
            moved = _guarded(target.log_prob_and_grad, (x * (1 + 4 * eps_g * probe_sign))[None])[
                1
            ][0]
            probe = np.abs(moved - g)
            del moved
        del probe_sign
        if f0 == -np.inf or zeroed:
            skipped += 1
            continue

        u *= h  # v, the step vector
        stencil = np.empty((4, d))
        for row, t in zip(stencil, T_STENCIL, strict=True):
            np.multiply(t, u, out=row)
            row += x
        if itemsize_f < 8:
            stencil[...] = stencil.astype(dtype_f)
        values = _guarded(target.log_prob, stencil)
        stencil -= x  # the displacements as rounded; then d1 in row 0 and d2 in row 2
        stencil[0] -= stencil[1]
        stencil[2] -= stencil[3]
        d1, d2 = stencil[0], stencil[2]
        vv = u @ u
        w1, w2 = d1 @ u / vv, d2 @ u / vv
        del u
        if not (np.isfinite(values).all() and w1 > 0 and w2 > w1):
            unusable += 1
            continue
        D, D2 = (values[0] - values[1]) / w1, (values[2] - values[3]) / w2
        G, G2 = g @ d1 / w1, g @ d2 / w2
        d2 = np.abs(d1, out=stencil[2])
        weight = np.abs(g, out=stencil[1])
        weight *= ULPS * eps_g
        np.maximum(weight, probe, out=weight)
        rounding_g = (weight @ d2) / w1
        d1 *= g
        norm = float(np.sqrt(np.sum(np.multiply(d1, d1, out=d1)))) / w1
        del stencil, d1, d2, weight
        rows.append(
            {
                "point": p,
                "D": D,
                "G": G,
                "gap": abs(G - D),
                "T": abs((D - G) - (D2 - G2)),
                "F4": values[2] + values[3] - 4 * (values[0] + values[1]) + 6 * f0,
                "L": max(np.abs(values).max(), abs(f0)),
                "w1": w1,
                "Rg": rounding_g,
                "h": h,
                "norm": norm,
            }
        )
    return rows, unusable, skipped, (itemsize_f, itemsize_g), (seen_f, seen_g), eps_f


def check(
    log_prob: Any,
    grad: Any,
    initial: Any,
    rng: Any,
    *,
    floor: float = 0.0,
    vectorize: bool = False,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> Verdict:
    """Compare ``grad`` with central differences of ``log_prob`` along ``POINTS`` lines.

    Parameters
    ----------
    log_prob, grad : callable
        The user's functions, as ``CoupledSampler`` takes them.
    initial : distribution
        The initial distribution, as ``as_initial`` takes it.
    rng : int, SeedSequence or Generator
        The sampler's ``rng`` or its root seed; the check reads only namespace 3 of it.
    floor : float
        0 for (N); ``PHI`` for (H).
    vectorize : bool
    args, kwargs
        As ``CoupledSampler`` takes them.

    Returns
    -------
    Verdict
        Its lines, and ``warned`` for the design with this ``floor``.

    Raises
    ------
    GradientCheckError
        If ``log_prob`` or ``grad`` raise inside the check.
    """
    lp, gr = _Recorder(log_prob), _Recorder(grad)
    target = Target(lp, gr, args=args, kwargs=kwargs or {}, vectorize=vectorize)
    initial = as_initial(initial)
    root = root_seed(rng)
    assumed = (8, 8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*returned float")
        while True:
            # A function whose dtype narrows after the first point restarts the sweep
            # with the narrowest dtype; for any other function this runs once.
            rows, unusable, skipped, used, seen, eps_f = _sweep(
                target, lp, gr, initial, root, *assumed
            )
            if min(seen[0], used[0]) == used[0] and min(seen[1], used[1]) == used[1]:
                break
            assumed = (min(seen[0], used[0]), min(seen[1], used[1]))
    verdict = Verdict(
        lines_unusable=unusable,
        points_skipped=skipped,
        n_log_prob=target.n_log_prob,
        n_grad=target.n_grad,
        dtype_f=f"float{8 * used[0]}",
        dtype_g=f"float{8 * used[1]}",
        floor=floor,
    )
    if rows:
        relative = np.array([r["F4"] / r["L"] if r["L"] > 0 else 0.0 for r in rows])
        r_hat = float(np.sqrt(np.mean(relative**2) / 70))
        s_hat = float(np.sqrt(np.mean([r["F4"] ** 2 for r in rows]) / 70))
        for r in rows:
            rounding_f = (2 / r["w1"]) * max(
                r_hat * r["L"], s_hat, abs(r["F4"]) / np.sqrt(70), ULPS * eps_f * r["L"]
            )
            verdict.lines.append(
                Line(
                    point=r["point"],
                    gap=float(r["gap"]),
                    tolerance=float(SAFETY * (r["T"] + rounding_f + r["Rg"])),
                    step=float(r["h"]),
                    norm=r["norm"],
                    truncation=float(r["T"]),
                    rounding_f=float(rounding_f),
                    rounding_g=float(r["Rg"]),
                    largest=float(r["L"]),
                )
            )
    return verdict


def noise_limited(log_prob: Any, grad: Any, initial: Any, rng: Any, **kwargs: Any) -> Verdict:
    """Design (N): warns where ``grad`` and the differences disagree beyond their error."""
    return check(log_prob, grad, initial, rng, floor=0.0, **kwargs)


def harm_floor(log_prob: Any, grad: Any, initial: Any, rng: Any, **kwargs: Any) -> Verdict:
    """Design (H): (N), and only where the disagreement exceeds ``PHI`` nats over ``u``."""
    return check(log_prob, grad, initial, rng, floor=PHI, **kwargs)


DESIGNS = {"N": 0.0, "H": PHI}  # the floor of each design, for Verdict.warns
