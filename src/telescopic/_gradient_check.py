"""The gradient check: ``grad`` against central differences of ``log_prob`` along lines."""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.random import Generator, SeedSequence

from telescopic._random import GRADIENT_CHECK, replicate_generator
from telescopic._types import Distribution, FloatArray
from telescopic.diagnostics import GradientCheck
from telescopic.initial import draw
from telescopic.target import Target
from telescopic.warnings import InputWarning, warn

# The noise-limited design (N) of notes/experiments/x123: its registration fixed these
# constants, and its result and review are the evidence for them (spec section 4.4).
POINTS = 8  # lines, one per draw x
ULPS = 64.0  # units in the last place below which rounding is not resolved
SAFETY = 10.0  # a line disagrees above SAFETY times its estimated error
EPS64 = np.finfo(np.float64).eps
STENCIL = np.array([1.0, -1.0, 2.0, -2.0])
SIGNS = np.array([-1.0, 1.0])


class _Recorder:
    # The user's function, recording the narrowest float dtype it returned since reset.
    def __init__(self, function: Callable[..., Any]) -> None:
        functools.update_wrapper(self, function, updated=())  # its name, for errors
        self.function, self.itemsize = function, 8

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.function(*args, **kwargs)
        dtype = np.asarray(out).dtype
        if dtype.kind in "fc":  # a complex output counts as its real part, as the Target's
            self.itemsize = min(self.itemsize, np.finfo(dtype).dtype.itemsize)
        return out

    def reset(self) -> None:
        self.itemsize = 8


def _signs(generator: Generator, d: int) -> FloatArray:
    signs: FloatArray = generator.choice(SIGNS, size=d)
    return signs


def _streams(
    initial: Distribution, seed: SeedSequence
) -> tuple[int, FloatArray, Generator, Generator]:
    # A first pass over the check's stream, with no evaluation: the range of the 3 POINTS
    # draws in each coordinate, and two generators placed at the signs of the fallback
    # directions and of the probe of a narrow gradient. One draw at a time keeps the
    # memory a few vectors.
    generator = replicate_generator(seed, GRADIENT_CHECK, 0)
    first = draw(initial, generator)
    d = len(first)
    low, high = first, first.copy()
    for _ in range(1, 3 * POINTS):
        point = draw(initial, generator, d)
        np.minimum(low, point, out=low)
        np.maximum(high, point, out=high)
    high -= low
    probe = copy.deepcopy(generator)
    for _ in range(POINTS):
        _signs(probe, d)
    return d, high, generator, probe


def _direction(
    x: FloatArray, a: FloatArray, b: FloatArray, spread: FloatArray, signs: FloatArray
) -> FloatArray:
    # u = a - b; where it is 0 (a point start, or a repeated empirical point), a random
    # sign times the range of the draws, |x| where that is 0, or 1; and never below
    # sqrt(eps) |x|, so that a step cannot round away.
    u = a - b
    zero = u == 0
    if zero.any():
        fallback = np.where(spread > 0, spread, np.where(x != 0, np.abs(x), 1.0))
        u = np.where(zero, signs * fallback, u)
    smallest = np.sqrt(EPS64) * np.abs(x)
    direction: FloatArray = np.where(np.abs(u) < smallest, np.copysign(smallest, u), u)
    return direction


def _sweep(
    target: Target,
    recorders: tuple[_Recorder, _Recorder],
    initial: Distribution,
    seed: SeedSequence,
    itemsizes: tuple[int, int],
) -> tuple[list[tuple[float, ...]], int, int, tuple[int, int], tuple[int, int], float]:
    # One pass over the lines with the dtypes of log_prob and grad assumed at most
    # itemsizes; returns for each compared line (draw from 1, G, gap, T, F4, L, w1, R_g),
    # then the unusable lines, the skipped draws, the dtypes used and seen, and h.
    lp, gr = recorders
    itemsize_f, itemsize_g = itemsizes
    d, spread, fallback_signs, probe_signs = _streams(initial, seed)
    generator = replicate_generator(seed, GRADIENT_CHECK, 0)
    rows: list[tuple[float, ...]] = []
    unusable = skipped = 0
    seen_f = seen_g = 8
    h = eps_g = 0.0
    for p in range(POINTS):
        x = draw(initial, generator, d)
        a = draw(initial, generator, d)
        b = draw(initial, generator, d)
        u = _direction(x, a, b, spread, _signs(fallback_signs, d))
        probe_sign = _signs(probe_signs, d)

        lp.reset()
        gr.reset()
        zeroed = target.n_grad_zeroed
        f0, g = target.log_prob_and_grad(x[None])
        seen_f = min(seen_f, lp.itemsize)
        if p == 0:
            itemsize_f = min(itemsize_f, lp.itemsize)
        dtype_f = np.dtype(f"f{itemsize_f}")
        if itemsize_f < 8:
            # Evaluate only at points that log_prob's dtype represents, so that the
            # function sees the points the comparison uses.
            x = x.astype(dtype_f).astype(np.float64)
            zeroed = target.n_grad_zeroed
            f0, g = target.log_prob_and_grad(x[None])
        f0, g, zeroed = f0[0], g[0], target.n_grad_zeroed > zeroed
        seen_g = min(seen_g, gr.itemsize)
        if p == 0:
            itemsize_g = min(itemsize_g, gr.itemsize)
            eps_g = float(np.finfo(np.dtype(f"f{itemsize_g}")).eps)
            # The cube root taken in log_prob's dtype, as design (N) registered it.
            h = float(np.finfo(dtype_f).eps ** (1 / 3))

        # A narrow grad's error along the line is set by the Hessian, which |g| does not
        # bound: its movement under a relative perturbation 8 times its cast's measures it.
        probe = 0.0
        if itemsize_g < 8 and np.isfinite(f0) and not zeroed:
            moved = target.log_prob_and_grad((x * (1 + 4 * eps_g * probe_sign))[None])
            probe = np.abs(moved[1][0] - g)
        if f0 == -np.inf or zeroed:
            skipped += 1
            continue

        u *= h  # the step vector v
        stencil = np.empty((4, d))
        for row, t in zip(stencil, STENCIL, strict=True):
            np.multiply(t, u, out=row)
            row += x
        if itemsize_f < 8:
            stencil[...] = stencil.astype(dtype_f)
        values = target.log_prob(stencil)
        # The displacements as rounded, m_t = fl(x + t v) - x; then m_1 - m_-1 in row 0
        # and m_2 - m_-2 in row 2, and their widths along v, about 2 and 4.
        stencil -= x
        stencil[0] -= stencil[1]
        stencil[2] -= stencil[3]
        d1, d2 = stencil[0], stencil[2]
        vv = u @ u
        w1, w2 = d1 @ u / vv, d2 @ u / vv
        if not (np.isfinite(values).all() and w1 > 0 and w2 > w1):
            unusable += 1
            continue
        D, D2 = (values[0] - values[1]) / w1, (values[2] - values[3]) / w2
        G, G2 = g @ d1 / w1, g @ d2 / w2
        weight = np.abs(g, out=stencil[1])
        weight *= ULPS * eps_g
        np.maximum(weight, probe, out=weight)
        rows.append(
            (
                p + 1,
                G,
                abs(G - D),
                abs((D - G) - (D2 - G2)),
                values[2] + values[3] - 4 * (values[0] + values[1]) + 6 * f0,
                max(np.abs(values).max(), abs(f0)),
                w1,
                (weight @ np.abs(d1, out=stencil[2])) / w1,
            )
        )
    return rows, unusable, skipped, (itemsize_f, itemsize_g), (seen_f, seen_g), h


# Spec section 4.4.
def check_gradient(
    log_prob: Callable[..., Any],
    grad: Callable[..., Any],
    initial: Distribution,
    seed: SeedSequence,
    *,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None,
    vectorize: bool,
) -> GradientCheck:
    # Design (N) on the user's functions, from namespace GRADIENT_CHECK of seed, with a
    # Target of its own so that the sampler's counters are untouched. That Target counts
    # as having warned of a narrow dtype, which the sampler's own Target warns of in the
    # run, so the check filters no warning. Warns when a line disagrees, and when no line
    # could be compared.
    recorders = _Recorder(log_prob), _Recorder(grad)
    target = Target(*recorders, args=args, kwargs=kwargs, vectorize=vectorize)
    target._warned_precision = True
    assumed = (8, 8)
    try:
        while True:
            # A function whose dtype narrows after the first draw restarts the sweep with
            # the narrowest dtype; for any other function this runs once.
            rows, unusable, skipped, used, seen, h = _sweep(
                target, recorders, initial, seed, assumed
            )
            narrowest = (min(seen[0], used[0]), min(seen[1], used[1]))
            if narrowest == used:
                break
            assumed = narrowest
    except Exception as error:
        error.add_note(
            "raised in the gradient check, which evaluates log_prob and grad at draws of the "
            "initial distribution when the sampler is built"
        )
        raise
    number, predicted, gap, truncation, f4, largest, w1, rounding_g = np.reshape(rows, (-1, 8)).T
    tolerance = np.empty(0)
    if rows:
        # The rounding of log_prob from the fourth differences F4, whose rounding part has
        # variance 70 sigma^2: pooled in proportion to |log_prob| and in absolute size, the
        # line's own (a kink), and a floor of ULPS units in the last place.
        eps_f = np.finfo(np.dtype(f"f{used[0]}")).eps
        relative = np.divide(f4, largest, out=np.zeros(len(rows)), where=largest > 0)
        r_hat = np.sqrt(np.mean(relative**2) / 70)
        s_hat = np.sqrt(np.mean(f4**2) / 70)
        rounding_f = (2 / w1) * np.maximum(
            np.maximum(r_hat * largest, s_hat),
            np.maximum(np.abs(f4) / np.sqrt(70), ULPS * eps_f * largest),
        )
        tolerance = SAFETY * (truncation + rounding_f + rounding_g)
    record = GradientCheck(
        draw=number.astype(np.int64),
        gap=gap / h,
        tolerance=tolerance / h,
        predicted=predicted / h,
        n_skipped=skipped,
        n_unusable=unusable,
    )
    message = _message(record)
    if message:
        warn(message, InputWarning)
    return record


def _message(check: GradientCheck) -> str | None:
    if not len(check.gap):
        return (
            f"the gradient check compared no line, so grad is unchecked: of its {POINTS} "
            f"draws of the initial distribution, {check.n_skipped} had zero density or a "
            f"gradient that is not finite, and the lines from the other {check.n_unusable} "
            "reached zero density, as from a start on a boundary of the support; draw the "
            "initial distribution where the density is positive and grad is finite"
        )
    disagree = check.gap > check.tolerance
    if not disagree.any():
        return None
    ratio = check.ratio
    line = int(np.argmax(np.where(disagree, ratio, 0.0)))
    gap, predicted = check.gap[line], check.predicted[line]
    relative = (
        f"{gap / abs(predicted):.2g} times the predicted change"
        if predicted
        else "where grad predicts no change"
    )
    return (
        "grad disagrees with differences of log_prob: on a line through draw "
        f"{check.draw[line]} of {POINTS} of the initial distribution, the change of "
        "log_prob across the spread of the initial distribution differs from the one grad "
        f"predicts by {gap:.3g} nats, {relative}, and {ratio[line]:.3g} times the "
        "comparison's estimated error, so grad is not the derivative of log_prob as "
        "computed; look for a wrong sign, a wrong factor or a missing term in grad; a "
        "gradient that is approximate by design, as from a solver, numerical differences "
        "or single precision, is expected to warn, and check_grad=False turns the check off"
    )
