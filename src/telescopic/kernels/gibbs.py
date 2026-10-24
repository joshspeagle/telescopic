"""Gibbs samplers, coupled block by block through their full conditionals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws
from telescopic._types import FloatArray
from telescopic.couplings import maximal
from telescopic.kernels.base import Kernel, State, evaluate, evaluate_pair
from telescopic.target import Target

COUPLINGS = ("maximal", "crn", "two-scale")


class Gibbs(Kernel):
    r"""A Gibbs sampler that draws blocks of coordinates from their full conditionals.

    A step sweeps the blocks in order and draws each from its full conditional given the
    current values of the other coordinates (spec R-KER-GIBBS). Coupled chains draw each
    block from a coupling of their two conditionals:

    - ``"maximal"``: the rejection maximal coupling, with ``eta``;
    - ``"crn"``: common random numbers, which contract the chains but do not make them
      meet on their own: one uniform through ``ppf`` for a block of one coordinate, common
      standard normals for a multivariate normal conditional (an object with ``mean`` and
      ``cov``, such as ``scipy.stats.multivariate_normal``);
    - ``"two-scale"``: common random numbers for pairs farther apart than ``threshold``
      at the start of the sweep, the maximal coupling for the others (Biswas, Bhattacharya,
      Jacob & Johndrow 2022).

    Where the two chains agree on every coordinate outside a block, their conditionals are
    the same, and both chains get the same draw.

    Parameters
    ----------
    conditional : callable
        ``conditional(x, b)`` returns the full conditional of block ``b`` given the other
        coordinates of ``x``, a read-only point of shape ``(d,)``: a frozen distribution
        with ``rvs(random_state=...)`` and a normalised ``logpdf`` or ``logpmf``, such as
        ``scipy.stats.norm(mean, sd)``.
    blocks : sequence of array_like of int
        Index arrays that partition the coordinates ``0, ..., d - 1``.
    coupling : {"maximal", "crn", "two-scale"}
    eta : float
        Cap on the maximal coupling's meeting probability, in (0, 1]; 1 with ``"crn"``.
    threshold : float, optional
        The distance at which ``"two-scale"`` switches couplings; required by it, and
        refused by the others.

    Notes
    -----
    Sweeps loop over rows in Python, and build a conditional per row and block. A frozen
    SciPy distribution takes 250 to 300 µs to build, which can dominate a step; a small
    class with the methods used is faster, as in :mod:`telescopic.examples`. A step draws
    each block from the row's generator. A coupled step does the same for the maximal
    coupling and for identical conditionals, and reads 1 uniform (a block of one
    coordinate) or the block's size of normals (multivariate normal) per row and block for
    common random numbers. With ``log_prob=None`` the target has no log-density, and
    states carry 0.
    """

    def __init__(
        self,
        conditional: Callable[[FloatArray, int], Any],
        blocks: Sequence[npt.ArrayLike],
        *,
        coupling: Literal["maximal", "crn", "two-scale"] = "maximal",
        eta: float = 1.0,
        threshold: float | None = None,
    ) -> None:
        if not callable(conditional):
            raise TypeError(f"conditional must be callable, not {type(conditional).__name__}")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {', '.join(map(repr, COUPLINGS))}")
        if (coupling == "two-scale") != (threshold is not None):
            raise ValueError("threshold is required by coupling='two-scale', and only by it")
        if threshold is not None and not threshold > 0:
            raise ValueError(f"threshold must be positive, not {threshold}")
        if not 0 < eta <= 1:
            raise ValueError(f"eta must be in (0, 1], not {eta}")
        if coupling == "crn" and eta != 1:
            raise ValueError(
                "eta applies to the maximal coupling, which coupling='crn' never uses"
            )
        self.blocks = [np.atleast_1d(block) for block in blocks]
        indices = np.concatenate(self.blocks) if self.blocks else np.zeros(0)
        if (
            not all(block.ndim == 1 and len(block) for block in self.blocks)
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.array_equal(np.sort(indices), np.arange(len(indices)))
        ):
            raise ValueError(
                "blocks must be non-empty integer index arrays that partition 0, ..., d - 1"
            )
        self.conditional = conditional
        self.coupling, self.eta, self.threshold = coupling, eta, threshold
        # One rule for the three couplings: common random numbers beyond this distance.
        self._switch = threshold if threshold is not None else 0.0 if coupling == "crn" else np.inf

    @property
    def needs_log_prob(self) -> bool:
        return False

    @property
    def ndim(self) -> int | None:
        return sum(len(block) for block in self.blocks)

    def step(self, target: Target, state: State, draws: Draws) -> tuple[State, FloatArray]:
        x = state.x.copy()
        generators = [draws.generator(row) for row in range(len(x))]
        for b, block in enumerate(self.blocks):
            points = _read_only(x)
            for row, generator in enumerate(generators):
                x[row, block] = self.conditional(points[row], b).rvs(random_state=generator)
        return evaluate(target, x, state.grad is not None), np.ones(len(x))

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        x, y = state_x.x.copy(), state_y.x.copy()
        common = np.linalg.norm(x - y, axis=1) > self._switch
        for b, block in enumerate(self.blocks):
            points_x, points_y = _read_only(x), _read_only(y)
            outside = np.ones(x.shape[1], dtype=bool)
            outside[block] = False
            same = np.all(x[:, outside] == y[:, outside], axis=1)
            new_x = np.empty((len(x), len(block)))
            new_y = np.empty_like(new_x)
            # Identical conditionals: one draw for both chains (spec section 5.7).
            for row in same.nonzero()[0].tolist():
                drawn = self.conditional(points_x[row], b).rvs(random_state=draws.generator(row))
                new_x[row] = new_y[row] = drawn
            for rows, couple in (
                ((~same & common).nonzero()[0], partial(_common, size=len(block))),
                ((~same & ~common).nonzero()[0], partial(_maximal, eta=self.eta)),
            ):
                if len(rows):
                    new_x[rows], new_y[rows] = couple(
                        [self.conditional(points_x[row], b) for row in rows],
                        [self.conditional(points_y[row], b) for row in rows],
                        draws.subset(rows),
                    )
            x[:, block], y[:, block] = new_x, new_y
        identical = np.all(x == y, axis=1)
        return evaluate_pair(
            partial(evaluate, target, need_grad=state_x.grad is not None), x, y, identical
        )

    def __repr__(self) -> str:
        return (
            f"Gibbs({getattr(self.conditional, '__name__', 'conditional')}, "
            f"{len(self.blocks)} blocks, coupling={self.coupling!r}, eta={self.eta}, "
            f"threshold={self.threshold})"
        )


def _read_only(points: FloatArray) -> FloatArray:
    # A view that the user's conditionals cannot write through.
    view = points.view()
    view.flags.writeable = False
    return view


def _maximal(p: list[Any], q: list[Any], draws: Draws, eta: float) -> tuple[FloatArray, ...]:
    x, y, _, _ = maximal(p, q, draws, eta)
    return x.reshape(len(p), -1), y.reshape(len(p), -1)


def _common(p: list[Any], q: list[Any], draws: Draws, size: int) -> tuple[FloatArray, ...]:
    # Common random numbers: a uniform through each ppf for a block of one coordinate, or
    # normals through each multivariate normal's Cholesky factor.
    if size == 1 and all(hasattr(conditional, "ppf") for conditional in p + q):
        # A uniform of 0, drawn with probability 2^-53, would give ppf = -inf.
        u = np.maximum(draws.uniform(), np.finfo(np.float64).tiny)
        return tuple(
            np.array([c.ppf(value) for c, value in zip(side, u, strict=True)])[:, None]
            for side in (p, q)
        )
    if not all(
        hasattr(conditional, "mean") and hasattr(conditional, "cov") for conditional in p + q
    ):
        raise TypeError(
            "coupling='crn' needs conditionals with ppf for blocks of one coordinate, or "
            "multivariate normal ones with mean and cov"
        )
    z = draws.normal(len(np.atleast_1d(p[0].mean)))
    return tuple(
        np.array(
            [
                np.atleast_1d(c.mean) + np.linalg.cholesky(np.atleast_2d(c.cov)) @ row
                for c, row in zip(side, z, strict=True)
            ]
        )
        for side in (p, q)
    )
