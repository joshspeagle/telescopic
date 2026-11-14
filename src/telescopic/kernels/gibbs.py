"""Gibbs samplers, coupled block by block through their full conditionals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from telescopic._random import Draws
from telescopic._types import BoolArray, FloatArray, read_only, real
from telescopic.couplings import maximal
from telescopic.kernels.base import Kernel, State, evaluate, evaluate_pair
from telescopic.target import Target

COUPLINGS = ("maximal", "crn", "two-scale")


# Spec R-KER-GIBBS.
class Gibbs(Kernel):
    r"""A Gibbs sampler that draws blocks of coordinates from their full conditionals.

    A step sweeps the blocks in order and draws each from its full conditional given the
    current values of the other coordinates. Coupled chains draw each
    block from a coupling of their two conditionals:

    - ``"maximal"``: the rejection maximal coupling, with ``eta``;
    - ``"crn"``: common random numbers, which contract the chains but make them equal only
      by rounding. For a block of one coordinate whose two conditionals have ``ppf``, one
      uniform goes through each ``ppf``. For a block of several coordinates, or of one
      without ``ppf``, each chain draws from its own conditional with ``rvs``, from
      generators in the same state, so each keeps its conditional's law whatever its
      type. The two draws are then close when ``rvs`` reads a fixed count of numbers from
      the generator and maps them smoothly in the conditional's parameters, as a normal
      block drawn as ``mean + cholesky(cov) @ random_state.standard_normal(k)``, or with
      ``random_state.multivariate_normal(mean, cov, method="cholesky")``, does. NumPy's
      default method, a singular value decomposition, which SciPy's
      ``multivariate_normal`` calls, can change signs and order between nearby
      covariances, and then the two draws are as far apart as independent ones;
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
        with ``rvs(random_state=...)`` and a normalized ``logpdf`` or ``logpmf``, such as
        ``scipy.stats.norm(mean, sd)``. One value of ``logpdf`` per coordinate, as with
        array parameters, is summed: the joint log-density of independent coordinates.
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
    coupling and for identical conditionals. For common random numbers it reads 1 uniform
    per row and block through ``ppf``, or else draws from a new child of the row's
    generator, :meth:`~telescopic.kernels.Draws.child`, which leaves the row's own stream
    where it was.
    A conditional that draws a point that is not finite, or not of its block's size,
    raises ``ValueError``. With ``log_prob=None`` the target has no log-density, and states
    carry 0.

    Examples
    --------
    The pump-failure model has full conditionals, so a Gibbs kernel samples it without a
    log-density, which is never evaluated:

    >>> from telescopic import examples, kernels, sample
    >>> example = examples.pumps()
    >>> kernel = kernels.Gibbs(example.conditional, example.blocks)
    >>> result = sample(
    ...     None, example.initial, kernel=kernel, n_replicates=50, rng=0
    ... )
    >>> result.n_log_prob, result.mean().value.shape
    (0, (11,))
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
        if threshold is not None:
            threshold = real("threshold", threshold, 0, np.inf, closed=True)
        eta = real("eta", eta, 0, 1, closed=True)
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
        # -inf for "crn", so that rows whose distance underflows to 0 take them too.
        self._switch = (
            threshold if threshold is not None else -np.inf if coupling == "crn" else np.inf
        )

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
            points = read_only(x)
            drawn = [
                self.conditional(points[row], b).rvs(random_state=generator)
                for row, generator in enumerate(generators)
            ]
            x[:, block] = _block(drawn, b, points, len(block))
        return evaluate(target, x, state.grad is not None), np.ones(len(x))

    def coupled_step(
        self, target: Target, state_x: State, state_y: State, draws: Draws
    ) -> tuple[State, State]:
        x, y = state_x.x.copy(), state_y.x.copy()
        common = np.linalg.norm(x - y, axis=1) > self._switch
        for b, block in enumerate(self.blocks):
            points_x, points_y = read_only(x), read_only(y)
            new_x, new_y = self._couple(b, points_x, points_y, common, draws)
            x[:, block] = _block(new_x, b, points_x, len(block))
            y[:, block] = _block(new_y, b, points_y, len(block))
        identical = np.all(x == y, axis=1)
        return evaluate_pair(
            partial(evaluate, target, need_grad=state_x.grad is not None), x, y, identical
        )

    def _couple(
        self, b: int, x: FloatArray, y: FloatArray, common: BoolArray, draws: Draws
    ) -> tuple[list[Any], list[Any]]:
        # The coupled draws of block b for each pair of rows of x and y.
        outside = np.ones(x.shape[1], dtype=bool)
        outside[self.blocks[b]] = False
        same = np.all(x[:, outside] == y[:, outside], axis=1)
        new_x: list[Any] = [None] * len(x)
        new_y: list[Any] = [None] * len(x)
        # Identical conditionals: one draw for both chains (spec section 5.7).
        for row in same.nonzero()[0].tolist():
            new_x[row] = new_y[row] = self.conditional(x[row], b).rvs(
                random_state=draws.generator(row)
            )
        for rows, couple in (
            ((~same & common).nonzero()[0], partial(_common, size=len(self.blocks[b]))),
            ((~same & ~common).nonzero()[0], partial(_maximal, eta=self.eta)),
        ):
            if len(rows):
                drawn_x, drawn_y = couple(
                    [self.conditional(x[row], b) for row in rows],
                    [self.conditional(y[row], b) for row in rows],
                    draws.subset(rows),
                )
                for row, value_x, value_y in zip(rows.tolist(), drawn_x, drawn_y, strict=True):
                    new_x[row], new_y[row] = value_x, value_y
        return new_x, new_y

    def __repr__(self) -> str:
        return (
            f"Gibbs({getattr(self.conditional, '__name__', 'conditional')}, "
            f"{len(self.blocks)} blocks, coupling={self.coupling!r}, eta={self.eta}, "
            f"threshold={self.threshold})"
        )


def _block(drawn: list[Any], b: int, points: FloatArray, size: int) -> FloatArray:
    # The draws of block b, one per chain, if each is a finite point of the block's size:
    # one value would be broadcast to the whole block, and a NaN would never meet and
    # would spread to the other blocks through their conditionals.
    try:
        # Draws of one shape, such as scalars, in one conversion.
        block = np.array(drawn, dtype=np.float64).reshape(len(drawn), size)
        if np.isfinite(block).all():
            return block
    except ValueError:
        pass
    values = [np.ravel(value) for value in drawn]
    if all(len(value) == size for value in values):
        block = np.array(values, dtype=np.float64)
        if np.isfinite(block).all():
            return block
    row = next(
        row
        for row, value in enumerate(values)
        if len(value) != size or not np.isfinite(value).all()
    )
    raise ValueError(
        f"the conditional of block {b} given x = {points[row].tolist()} drew "
        f"{values[row].tolist()}, not a finite point of shape ({size},)"
    )


def _maximal(p: list[Any], q: list[Any], draws: Draws, eta: float) -> tuple[FloatArray, ...]:
    x, y, _, _ = maximal(p, q, draws, eta)
    return x, y


def _common(p: list[Any], q: list[Any], draws: Draws, size: int) -> tuple[list[Any], ...]:
    # Common random numbers, chosen row by row from the row's two conditionals, so that a
    # row's draw does not depend on the other rows (spec R-EXEC-2): a uniform through each
    # ppf for a block of one coordinate, or else each chain's own sampler, run from the
    # same state of a generator.
    drawn_x: list[Any] = [None] * len(p)
    drawn_y: list[Any] = [None] * len(p)
    quantile = np.array(
        [size == 1 and hasattr(a, "ppf") and hasattr(b, "ppf") for a, b in zip(p, q, strict=True)],
        dtype=bool,
    )
    rows = quantile.nonzero()[0]
    # A uniform of 0, drawn with probability 2^-53, would give ppf = -inf.
    u = np.maximum(draws.subset(rows).uniform(), np.finfo(np.float64).tiny)
    for row, value in zip(rows.tolist(), u, strict=True):
        # Each ppf gives a scalar, or an array for array parameters.
        drawn_x[row], drawn_y[row] = p[row].ppf(value), q[row].ppf(value)
    for row in (~quantile).nonzero()[0].tolist():
        # A child of the row's generator, set back to its first state before the second
        # chain draws.
        generator = draws.child(row)
        start = generator.bit_generator.state
        drawn_x[row] = p[row].rvs(random_state=generator)
        generator.bit_generator.state = start
        drawn_y[row] = q[row].rvs(random_state=generator)
    return drawn_x, drawn_y
