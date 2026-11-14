"""Signed measures: the weighted atoms of the replicates' estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from telescopic._types import FloatArray, IntArray, integer, integral, read_only, values_of

__all__ = ["SignedMeasure"]


# Spec R-EST-6 and R-OUT-2.
@dataclass(frozen=True, eq=False)
class SignedMeasure:
    r"""The signed measures :math:`\hat\pi_r` of :math:`R` replicates.

    .. math:: \hat\pi_r(h) = \sum_{i : \mathrm{replicate}_i = r} \omega_i\,h(x_i)

    Each replicate's weights sum to 1. Some are negative, so an estimate can fall outside
    the range of :math:`h`.

    The arrays are checked at construction and stored as read-only views, without a copy:
    the attributes cannot be written to, but the arrays passed in still can.

    Attributes
    ----------
    atoms : ndarray, shape (N, d)
    weights : ndarray, shape (N,)
    replicate : ndarray of int, shape (N,)
        The position, from 0 to R - 1, of the replicate each atom belongs to.
    n_replicates : int
    """

    atoms: FloatArray
    weights: FloatArray
    replicate: IntArray
    n_replicates: int

    def __post_init__(self) -> None:
        # Checked in one O(N) pass and stored as read-only views, without a copy: the sums
        # cached for per_replicate would go stale if the arrays changed through these
        # attributes (DECISIONS #107).
        n_replicates = integer("n_replicates", self.n_replicates, minimum=0)
        atoms, weights = np.asarray(self.atoms), np.asarray(self.weights)
        replicate = np.asarray(self.replicate)
        for name, array in (("atoms", atoms), ("weights", weights)):
            if array.dtype.kind not in "iuf":
                raise TypeError(
                    f"{name} must be real numbers, not an array of dtype {array.dtype}"
                )
        if atoms.ndim != 2:
            raise ValueError(f"atoms must have shape (N, d), not {atoms.shape}")
        if weights.shape != (len(atoms),):
            raise ValueError(f"weights must have shape ({len(atoms)},), not {weights.shape}")
        if (
            replicate.shape != weights.shape
            or replicate.dtype.kind not in "iuf"
            or not integral(replicate, 0)
            or np.any(replicate >= n_replicates)
        ):
            raise ValueError(
                f"replicate must give each of the {len(atoms)} atoms an integer from 0 to "
                f"n_replicates - 1 = {n_replicates - 1}"
            )
        index = replicate.astype(np.int64, copy=False)
        for name, stored in (("atoms", atoms), ("weights", weights), ("replicate", index)):
            object.__setattr__(self, name, read_only(stored))
        object.__setattr__(self, "n_replicates", n_replicates)

    def per_replicate(self, h: Callable[[FloatArray], npt.ArrayLike]) -> FloatArray:
        r"""The values :math:`\hat\pi_r(h)` of each replicate.

        Parameters
        ----------
        h : callable
            Maps atoms of shape ``(n, d)`` to values of shape ``(n,)`` or ``(n, p)``.

        Returns
        -------
        ndarray, shape (R,) or (R, p)

        Notes
        -----
        Cost: one call of ``h`` on all :math:`N` atoms, then one sparse product.
        """
        per_replicate: FloatArray = self._sums @ values_of(h, self.atoms)
        return per_replicate

    @cached_property
    def _sums(self) -> csr_array:
        # Row r holds the weights of replicate r's atoms, so a product sums each replicate.
        n = len(self.weights)
        return csr_array(
            (self.weights, (self.replicate, np.arange(n))), shape=(self.n_replicates, n)
        )

    def mass(self, cell: IntArray, n_cells: int) -> csr_array:
        """Each replicate's signed mass in each of ``n_cells`` cells, such as histogram bins.

        Parameters
        ----------
        cell : ndarray of int, shape (N,)
            The cell of each atom, from 0 to ``n_cells - 1``, or -1 for none.
        n_cells : int
            The number of cells.

        Returns
        -------
        scipy.sparse.csr_array, shape (R, n_cells)
            At most one stored entry per atom, so its memory follows the atoms, not
            ``R * n_cells``.
        """
        n_cells = integer("n_cells", n_cells, minimum=1)
        cell = np.asarray(cell)
        if cell.shape != self.weights.shape or not integral(cell, -1) or np.any(cell >= n_cells):
            raise ValueError(
                f"cell must give each of the {len(self.weights)} atoms a cell from 0 to "
                f"n_cells - 1 = {n_cells - 1}, or -1"
            )
        cell = cell.astype(np.int64, copy=False)
        # Built as CSR directly: each row holds at most one entry, already in order.
        inside = cell >= 0
        indptr = np.concatenate([[0], np.cumsum(inside)])
        indicator: csr_array = csr_array(
            (np.ones(indptr[-1]), cell[inside], indptr), shape=(len(cell), n_cells)
        )
        masses: csr_array = self._sums @ indicator
        return masses
