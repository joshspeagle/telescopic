"""Signed measures: the weighted atoms of the replicates' estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from telescopic._types import FloatArray, IntArray, integer, read_only


@dataclass(frozen=True, eq=False)
class SignedMeasure:
    r"""The signed measures :math:`\hat\pi_r` of :math:`R` replicates (spec R-EST-6, R-OUT-2).

    .. math:: \hat\pi_r(h) = \sum_{i : \mathrm{replicate}_i = r} \omega_i\,h(x_i)

    Each replicate's weights sum to 1. Some are negative, so an estimate can fall outside
    the range of :math:`h`.

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
        # An h that writes to its input would change every later estimate.
        values = np.asarray(h(read_only(self.atoms)), dtype=np.float64)
        if values.ndim not in (1, 2) or len(values) != len(self.atoms):
            raise ValueError(
                f"h must map atoms of shape (n, d) to shape (n,) or (n, p); for n = "
                f"{len(self.atoms)} it returned shape {values.shape}"
            )
        per_replicate: FloatArray = self._sums @ values
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

        Returns
        -------
        scipy.sparse.csr_array, shape (R, n_cells)
            At most one stored entry per atom, so its memory follows the atoms, not
            ``R * n_cells``.
        """
        n_cells = integer("n_cells", n_cells, minimum=1)
        if np.shape(cell) != self.weights.shape or not np.all((cell >= -1) & (cell < n_cells)):
            raise ValueError(
                f"cell must give each of the {len(self.weights)} atoms a cell from 0 to "
                f"n_cells - 1 = {n_cells - 1}, or -1"
            )
        # Built as CSR directly: each row holds at most one entry, already in order.
        inside = cell >= 0
        indptr = np.concatenate([[0], np.cumsum(inside)])
        indicator: csr_array = csr_array(
            (np.ones(indptr[-1]), cell[inside], indptr), shape=(len(cell), n_cells)
        )
        masses: csr_array = self._sums @ indicator
        return masses
