"""Signed measures: the weighted atoms of the replicates' estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

from telescopic._types import FloatArray, IntArray


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
        # A read-only view: an h that writes to its input would change every later estimate.
        atoms = self.atoms.view()
        atoms.flags.writeable = False
        values = np.asarray(h(atoms), dtype=np.float64)
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

    def mass(self, cell: IntArray, n_cells: int) -> FloatArray:
        """Each replicate's signed mass in each of ``n_cells`` cells, such as histogram bins.

        Parameters
        ----------
        cell : ndarray of int, shape (N,)
            The cell of each atom, from 0 to ``n_cells - 1``, or -1 for none.
        n_cells : int

        Returns
        -------
        ndarray, shape (R, n_cells)
        """
        inside = cell >= 0
        masses: FloatArray = np.bincount(
            self.replicate[inside] * n_cells + cell[inside],
            weights=self.weights[inside],
            minlength=self.n_replicates * n_cells,
        ).reshape(self.n_replicates, n_cells)
        return masses
