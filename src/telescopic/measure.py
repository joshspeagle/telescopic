"""Signed measures: the weighted atoms of the replicates' estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from telescopic._random import FloatArray
from telescopic.estimators import IntArray


@dataclass(frozen=True)
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
        Cost: one call of ``h`` on all :math:`N` atoms.
        """
        values = np.asarray(h(self.atoms), dtype=np.float64)
        if values.ndim not in (1, 2) or len(values) != len(self.atoms):
            raise ValueError(
                f"h must map atoms of shape (n, d) to shape (n,) or (n, p); for n = "
                f"{len(self.atoms)} it returned shape {values.shape}"
            )
        columns = values.reshape(len(values), -1) * self.weights[:, None]
        sums = np.stack(
            [
                np.bincount(self.replicate, weights=column, minlength=self.n_replicates)
                for column in columns.T
            ],
            axis=1,
        )
        per_replicate: FloatArray = sums.reshape(self.n_replicates, *values.shape[1:])
        return per_replicate
