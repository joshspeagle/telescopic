"""Row-independent linear algebra for kernels (spec R-KER-6), and validated covariances."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular

from telescopic._types import FloatArray


def matvec(matrix: FloatArray, vectors: FloatArray) -> FloatArray:
    # matrix @ v for each row v of vectors. Unlike a BLAS product, einsum without
    # optimisation gives a row the same bits whatever the other rows of the batch.
    product: FloatArray = np.einsum("kj,ij->ik", matrix, vectors)
    return product


def inverse_lower(chol: FloatArray) -> FloatArray:
    # The inverse of a lower-triangular factor, computed once per kernel so that
    # whitening is a matvec.
    inverse: FloatArray = solve_triangular(chol, np.eye(len(chol)), lower=True)
    return inverse


def covariance(
    name: str, cov: npt.ArrayLike, ndim: int | None = None
) -> tuple[FloatArray, FloatArray]:
    # A validated covariance matrix, of size ndim if given, and its lower Cholesky
    # factor; errors name the argument.
    matrix = np.array(cov, dtype=np.float64, ndmin=2)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, not shape {matrix.shape}")
    if ndim is not None and len(matrix) != ndim:
        raise ValueError(f"{name} must have shape {(ndim, ndim)}, not {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    # Symmetric up to rounding: |C_ij - C_ji| <= 1e-8 sqrt(C_ii C_jj), a test that does
    # not depend on the scales of the coordinates; the symmetric part is then used.
    root = np.sqrt(np.abs(np.diag(matrix)))
    if not np.all(np.abs(matrix - matrix.T) <= 1e-8 * np.outer(root, root)):
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) / 2
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} must be positive definite") from None
    return matrix, chol
