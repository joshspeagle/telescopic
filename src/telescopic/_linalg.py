"""Row-independent linear algebra for kernels, and validated covariances."""

# Spec R-KER-6 and D-20.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular

from telescopic._types import FloatArray, real_array


def matvec(matrix: FloatArray, vectors: FloatArray, blas: bool) -> FloatArray:
    # matrix @ v for each row v of vectors, of shape (..., n, d); each (n, d) array of a
    # stack is a separate product. Unlike a BLAS product, einsum without optimization
    # gives a row the same bits whatever the other rows of the batch, but is several times
    # slower beyond a few dimensions, so kernels use BLAS only for vectorized targets
    # (DECISIONS #189).
    if not blas:
        product: FloatArray = np.einsum("kj,...ij->...ik", matrix, vectors)
        return product
    # Silent on overflow, as einsum is: kernels reject the non-finite results.
    with np.errstate(over="ignore", invalid="ignore"):
        product = vectors @ matrix.T
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
    matrix = _symmetric(name, cov, ndim)
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} must be positive definite") from None
    return matrix, chol


def _symmetric(name: str, cov: npt.ArrayLike, ndim: int | None) -> FloatArray:
    # The symmetric part of a finite square matrix, of size ndim if given, symmetric up
    # to rounding: |C_ij - C_ji| <= 1e-8 sqrt(C_ii C_jj), a test that does not depend on
    # the coordinates' scales.
    matrix = real_array(name, cov, "real numbers of shape (d, d)", ndmin=2)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, not shape {np.shape(cov)}")
    if ndim is not None and len(matrix) != ndim:
        raise ValueError(f"{name} must have shape {(ndim, ndim)}, not {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    root = np.sqrt(np.abs(np.diag(matrix)))
    if not np.all(np.abs(matrix - matrix.T) <= 1e-8 * np.outer(root, root)):
        raise ValueError(f"{name} must be symmetric")
    symmetric: FloatArray = (matrix + matrix.T) / 2
    return symmetric
