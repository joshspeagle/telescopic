"""Row-independent linear algebra for kernels, and validated covariances."""

# Spec R-KER-6.

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular

from telescopic._types import FloatArray, real_array


def matvec(matrix: FloatArray, vectors: FloatArray) -> FloatArray:
    # matrix @ v for each row v of vectors. Unlike a BLAS product, einsum without
    # optimization gives a row the same bits whatever the other rows of the batch.
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
    matrix = _symmetric(name, cov, ndim)
    try:
        chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} must be positive definite") from None
    return matrix, chol


def square_root(cov: FloatArray) -> FloatArray:
    # A matrix A with A A^T = (C C^T)^(1/2), the covariance of the normal that NumPy's
    # Generator.multivariate_normal draws for a square matrix C with its default method,
    # mean + U sqrt(S) z from the singular value decomposition C = U S V^T; for a
    # symmetric C it is |C| = V |Lambda| V^T, C itself when C is positive semi-definite.
    # A is the Cholesky factor, which changes smoothly with C, or else V sqrt(|Lambda|).
    if not np.array_equal(cov, cov.T):
        u, s, _ = np.linalg.svd(cov)
        cov = (u * s) @ u.T
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(cov)
        root: FloatArray = vectors * np.sqrt(np.abs(values))
        return root


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
