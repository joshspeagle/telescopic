"""Row-independent linear algebra for kernels (spec R-KER-6)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from telescopic._random import FloatArray


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
