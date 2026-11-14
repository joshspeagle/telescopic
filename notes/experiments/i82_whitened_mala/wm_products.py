"""How many matrix-vector products does a step of each form make?

Question. Issue #82 counted 7 products per `MALA.step` and 12 per `MALA.coupled_step`,
plus those of `couplings.reflection_maximal`, before the step costs of issue #83 changed
the reflection. This counts them at this commit, and those of `WhitenedMALA`.

Design. `telescopic._linalg.matvec` is replaced, in the namespaces of
`telescopic.kernels.mala`, `telescopic.couplings` and `wm_kernel`, by a wrapper that
records the caller's function and the shapes; each form then makes one step and one
coupled step of 64 chains (pairs) at d = 50 on a vectorized Gaussian target whose
gradient is one dense product (not counted: it is the target's own). The products with
C, C^T and C^{-1} are the kernels' only (d x d) products.

Result (`wm_products.out`, at commit 9932df7): `MALA.step` makes 7 products: 4 in
`_mean` (2 for the mean at x, 2 for the mean at x' called from `_log_alpha`), 1 for L z,
and 2 in `_log_alpha` (the forward and the backward residuals). `MALA.coupled_step`
makes 14: 8 in `_mean`, 4 in `_log_alpha` and 2 in `reflection_maximal` (the whitened
difference of the means and C z). `WhitenedMALA.step` makes 3 (2 whitened drifts and
L (h/2 + z)), and its coupled step 7 (4 whitened drifts, the whitened difference of the
positions and the two proposals). Every product is of 64 rows with a 50 x 50 matrix.

Run from the repository root:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:<this folder> \
    python <this folder>/wm_products.py
"""

import collections
import sys

import numpy as np
import wm_kernel
from wm_timing import dense_target

import telescopic.couplings
from telescopic import examples, kernels
from telescopic._linalg import matvec
from telescopic.kernels import mala
from telescopic.testing import random_draws

calls = collections.Counter()


def counted(matrix, vectors, *, blas):
    caller = sys._getframe(1).f_code.co_name
    calls[(caller, matrix.shape, vectors.shape)] += 1
    return matvec(matrix, vectors, blas=blas)


for module in (mala, telescopic.couplings, wm_kernel):
    module.matvec = counted

d, n = 50, 64
target = dense_target(d)
cov = examples.gaussian(d).cov
rng = np.random.default_rng(0)
chol = np.linalg.cholesky(cov)
for kernel in (
    kernels.MALA(1.65 * d ** (-1 / 6), cov=cov),
    wm_kernel.WhitenedMALA(1.65 * d ** (-1 / 6), cov=cov),
):
    x = kernel.init_state(target, rng.standard_normal((n, d)) @ chol.T)
    y = kernel.init_state(target, rng.standard_normal((n, d)) @ chol.T)
    for name, step, states in (
        ("step", kernel.step, (x,)),
        ("coupled_step", kernel.coupled_step, (x, y)),
    ):
        calls.clear()
        step(target, *states, random_draws(n, 1))
        total = sum(calls.values())
        print(f"{type(kernel).__name__}.{name}: {total} products")
        for (caller, matrix, vectors), count in sorted(calls.items()):
            print(f"    {count} x in {caller}: {matrix} times rows of {vectors}")
