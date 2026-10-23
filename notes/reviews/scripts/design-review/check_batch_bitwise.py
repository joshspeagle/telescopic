"""Does a row's result of telescopic's own batched arithmetic depend on the batch size?

R-EXEC-2 promises bitwise-identical replicates across batch sizes when the log-density is
evaluated point by point. The kernels themselves still do batched linear algebra:
proposal = x + scale * z @ chol.T (RWMH, MALA, HMC), drift = grad @ Sigma (MALA),
dot products e . z (reflection). This checks whether row 0 of each op is bitwise identical
for batch sizes n = 1, 2, 7, 64, 256, 1000.
"""

import numpy as np

rng = np.random.default_rng(3)
sizes = (1, 2, 3, 7, 16, 64, 256, 1000)
for d in (2, 5, 10, 30, 100):
    A = rng.normal(size=(d, d))
    chol = np.linalg.cholesky(A @ A.T / d + np.eye(d))
    Z = rng.normal(size=(max(sizes), d))
    E = rng.normal(size=(max(sizes), d))
    ops = {
        "z @ chol.T": lambda z, e: z @ chol.T,
        "einsum('ij,kj->ik')": lambda z, e: np.einsum("ij,kj->ik", z, chol),
        "np.dot(z, chol.T)": lambda z, e: np.dot(z, chol.T),
        "sum(e*z, -1)": lambda z, e: np.sum(e * z, axis=-1),
        "einsum('ij,ij->i')": lambda z, e: np.einsum("ij,ij->i", e, z),
        "(z[:,None,:]*chol).sum(-1)": lambda z, e: (z[:, None, :] * chol[None]).sum(-1),
        "solve_triangular-ish (lstsq via solve)": lambda z, e: np.linalg.solve(chol, z.T).T,
    }
    for name, op in ops.items():
        ref = None
        mismatch_rows = 0
        checked = 0
        for n in sizes:
            out = op(Z[:n], E[:n])
            # compare every row present in this batch with the same row computed alone
            for r in range(min(n, 8)):
                alone = op(Z[r : r + 1], E[r : r + 1])[0]
                checked += 1
                if not np.array_equal(out[r], alone):
                    mismatch_rows += 1
        print(f"d={d:4d} {name:40s}: rows differing from the single-row result: {mismatch_rows}/{checked}")
