"""For each row r, count distinct bit patterns of row r's result across batch sizes n > r."""
import numpy as np

rng = np.random.default_rng(3)
sizes = (1, 2, 3, 4, 5, 7, 8, 16, 33, 64, 100, 256, 1000)
for d in (2, 5, 10, 30):
    A = rng.normal(size=(d, d))
    chol = np.linalg.cholesky(A @ A.T / d + np.eye(d))
    Z = rng.normal(size=(max(sizes), d))
    for name, op in {"z @ chol.T": lambda z: z @ chol.T,
                     "einsum('ij,kj->ik', z, chol)": lambda z: np.einsum("ij,kj->ik", z, chol)}.items():
        worst = 0
        for r in range(4):
            pats = {op(Z[:n])[r].tobytes() for n in sizes if n > r}
            worst = max(worst, len(pats))
        print(f"d={d:3d} {name:30s}: max distinct results of one row across batch sizes = {worst}")
