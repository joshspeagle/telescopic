import numpy as np
rng = np.random.default_rng(3)
d = 10
A = rng.normal(size=(d, d)); chol = np.linalg.cholesky(A @ A.T / d + np.eye(d))
Z = rng.normal(size=(1000, d))
ref = (Z @ chol.T)[0].tobytes()
print("batch sizes where row 0 of z @ chol.T differs from its value at n=1000:",
      [n for n in range(1, 41) if (Z[:n] @ chol.T)[0].tobytes() != ref])
ref1 = (Z[:1] @ chol.T)[0]
print("max |difference| n=1 vs n=1000:", np.max(np.abs(ref1 - (Z @ chol.T)[0])))
