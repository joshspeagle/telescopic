"""Do identical rows at different batch positions give bitwise-identical log-densities?"""
import numpy as np
rng = np.random.default_rng(0)
for d in [3, 7, 10, 33, 100, 257]:
    A = rng.standard_normal((d, d)); P = A @ A.T + d * np.eye(d)  # precision matrix
    mism_gemm = mism_einsum = mism_scalar = 0; trials = 0
    for B in [2, 5, 17, 64, 255, 1000]:
        X = rng.standard_normal((B, d))
        for j in range(1, B, max(1, B // 8)):
            X[j] = X[0]
            trials += 1
        # batched quadratic form via BLAS gemm
        q_gemm = np.einsum("ij,ij->i", X @ P, X)
        q_ein = np.einsum("ij,jk,ik->i", X, P, X, optimize=False)
        q_one = np.array([x @ P @ x for x in X])  # one-at-a-time (matvec)
        for j in range(1, B, max(1, B // 8)):
            mism_gemm += q_gemm[j] != q_gemm[0]
            mism_einsum += q_ein[j] != q_ein[0]
            mism_scalar += q_one[j] != q_gemm[j]
    print(f"d={d:4d}: identical rows giving different batched values: gemm {mism_gemm}/{trials}, "
          f"einsum {mism_einsum}/{trials}; batched-vs-single-call mismatches {mism_scalar}/{trials}")
