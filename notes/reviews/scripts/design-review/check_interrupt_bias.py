"""Bias of the spec's interrupt rule (R-EXEC-6).

On the first Ctrl-C the engine stops starting replicates, finishes those in progress and
returns the started indices, a prefix. With B lanes and refill, the indices started by
interrupt time T are exactly 'all completed by T plus the one in progress on each lane',
so the returned average is GH91's pooled complete-in-progress estimator, i.e. a ratio
estimator. For B = 1 it is GH90's X-bar_{N(t)+1}, whose bias is of order 1/t.

i.i.d. (H, C) pairs come from the census toy (coupled RWMH, N(0,1) target, pi0 = N(3,1),
sigma 1, lag 1, k = 2, m = 10; C in lockstep iterations = max(tau, m)). Truth = pool mean.
"""
import time

import numpy as np

from coupled_rwmh import run_pairs

t0 = time.perf_counter()
H, tau, C = run_pairs(1_000_000, d=1, sigma=1.0, mu0=3.0, k=2, m=10, rng=2026)
mu = H.mean()
print(f"pool n={H.size}: mean H {mu:+.4f}, sd(H) {H.std():.2f}, E[C] {C.mean():.2f}, "
      f"corr(H,C) {np.corrcoef(H, C)[0, 1]:+.3f} [{time.perf_counter() - t0:.1f}s]")
rng = np.random.default_rng(5)
for B in (1, 2, 4, 16, 64):
    for T in (5, 15, 30, 100):
        n_macro = 400_000 if B <= 4 else 100_000
        n_max = int(np.ceil(T / C.min())) + 3
        est = []
        for start in range(0, n_macro, 20_000):
            mm = min(20_000, n_macro - start)
            idx = rng.integers(0, H.size, size=(mm, B, n_max))
            Hs, Cs = H[idx], C[idx]
            N = (Cs.cumsum(axis=2) <= T).sum(axis=2)  # completed by T on each lane
            started = np.arange(n_max)[None, None, :] <= N[:, :, None]  # + the one in progress
            est.append((Hs * started).sum((1, 2)) / (N + 1).sum(1))
        est = np.concatenate(est)
        b, se = est.mean() - mu, est.std() / np.sqrt(est.size)
        flag = "  <-- biased" if abs(b) > 5 * se else ""
        print(f"B={B:3d} lanes, interrupt at T={T:3d} iters ({T / C.mean():4.1f} E[C]): "
              f"bias {b:+.4f} (se {se:.4f}, z {b / se:+6.1f}), bias/sd(H) {b / H.std():+.4f}{flag}")
