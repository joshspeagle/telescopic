"""Bias of execution/stopping policies for i.i.d. random-length unbiased estimators.

A pool of i.i.d. (H, C) pairs is produced by real coupled RWMH chains (coupled_rwmh.py).
Policies are then simulated exactly on B 'lanes' (= processors / lockstep slots with refill):
each lane runs an i.i.d. sequence of replicates back to back; time is in lockstep iterations.
Truth = pool mean of H (bootstrap world), so biases are exact up to Monte Carlo error.
"""
import sys
import time
import numpy as np
from coupled_rwmh import run_pairs

SIG, MU0, K, M_ = 1.0, 3.0, 2, 10
t0 = time.perf_counter()
H, tau, C = run_pairs(2_000_000, d=1, sigma=SIG, mu0=MU0, k=K, m=M_, rng=2026)
mu = H.mean()
print(f"pool: n={H.size} truth(pool mean)={mu:+.5f} sd(H)={H.std():.3f} E[C]={C.mean():.2f} "
      f"P(C>m)={np.mean(C > M_):.3f} max C={C.max()} corr(H,C)={np.corrcoef(H, C)[0,1]:+.3f} "
      f"[{time.perf_counter()-t0:.1f}s]")

rng = np.random.default_rng(99)


def simulate(B, T, n_macro, R_first=None, P1_frac=0.25, chunk=500):
    n_max = int(np.ceil(T / C.min())) + 3
    out = {k: [] for k in ["pooled_naive", "pooled_np1", "lane_drop_empty", "lane_zero_empty",
                           "lane_GH_unbiased", "GH91_preselect", "first_R_completed",
                           "index_fixed_R"]}
    makespan_GH, makespan_np1, n_used = [], [], []
    P1 = max(1, int(P1_frac * B))
    for start in range(0, n_macro, chunk):
        mm = min(chunk, n_macro - start)
        idx = rng.integers(0, H.size, size=(mm, B, n_max))
        Hs, Cs = H[idx], C[idx]
        cum = Cs.cumsum(axis=2)
        assert np.all(cum[:, :, -1] > T), "n_max too small"
        done = cum <= T
        N = done.sum(axis=2)
        sumH = (Hs * done).sum(axis=2)
        # pooled naive: all completed by T
        out["pooled_naive"].append(sumH.sum(1) / np.maximum(N.sum(1), 1))
        # pooled N+1: finish replicate in progress on every lane
        np1 = np.arange(n_max)[None, None, :] <= N[:, :, None]
        out["pooled_np1"].append((Hs * np1).sum((1, 2)) / (N + 1).sum(1))
        makespan_np1.append(cum[np.arange(mm)[:, None], np.arange(B)[None, :], N].max(1))
        avg = np.where(N > 0, sumH / np.maximum(N, 1), 0.0)
        has = N > 0
        out["lane_drop_empty"].append(np.where(has.any(1), (avg * has).sum(1) / np.maximum(has.sum(1), 1), np.nan))
        out["lane_zero_empty"].append(avg.mean(1))
        Htil = np.where(has, avg, Hs[:, :, 0])  # wait for first replicate on empty lanes
        out["lane_GH_unbiased"].append(Htil.mean(1))
        makespan_GH.append(np.maximum(T, Cs[:, :, 0].max(1) * 0 + np.where(has, T, Cs[:, :, 0]).max(1)))
        rem = np.where(Cs[:, :P1, 0] > T, Hs[:, :P1, 0], 0.0)
        out["GH91_preselect"].append(avg.mean(1) + rem.mean(1))
        if R_first is not None:
            flatc = cum.reshape(mm, -1)
            flath = Hs.reshape(mm, -1)
            S = np.partition(flatc, R_first - 1, axis=1)[:, R_first - 1]  # time of R-th completion
            use = flatc <= S[:, None]
            out["first_R_completed"].append((flath * use).sum(1) / use.sum(1))
            n_used.append(use.sum(1))
            out["index_fixed_R"].append(Hs.reshape(mm, -1)[:, :R_first].mean(1))
    res = {}
    for k, v in out.items():
        if not v:
            continue
        a = np.concatenate(v)
        a = a[np.isfinite(a)]
        res[k] = (a.mean() - mu, a.std() / np.sqrt(a.size))
    extra = dict(makespan_np1=np.concatenate(makespan_np1).mean(),
                 makespan_GH=np.concatenate(makespan_GH).mean())
    if n_used:
        extra["first_R_mean_used"] = np.concatenate(n_used).mean()
    return res, extra


for B, T, n_macro, R in [(64, 15, 40000, 256), (64, 30, 40000, 256), (64, 100, 20000, 256),
                         (1024, 15, 4000, 1024), (1024, 30, 4000, 4096)]:
    t1 = time.perf_counter()
    res, extra = simulate(B, T, n_macro, R_first=R)
    print(f"\nB={B} lanes, budget T={T} iters (={T/C.mean():.1f} x E[C]), macro-reps={n_macro}, R_first={R} "
          f"[{time.perf_counter()-t1:.1f}s]")
    for k, (b, se) in res.items():
        flag = "  <-- biased" if abs(b) > 4 * se else ""
        print(f"  {k:20s} bias={b:+.4f} (se {se:.4f}, z={b/se:+.1f}){flag}")
    print("  " + ", ".join(f"{k}={v:.1f}" for k, v in extra.items()))
