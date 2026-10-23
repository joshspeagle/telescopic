"""Exact bias of estimator index bugs and replicates needed to detect them (scratch).

Two configurations on K=6 states (same target as check_finite_state.py):
  'cycle'  : +-1 random-walk proposal on a cycle (slow meeting)
  'indep'  : independence proposal uniform on all K states (fast meeting)
Both chains start at state 0 (point mass far from the mode), lag 1.
Exact E[H_{k:m}] under weight bugs computed by the off-diagonal recursion:
  E[BC] = sum_l w_l * E[(h(X_l) - h(Y_{l-1})) 1(tau > l)].
Variance of H estimated by simulation of the correct estimator.
"""
import numpy as np

K = 6
w = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0]); pi = w / w.sum()
h = np.arange(K, dtype=float); pi_h = float(pi @ h)
acc = lambda x, xp: min(1.0, pi[xp] / pi[x])

def build(kind):
    def qrow(x):
        q = np.zeros(K)
        if kind == "cycle":
            q[(x - 1) % K] += .5; q[(x + 1) % K] += .5
        else:
            q[:] = 1.0 / K
        return q
    P = np.zeros((K, K))
    for x in range(K):
        q = qrow(x)
        for xp in range(K):
            if xp != x: P[x, xp] += q[xp] * acc(x, xp)
        P[x, x] = 1 - P[x].sum()
    PP = np.zeros((K * K, K * K))
    for x in range(K):
        for y in range(K):
            qx, qy = qrow(x), qrow(y); mn = np.minimum(qx, qy); a = mn.sum()
            J = np.diag(mn) + ((1 - a) * np.outer((qx - mn) / (1 - a), (qy - mn) / (1 - a)) if a < 1 else 0)
            for xp in range(K):
                for yp in range(K):
                    if J[xp, yp] == 0: continue
                    ax, ay = acc(x, xp), acc(y, yp)
                    for nx, ny, pr in [(xp, yp, min(ax, ay)), (xp, y, max(0, ax - ay)), (x, yp, max(0, ay - ax)), (x, y, 1 - max(ax, ay))]:
                        if pr > 0: PP[x * K + y, nx * K + ny] += J[xp, yp] * pr
    return P, PP, qrow

def exact_H(P, PP, k, m, weight):
    pi0 = np.zeros(K); pi0[0] = 1
    # MCMC part
    mc = np.mean([pi0 @ np.linalg.matrix_power(P, t) @ h for t in range(k, m + 1)])
    diag = [x * K + x for x in range(K)]
    off = np.ones(K * K, bool); off[diag] = False
    dh = (h[:, None] - h[None, :]).ravel()  # h(x) - h(y) for pair (x, y)
    b = np.kron(pi0 @ P, pi0)  # (X_1, Y_0)
    b[~off] = 0.0               # restrict to tau > 1
    bc, l = 0.0, 1
    while b.sum() > 1e-15 and l < 100000:
        if l >= k + 1:
            bc += weight(l, k, m) * float(b @ dh)
        b = b @ PP; b[~off] = 0.0; l += 1
    return mc + bc

W = {
    "correct": lambda l, k, m: min(1.0, (l - k) / (m - k + 1)),
    "weight_off_by_one": lambda l, k, m: min(1.0, (l - k + 1) / (m - k + 1)),
    "no_cap": lambda l, k, m: (l - k) / (m - k + 1),
    "drop_correction": lambda l, k, m: 0.0,
}
rng = np.random.default_rng(3)

def sim_sd(P, PP, k, m, R=20000):
    # simulate correct H_{k:m} via the exact pair chain (vectorised over replicates)
    cumP = np.cumsum(P, 1); cumPP = np.cumsum(PP, 1)
    x0 = np.zeros(R, int); y0 = np.zeros(R, int)
    x1 = (rng.uniform(size=R)[:, None] > cumP[x0]).sum(1)
    T = 4000
    X = np.zeros((R, T + 1), int); Y = np.zeros((R, T + 1), int)
    X[:, 0] = x0; X[:, 1] = x1; Y[:, 0] = y0
    tau = np.full(R, -1)
    for t in range(1, T):
        s = X[:, t] * K + Y[:, t - 1]
        ns = (rng.uniform(size=R)[:, None] > cumPP[s]).sum(1)
        X[:, t + 1], Y[:, t] = divmod(ns, K)
        newly = (tau < 0) & (X[:, t + 1] == Y[:, t]); tau[newly] = t + 1
        if (tau > 0).all() and t + 1 >= m: break
    assert (tau > 0).all()
    H = X[:, k:m + 1].mean(1).astype(float)
    for r in range(R):
        for l in range(k + 1, tau[r]):
            H[r] += min(1.0, (l - k) / (m - k + 1)) * (h[X[r, l]] - h[Y[r, l - 1]])
    return H.mean(), H.std(ddof=1), tau.mean()

for kind in ["cycle", "indep"]:
    P, PP, _ = build(kind)
    for (k, m) in [(0, 0), (2, 10), (5, 50)]:
        Hm, sd, mt = sim_sd(P, PP, k, m)
        line = [f"{kind:5s} k={k} m={m}: E[tau]~{mt:.2f} sd(H)~{sd:.3f} sim mean={Hm:.4f} pi(h)={pi_h:.4f}"]
        for name, wf in W.items():
            eh = exact_H(P, PP, k, m, wf); bias = eh - pi_h
            Rneed = (5 * sd / abs(bias)) ** 2 if abs(bias) > 1e-12 else float("inf")
            line.append(f"  {name:18s} exact E[H]={eh:.5f} bias={bias:+.5f} R for 5-sigma={Rneed:,.0f}")
        print("\n".join(line))
