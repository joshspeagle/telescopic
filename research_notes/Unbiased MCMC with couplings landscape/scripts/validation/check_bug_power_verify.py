"""Verify exact bias of the weight_off_by_one bug (cycle, k=2, m=10) by large-R simulation."""
import numpy as np, sys
sys.path.insert(0, "scripts")
K = 6
w = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0]); pi = w / w.sum(); h = np.arange(K, dtype=float)
acc = lambda x, xp: min(1.0, pi[xp] / pi[x])
def qrow(x):
    q = np.zeros(K); q[(x - 1) % K] += .5; q[(x + 1) % K] += .5; return q
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
rng = np.random.default_rng(99)
cumP = np.cumsum(P, 1); cumPP = np.cumsum(PP, 1)
k, m, R, T = 2, 10, 400_000, 600
X = np.zeros((R, T + 1), np.int8); Y = np.zeros((R, T + 1), np.int8)
X[:, 1] = (rng.uniform(size=R)[:, None] > cumP[np.zeros(R, int)]).sum(1)
tau = np.full(R, -1)
for t in range(1, T):
    s = X[:, t].astype(int) * K + Y[:, t - 1]
    ns = (rng.uniform(size=R)[:, None] > cumPP[s]).sum(1)
    X[:, t + 1], Y[:, t] = divmod(ns, K)
    newly = (tau < 0) & (X[:, t + 1] == Y[:, t]); tau[newly] = t + 1
    if (tau > 0).all() and t + 1 >= m: break
print("unmet fraction:", np.mean(tau < 0))
ok = tau > 0
l = np.arange(T + 1)
Dl = h[X[:, 1:].astype(int)] - h[Y[:, :-1].astype(int)]          # Delta_l for l = 1..T
lv = l[1:]
mask = (lv[None, :] >= k + 1) & (lv[None, :] < tau[:, None])
mc = X[:, k:m + 1].mean(1)
for name, wf in [("correct", lambda l: np.minimum(1, (l - k) / (m - k + 1))),
                 ("weight_off_by_one", lambda l: np.minimum(1, (l - k + 1) / (m - k + 1)))]:
    H = mc + (mask * wf(lv)[None, :] * Dl).sum(1)
    H = H[ok]
    print(f"{name:18s} mean={H.mean():.4f} SE={H.std(ddof=1)/np.sqrt(H.size):.4f} pi(h)={pi @ h:.4f} sd={H.std(ddof=1):.3f}")
