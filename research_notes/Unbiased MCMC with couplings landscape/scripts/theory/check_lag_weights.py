"""Numerical check of lag-L time-averaged unbiased MCMC weights on a 3-state chain.
Compares (a) definition-based average of H_l, (b) floor/ceil weights (Atchade & Jacob 2024 eq 2.10,
unbiasedmcmc H_bar.R), (c) old weights min(ceil((t-k)/L), m-k+1)/(m-k+1) (commented-out code)."""
import math, sys
import numpy as np

rng = np.random.default_rng(12345)
P = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
w, v = np.linalg.eig(P.T); pi = np.real(v[:, np.argmin(abs(w - 1))]); pi = pi / pi.sum()
pi0 = np.array([1.0, 0.0, 0.0])           # point mass start (allows early meetings)
h = np.array([0.0, 1.0, 5.0])             # test function values
target = float(pi @ h)

def step(x):
    return rng.choice(3, p=P[x])

def coupled_step(x, y):
    p, q = P[x], P[y]
    mn = np.minimum(p, q); c = mn.sum()
    if rng.random() < c:
        z = rng.choice(3, p=mn / c); return z, z
    rp = (p - mn) / (1 - c); rq = (q - mn) / (1 - c)
    return rng.choice(3, p=rp), rng.choice(3, p=rq)

def run(k, m, L):
    X = [rng.choice(3, p=pi0)]; Y = [rng.choice(3, p=pi0)]
    for t in range(1, L + 1):
        X.append(step(X[-1]))
    tau = L if X[L] == Y[0] else math.inf      # tau = inf{t >= L : X_t = Y_{t-L}}
    t = L
    while t < max(m, tau):
        t += 1
        if tau < math.inf:
            X.append(step(X[-1])); Y.append(X[-1 - L] if False else X[t - L])  # keep Y_{t-L} = X_t? no: faithful copy
            Y[-1] = X[t]  # after meeting Y_{t-L} = X_t
        else:
            x, y = coupled_step(X[-1], Y[-1]); X.append(x); Y.append(y)
            if x == y: tau = t
    hX = h[np.array(X)]; hY = h[np.array(Y)]
    # (a) definition: H_l = h(X_l) + sum_{j>=1, l+jL <= tau-1} (h(X_{l+jL}) - h(Y_{l+(j-1)L}))
    Hdef = 0.0
    for l in range(k, m + 1):
        s = hX[l]; j = 1
        while l + j * L <= tau - 1:
            s += hX[l + j * L] - hY[l + (j - 1) * L]; j += 1
        Hdef += s
    Hdef /= (m - k + 1)
    mcmc = hX[k:m + 1].mean()
    bc_new = 0.0; bc_old = 0.0
    if tau < math.inf:
        for tt in range(k + L, int(tau)):
            d = hX[tt] - hY[tt - L]
            bc_new += (math.floor((tt - k) / L) - math.ceil(max(L, tt - m) / L) + 1) * d
            bc_old += min(math.ceil((tt - k) / L), m - k + 1) * d
    Hnew = mcmc + bc_new / (m - k + 1); Hold = mcmc + bc_old / (m - k + 1)
    cost = L + 2 * (tau - L) + max(0, m - tau)
    return Hdef, Hnew, Hold, tau, cost

for (k, m, L) in [(0, 0, 2), (0, 3, 2), (1, 6, 3), (2, 4, 1), (0, 5, 4)]:
    R = 200000
    out = np.array([run(k, m, L) for _ in range(R)])
    maxdiff = np.max(np.abs(out[:, 0] - out[:, 1]))
    se = out[:, 1].std(ddof=1) / math.sqrt(R); se_old = out[:, 2].std(ddof=1) / math.sqrt(R)
    print(f"k={k} m={m} L={L}: target={target:.5f} | def-vs-new max|diff|={maxdiff:.2e} | "
          f"new mean={out[:,1].mean():.5f} (se {se:.5f}) | old mean={out[:,2].mean():.5f} (se {se_old:.5f}) | "
          f"E[tau]={out[:,3].mean():.3f} min tau={out[:,3].min():.0f} E[cost]={out[:,4].mean():.3f}")
