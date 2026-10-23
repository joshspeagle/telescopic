"""Follow-up: heavy tails of H_{k:m} on AR(1) depend on pi_0; coverage vs R (scratch)."""
import numpy as np
from scipy import stats
rho = 0.9; s = np.sqrt(1 - rho**2); rng = np.random.default_rng(11)

def coupled_step(x, y):
    eps = rng.normal(size=x.size); xn = rho * x + s * eps
    z = rho * (x - y) / s
    met = np.log(rng.uniform(size=x.size)) <= np.minimum(0.0, -0.5 * (eps + z) ** 2 + 0.5 * eps**2)
    return xn, np.where(met, xn, rho * y - s * eps)

def estimators_chunk(k, m, R, init, T=800):
    X = np.zeros((R, T + 1)); Y = np.zeros((R, T + 1))
    X[:, 0], Y[:, 0] = init(R), init(R)
    X[:, 1] = rho * X[:, 0] + s * rng.normal(size=R)
    tau = np.full(R, -1); t = 1
    while (tau < 0).any() or t < m:
        assert t < T, "increase T"
        xn, yn = coupled_step(X[:, t], Y[:, t - 1])
        xn = np.where(tau > 0, rho * X[:, t] + s * rng.normal(size=R), xn)
        yn = np.where(tau > 0, xn, yn)
        X[:, t + 1], Y[:, t] = xn, yn
        new = (tau < 0) & (xn == yn); tau[new] = t + 1; t += 1
    lv = np.arange(1, T + 1)
    w = np.minimum(1, (lv - k) / (m - k + 1))
    mask = (lv[None] >= k + 1) & (lv[None] < tau[:, None])
    return X[:, k:m + 1].mean(1) + (mask * w[None] * (X[:, 1:] - Y[:, :-1])).sum(1), tau

def estimators(k, m, R, init, chunk=10_000):
    out = [estimators_chunk(k, m, min(chunk, R - i), init) for i in range(0, R, chunk)]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

inits = {"stationary N(0,1)": lambda R: rng.normal(size=R),
         "point mass x0=10": lambda R: np.full(R, 10.0),
         "N(10, 1) indep": lambda R: 10 + rng.normal(size=R)}
for name, init in inits.items():
    _, tau = estimators(1, 1, 5000, init)
    k = int(np.quantile(tau, 0.9)); m = 10 * k
    H, _ = estimators(k, m, 200_000, init)
    print(f"{name:18s}: E[tau]={tau.mean():.2f} k=q90={k} m={m} mean={H.mean():+.4f} SE={H.std(ddof=1)/np.sqrt(H.size):.4f} "
          f"var={H.var(ddof=1):.4f} kurtosis={stats.kurtosis(H):.1f} max|H|={np.abs(H).max():.1f}", flush=True)
    for R in [50, 500]:
        B = 400
        Hb = H[: B * R].reshape(B, R)
        se = Hb.std(1, ddof=1) / np.sqrt(R)
        cov = np.mean(np.abs(Hb.mean(1)) <= stats.t.ppf(0.975, R - 1) * se)
        print(f"     coverage of 95% t-interval for E[x]=0, R={R}, B={B}: {cov:.3f}", flush=True)
