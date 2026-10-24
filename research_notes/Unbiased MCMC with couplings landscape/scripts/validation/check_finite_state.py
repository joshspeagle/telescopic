"""Finite-state exact checks for coupled MH chains (scratch research script).

Target pi on K states, MH with a +-1 random walk proposal on a cycle.
Coupled kernel: maximal coupling of proposals (independent residuals, Jacob et al. 2020
Sec. 5.4 discrete construction) + common uniform for acceptance.

Exact (deterministic) quantities:
  * coupled transition matrix PP on K^2 pairs; marginal constraints checked exactly;
  * exact pmf of the L-lag meeting time tau^(L) = inf{t > L : X_t = Y_{t-L}};
  * exact TV(pi_0 P^t, pi) and exact L-lag bound E[max(0, ceil((tau - L - t)/L))].
Statistical checks against a simple reference simulator:
  * chi-square test of simulated meeting times against the exact pmf;
  * z-test of mean of H_{k:m} against pi(h), and power against index bugs.
"""
import numpy as np
from scipy import stats

K = 6
w = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0])
pi = w / w.sum()
h = np.arange(K, dtype=float)  # test function h(x) = x
pi_h = float(pi @ h)


def proposal_row(x):
    q = np.zeros(K)
    q[(x - 1) % K] += 0.5
    q[(x + 1) % K] += 0.5
    return q


def acc(x, xp):
    return min(1.0, pi[xp] / pi[x])  # symmetric proposal


# Exact MH kernel
P = np.zeros((K, K))
for x in range(K):
    q = proposal_row(x)
    for xp in range(K):
        if q[xp] > 0 and xp != x:
            P[x, xp] += q[xp] * acc(x, xp)
    P[x, x] = 1.0 - P[x].sum()
assert np.allclose(pi @ P, pi)


def max_coupling_joint(qx, qy):
    """Exact joint pmf of the maximal coupling with independent residuals."""
    m = np.minimum(qx, qy)
    alpha = m.sum()
    J = np.diag(m)
    if alpha < 1.0:
        rx = (qx - m) / (1 - alpha)
        ry = (qy - m) / (1 - alpha)
        J = J + (1 - alpha) * np.outer(rx, ry)
    return J


# Exact coupled kernel on pairs, index pair (x, y) -> x*K + y
PP = np.zeros((K * K, K * K))
for x in range(K):
    for y in range(K):
        J = max_coupling_joint(proposal_row(x), proposal_row(y))
        for xp in range(K):
            for yp in range(K):
                if J[xp, yp] == 0:
                    continue
                ax, ay = acc(x, xp), acc(y, yp)
                # common uniform U: both accept w.p. min(ax, ay), only the one with the
                # larger acceptance prob accepts w.p. |ax - ay|, both reject otherwise
                both, onlyx, onlyy = min(ax, ay), max(0.0, ax - ay), max(0.0, ay - ax)
                none = 1.0 - max(ax, ay)
                for (nx, ny, pr) in [(xp, yp, both), (xp, y, onlyx), (x, yp, onlyy), (x, y, none)]:
                    if pr > 0:
                        PP[x * K + y, nx * K + ny] += J[xp, yp] * pr
# Exact marginal and faithfulness constraints
PP4 = PP.reshape(K, K, K, K)
assert np.allclose(PP4.sum(axis=3), P[:, None, :]), "X marginal wrong"
assert np.allclose(PP4.sum(axis=2), P[None, :, :]), "Y marginal wrong"
diag = np.array([x * K + x for x in range(K)])
assert np.allclose(PP[np.ix_(diag, diag)].sum(axis=1), 1.0), "not faithful"
print("exact checks: marginals and faithfulness hold to machine precision")

pi0 = np.zeros(K)
pi0[0] = 1.0  # both chains start at state 0


def exact_meeting_pmf(L, tmax=4000):
    a = np.kron(pi0 @ np.linalg.matrix_power(P, L), pi0)  # (X_L, Y_0) independent
    off = np.ones(K * K, dtype=bool)
    off[diag] = False
    pmf = np.zeros(tmax + 1)  # pmf[s] = P(tau - L = s), s >= 1
    b = a.copy()
    for s in range(1, tmax + 1):
        b = b @ PP
        pmf[s] = b[diag].sum()
        b[~off] = 0.0
    assert 1 - pmf.sum() < 1e-12, "truncate later"
    return pmf


def exact_tv(t):
    return 0.5 * np.abs(pi0 @ np.linalg.matrix_power(P, t) - pi).sum()


def bound_from_pmf(pmf, L, t):
    s = np.arange(pmf.size)  # s = tau - L
    return float(np.sum(pmf * np.maximum(0, np.ceil((s - t) / L))))


for L in [1, 5, 20]:
    pmf = exact_meeting_pmf(L)
    Es = float(np.sum(np.arange(pmf.size) * pmf))
    viol = [t for t in range(0, 60) if bound_from_pmf(pmf, L, t) < exact_tv(t) - 1e-12]
    print(f"L={L:2d}: exact E[tau-L]={Es:.4f}; bound>=TV for t in [0,60): {not viol}; "
          + "; ".join(f"t={t}: TV={exact_tv(t):.4f} bound={bound_from_pmf(pmf, L, t):.4f}" for t in [0, 5, 10, 20]))


# ---------------- reference simulator ----------------
rng = np.random.default_rng(12345)


def step(x, u_prop, u_acc):
    xp = (x + (1 if u_prop < 0.5 else -1)) % K
    return xp if u_acc <= acc(x, xp) else x


def coupled_step(x, y):
    J = max_coupling_joint(proposal_row(x), proposal_row(y))
    idx = rng.choice(K * K, p=J.ravel())
    xp, yp = divmod(idx, K)
    u = rng.uniform()
    return (xp if u <= acc(x, xp) else x), (yp if u <= acc(y, yp) else y)


def run(L, k, m, bug=None):
    """Return tau and H_{k:m} for one coupled run with lag L (Jacob et al. eq. 2.1, lag L)."""
    X = [0]
    Y = [0]
    for _ in range(L):
        X.append(step(X[-1], rng.uniform(), rng.uniform()))
    t = L
    tau = None
    while tau is None or t < m:
        if tau is None:
            xn, yn = coupled_step(X[t], Y[t - L])
        else:
            xn = step(X[t], rng.uniform(), rng.uniform())
            yn = xn
        X.append(xn)
        Y.append(yn)
        t += 1
        if tau is None and X[t] == Y[t - L]:
            tau = t
    X = np.array(X, float)
    Y = np.array(Y, float)
    mcmc = X[k:m + 1].mean()
    bc = 0.0
    for l in range(k + L, tau):  # sum_{l=k+L}^{tau-1}
        j = (l - k) // L  # number of lag-L terms: lag-L generalisation of min(1, (l-k)/(m-k+1))
        wgt = min(1.0, j * L / (m - k + 1)) if False else min(1.0, np.ceil((l - k) / L) / ((m - k + 1) / L))
        if bug == "weight_off_by_one":
            wgt = min(1.0, (l - k + 1) / (m - k + 1))
        if bug == "wrong_lag_index":
            bc += wgt * (h[int(X[l])] - h[int(Y[l - L + 1])]) if l - L + 1 < len(Y) else 0.0
        else:
            bc += wgt * (h[int(X[l])] - h[int(Y[l - L])])
    return tau, mcmc + bc


# Lag-1 version of the weight: min(1, (l-k)/(m-k+1)); check the L=1 algebra directly.
def run_L1(k, m, bug=None):
    X = [0]
    Y = [0]
    X.append(step(X[0], rng.uniform(), rng.uniform()))
    t, tau = 1, None
    while tau is None or t < m:
        if tau is None:
            xn, yn = coupled_step(X[t], Y[t - 1])
        else:
            xn = step(X[t], rng.uniform(), rng.uniform())
            yn = xn
        X.append(xn)
        Y.append(yn)
        t += 1
        if tau is None and X[t] == Y[t - 1]:
            tau = t
    X = np.array(X, float)
    Y = np.array(Y, float)
    mcmc = X[k:m + 1].mean()
    lo = k + 1
    if bug == "sum_from_k":
        lo = k  # weight at l=k is 0 -> harmless; use instead a cap bug
    bc = 0.0
    for l in range(lo, tau):
        wgt = min(1.0, (l - k) / (m - k + 1))
        if bug == "weight_off_by_one":
            wgt = min(1.0, (l - k + 1) / (m - k + 1))
        if bug == "no_cap":
            wgt = (l - k) / (m - k + 1)
        if bug == "wrong_lag_index":
            bc += wgt * (h[int(X[l])] - h[int(Y[l])])
        else:
            bc += wgt * (h[int(X[l])] - h[int(Y[l - 1])])
    if bug == "drop_correction":
        bc = 0.0
    return tau, mcmc + bc


# Chi-square test of simulated lag-1 meeting times vs exact pmf
pmf1 = exact_meeting_pmf(1)
N = 20_000
taus = np.array([run_L1(0, 0)[0] for _ in range(N)]) - 1  # s = tau - 1
smax = int(np.searchsorted(np.cumsum(pmf1) * N, N - 5 * 1.0))  # pool tail so expected >= ~5
edges = list(range(1, smax + 1))
obs = np.array([np.sum(taus == s) for s in edges] + [np.sum(taus > smax)])
exp = np.array([pmf1[s] * N for s in edges] + [(1 - pmf1[: smax + 1].sum()) * N])
keep = exp > 0
chi2 = stats.chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum())
print(f"lag-1 meeting times: N={N}, mean(sim)={taus.mean():.4f} vs exact {np.sum(np.arange(pmf1.size) * pmf1):.4f}; "
      f"chi-square p={chi2.pvalue:.3f} ({keep.sum()} bins)")

# z-test of E[H_{k:m}] = pi(h), and power against bugs (k small so the correction matters)
k, m = 2, 10
for bug in [None, "weight_off_by_one", "no_cap", "wrong_lag_index", "drop_correction"]:
    R = 20_000
    H = np.array([run_L1(k, m, bug)[1] for _ in range(R)])
    z = (H.mean() - pi_h) / (H.std(ddof=1) / np.sqrt(R))
    print(f"H_(k={k}:m={m}) bug={bug!s:18s} mean={H.mean():.4f} pi(h)={pi_h:.4f} "
          f"sd={H.std(ddof=1):.3f} z={z:+.2f} (R={R})")
