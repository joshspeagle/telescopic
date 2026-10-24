"""Exact Ising reference values by enumeration on an n x n periodic lattice (scratch).
pi_theta(x) ∝ exp(theta * t(x)), t(x) = sum over neighbour pairs x_i x_j (each pair once),
matching Jacob et al. (2020) Sec. 5.3 notation."""
import itertools, numpy as np
def exact(n, thetas):
    N = n * n
    idx = np.arange(2**N, dtype=np.uint32)
    spins = ((idx[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.int8) * 2 - 1
    g = spins.reshape(-1, n, n).astype(np.int32)
    t = (g * np.roll(g, 1, axis=1)).sum((1, 2)) + (g * np.roll(g, 1, axis=2)).sum((1, 2))
    absm = np.abs(g.sum((1, 2))) / N
    out = []
    for th in thetas:
        lw = th * t; w = np.exp(lw - lw.max()); w /= w.sum()
        out.append((th, float(w @ t), float(w @ t**2 - (w @ t) ** 2), float(w @ absm)))
    return out
beta_c = 0.5 * np.log(1 + np.sqrt(2))
print(f"Onsager critical inverse temperature (infinite lattice): {beta_c:.6f}")
for n in [3, 4]:
    for th, Et, Vt, Em in exact(n, [0.1, 0.3, 0.4, beta_c, 0.5]):
        print(f"n={n} theta={th:.4f}: E[t]={Et:.4f} Var[t]={Vt:.4f} E|m|={Em:.4f}")
