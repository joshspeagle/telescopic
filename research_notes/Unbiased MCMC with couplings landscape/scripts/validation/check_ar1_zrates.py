"""False-alarm rates of |z| thresholds for unbiasedness tests at realistic R (scratch).
AR(1) rho=0.9, both chains from point mass x0=10, reflection-maximal coupling, k=q90=15, m=150."""
import numpy as np, sys
from scipy import stats
sys.argv = ["x"]
exec(open("scripts/check_ar1_tails.py").read().split("inits = {")[0])  # reuse functions
rng = np.random.default_rng(5)
init = lambda R: np.full(R, 10.0)
k, m = 15, 150
for R, B in [(5_000, 200), (20_000, 100)]:
    H, _ = estimators(k, m, R * B, init, chunk=20_000)
    Hb = H.reshape(B, R)
    z = Hb.mean(1) / (Hb.std(1, ddof=1) / np.sqrt(R))
    print(f"R={R:6d} B={B}: frac|z|>1.96={np.mean(np.abs(z) > 1.96):.3f} frac|z|>3={np.mean(np.abs(z) > 3):.3f} "
          f"frac|z|>4={np.mean(np.abs(z) > 4):.3f} max|z|={np.abs(z).max():.2f} skew(z)={stats.skew(z):+.2f}", flush=True)
