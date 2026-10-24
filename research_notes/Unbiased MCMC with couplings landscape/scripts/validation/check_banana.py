"""Exact moments of the Heng & Jacob (2019) banana, U(x) = (1-x1)^2 + 10 (x2 - x1^2)^2 (scratch).
exp(-U) factorises: x1 ~ N(1, 1/2), x2 | x1 ~ N(x1^2, 1/20).
E[x1]=1, Var[x1]=1/2, E[x2]=E[x1^2]=1.5, Var[x2]=1/20 + Var(x1^2) = 0.05 + (2*0.25 + 4*1*0.5) = 2.55."""
import numpy as np
rng = np.random.default_rng(0); n = 4_000_000
x1 = rng.normal(1, np.sqrt(0.5), n); x2 = rng.normal(x1**2, np.sqrt(1 / 20))
print(f"E[x1]={x1.mean():.4f} Var[x1]={x1.var():.4f} E[x2]={x2.mean():.4f} Var[x2]={x2.var():.4f} (theory 1, 0.5, 1.5, 2.55)")
