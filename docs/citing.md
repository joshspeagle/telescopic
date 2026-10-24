# Citing

If you use telescopic, please cite the software, whose citation is in the repository's
`CITATION.cff`, and the methods it implements:

- the estimators: Glynn & Rhee (2014) and Jacob, O'Leary & Atchadé (2020);
- the lag and the distance bounds: Biswas, Jacob & Vanetti (2019);
- the automatic choice of `k`, `m` and the lag: Atchadé & Jacob (2024);
- the automatic tuning: Haario, Saksman & Tamminen (2001) for the adaptive Metropolis
  chains, and Hoffman & Gelman (2014) for the dual averaging of step sizes;
- the reflection-maximal coupling of the random walk: Jacob, O'Leary & Atchadé (2020)
  and O'Leary (2021);
- coupled HMC: Heng & Jacob (2019);
- the coupled independence sampler: O'Leary, Wang & Jacob (2021);
- the two-scale Gibbs coupling: Biswas, Bhattacharya, Jacob & Johndrow (2022).

## References

The works cited in this documentation and in the docstrings of the API reference:

- Andrieu, C. & Thoms, J. (2008). A tutorial on adaptive MCMC. Stat. Comput. 18(4),
  343–373.
- Atchadé, Y. F. & Jacob, P. E. (2024). Unbiased Markov chain Monte Carlo: what, why,
  and how. arXiv:2406.06851.
- Beskos, A., Pillai, N., Roberts, G., Sanz-Serna, J.-M. & Stuart, A. (2013). Optimal
  tuning of the hybrid Monte Carlo algorithm. Bernoulli 19(5A), 1501–1534.
- Biswas, N., Bhattacharya, A., Jacob, P. E. & Johndrow, J. E. (2022). Coupling-based
  convergence assessment of some Gibbs samplers for high-dimensional Bayesian
  regression with shrinkage priors. J. R. Stat. Soc. B, doi:10.1111/rssb.12495.
- Biswas, N., Jacob, P. E. & Vanetti, P. (2019). Estimating convergence of Markov chains
  with L-lag couplings. NeurIPS.
- Craiu, R. V., Rosenthal, J. & Yang, C. (2009). Learn from thy neighbor: parallel-chain
  and regional adaptive MCMC. J. Am. Stat. Assoc. 104(488), 1454–1466.
- Douc, R., Jacob, P. E., Lee, A. & Vats, D. (2022). Solving the Poisson equation using
  coupled Markov chains. arXiv:2206.05691.
- Gerber, M. & Lee, A. (2020). Discussion of "Unbiased Markov chain Monte Carlo methods
  with couplings" by Jacob, O'Leary and Atchadé. J. R. Stat. Soc. B 82(3).
- Glynn, P. W. & Rhee, C.-H. (2014). Exact estimation for Markov chain equilibrium
  expectations. J. Appl. Probab. 51A, 377–389.
- Glynn, P. W. & Whitt, W. (1992). The asymptotic efficiency of simulation estimators.
  Oper. Res. 40(3), 505–520.
- Haario, H., Saksman, E. & Tamminen, J. (2001). An adaptive Metropolis algorithm.
  Bernoulli 7(2), 223–242.
- Heng, J. & Jacob, P. E. (2019). Unbiased Hamiltonian Monte Carlo with couplings.
  Biometrika 106(2), 287–302.
- Hill, B. M. (1975). A simple general approach to inference about the tail of a
  distribution. Ann. Statist. 3(5), 1163–1174.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path
  lengths in Hamiltonian Monte Carlo. J. Mach. Learn. Res. 15, 1593–1623.
- Jacob, P. E., O'Leary, J. & Atchadé, Y. F. (2020). Unbiased Markov chain Monte Carlo
  methods with couplings. J. R. Stat. Soc. B 82(3), 543–600.
- Middleton, L., Deligiannidis, G., Doucet, A. & Jacob, P. E. (2020). Unbiased Markov
  chain Monte Carlo for intractable target distributions. Electron. J. Stat. 14(2),
  2842–2891.
- Neal, R. M. (2003). Slice sampling. Ann. Statist. 31(3), 705–767.
- O'Leary, J. (2021). Couplings of the random-walk Metropolis algorithm.
  arXiv:2102.01790.
- O'Leary, J., Wang, G. & Jacob, P. E. (2021). Maximal couplings of the
  Metropolis-Hastings algorithm. AISTATS, PMLR 130.
- Roberts, G. O., Gelman, A. & Gilks, W. R. (1997). Weak convergence and optimal scaling
  of random walk Metropolis algorithms. Ann. Appl. Probab. 7(1), 110–120.
- Roberts, G. O. & Rosenthal, J. S. (1998). Optimal scaling of discrete approximations
  to Langevin diffusions. J. R. Stat. Soc. B 60(1), 255–268.
- Thorisson, H. (2000). Coupling, Stationarity, and Regeneration. Springer.
