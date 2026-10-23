# Unbiased MCMC via couplings and telescoping sums: estimators, guarantees, efficiency, budgets (theory notes for `telescopic`)

Provenance. I downloaded the primary texts below and read them as extracted text. Jacob, O'Leary & Atchadé, arXiv v5 (17 Jul 2019), the accepted version of the JRSSB 2020 paper. Atchadé & Jacob, arXiv v1 (10 Jun 2024). Douc, Jacob, Lee & Vats, arXiv v5 (9 Dec 2025). Craiu & Meng, arXiv v4 (14 Apr 2021). Middleton, Deligiannidis, Doucet & Jacob, arXiv v3 (15 Jun 2020). Vihola, arXiv v4 (11 May 2017). Biswas, Jacob & Vanetti, arXiv v3 (28 Oct 2019). Glynn & Rhee, arXiv v1. Glynn & Heidelberger 1990 and 1991 (author copies on Glynn's site). The discussion contributions of Vanetti & Doucet and of South, Nemeth & Oates (PDFs). I read the published JRSSB discussion and the authors' reply on the OUP HTML page through a summarising fetch tool; those items are marked "via OUP HTML". I also read the code of `pierrejacob/unbiasedmcmc` v0.3.1 at commit `660c698` (last commit 2026-09-23). Status tags: [PR] peer-reviewed, [PP] preprint, [DISC] published discussion contribution, [CODE] source code. [OWN] marks my own derivation or simulation and appears only under Inferences.

Common notation in these notes, used unless a source is quoted: π is the target, $P$ the π-invariant kernel, $\bar P$ the coupled kernel, $\pi_0$ the initial distribution, $L\ge 1$ the lag, $k$ the first retained index ("burn-in"), $m$ the last retained index, $\tau$ the meeting time and $\Delta_t := h(X_t)-h(Y_{t-L})$. Atchadé–Jacob and Douc et al. call the last index $\ell$, and Craiu–Meng call it $r$. Atchadé–Jacob and Douc et al. also use $m$, but for the moment index of $h\in L^m(\pi)$.

## 1. Exact definitions with verified indices: Glynn–Rhee, randomized truncation (McLeish; Rhee–Glynn), Jacob–O'Leary–Atchadé $H_k$, $H_{k:m}$, lag-$L$ weights, $\tau$/$\tau^{(L)}$, initialisation, faithfulness; cross-check against `unbiasedmcmc`

### Takeaway
The quantity to implement is

$$H^{(L)}_{k:m}=\frac{1}{m-k+1}\Big[\sum_{t=k}^{m}h(X_t)+\sum_{t=k+L}^{\tau-1}\Big(\Big\lfloor\tfrac{t-k}{L}\Big\rfloor-\Big\lceil\tfrac{\max(L,\,t-m)}{L}\Big\rceil+1\Big)\big(h(X_t)-h(Y_{t-L})\big)\Big].$$

It is built as follows. Draw $X_0,Y_0\sim\pi_0$ independently. Advance $X$ alone for $L$ steps. Then draw $(X_t,Y_{t-L})\sim\bar P$ until $X_t=Y_{t-L}$ and $t\ge m$. Enforce $X_t=Y_{t-L}$ for all $t\ge\tau$.

This formula matches three sources: the current R package, the 2024 Atchadé–Jacob review and Douc et al. It differs from the lag-$L$ weight printed in the 2020 JRSSB rejoinder, which is $\min\{1,\lceil (l-k)/L\rceil/(m-k+1)\}$. That weight is wrong for $L\ge 2$: it was biased in my simulation, and it survives in the R package only as commented-out code.

The randomized-truncation estimators of McLeish and Rhee–Glynn are the general device. Coupling-based estimators are the special case where exact meeting makes the truncation variable unnecessary ($N=\infty$).

### Cited Findings
**Randomized truncation (McLeish 2011; Rhee & Glynn 2015), as restated by Vihola (2018) [PR]**
- Setting and indices. $(Y_i)_{i\ge1}$ and $Y$ are integrable with $\mathbb E Y_i\to\mathbb E Y$, and $Y_0\equiv0$. The $(\Delta_i)_{i\ge1}$ satisfy $\mathbb E\Delta_i=\mathbb EY_i-\mathbb EY_{i-1}$. $R\in\mathbb N=\{1,2,\dots\}$ is independent of $(\Delta_i)$, with $P(R=i)=p_i$ and $\tilde p_i:=\sum_{j\ge i}p_j=P(R\ge i)$ — [Vihola 2018, arXiv:1512.01022](https://arxiv.org/abs/1512.01022)
- Single-term estimator (Vihola, Theorem 3). It is $Z^{(1)}:=\Delta_R/p_R$. Under Condition 2 ($\Delta_i$ independent, $p_i>0$, $\sum_i \mathbb E\Delta_i^2/p_i<\infty$) it satisfies $\mathbb EZ^{(1)}=\mathbb EY$ and $\mathbb E(Z^{(1)})^2=\sum_{i\ge1}\mathbb E\Delta_i^2/p_i$ — [Vihola 2018](https://arxiv.org/abs/1512.01022)
- Sum estimator (Vihola, Theorem 5): $Z^{(1)}_\Sigma:=\sum_{i=1}^{R}\Delta_i/\tilde p_i$. There are two cases.
  - Coupled sum: $\Delta_i=Y_i-Y_{i-1}$ and $\sum_i \mathbb E(Y_{i-1}-Y)^2/\tilde p_i<\infty$. Then $\mathbb E(Z_\Sigma^{(1)})^2=\sum_{i\ge1}[\mathbb E(Y_{i-1}-Y)^2-\mathbb E(Y_i-Y)^2]/\tilde p_i$.
  - Independent sum: $(\Delta_i)$ independent and $\sum_i[\mathrm{var}(\Delta_i)+(\mathbb EY-\mathbb EY_{i-1})^2]/\tilde p_i<\infty$.
  - Vihola states that McLeish (2011) first suggested these estimators, "but with different conditions and different expression for the variance" — [Vihola 2018](https://arxiv.org/abs/1512.01022)
- The Atchadé–Jacob review gives the same device with the truncation variable on $\{0,1,2,\dots\}$, citing Glynn (1983) and Rychlik (1990). Write $b_\infty=\sum_{k\ge0}a_k$ with $a_k=b_k-b_{k-1}$ and $b_{-1}=0$, and let $\xi$ have $p_k=P(\xi=k)>0$.
  - Single term: $G=a_\xi/p_\xi$, with $\mathbb E[G^2]=\sum_k a_k^2/p_k$.
  - Coupled sum: $H=a_0+\sum_{k=1}^{\xi}a_k/P(\xi\ge k)$, which "has finite variance under weaker conditions than that of G".
  - The expected cost is $\mathbb E\xi$. Cost is smaller if $(p_k)$ decays fast, while variance is smaller if it decays slowly — [Atchadé & Jacob 2024 [PP], arXiv:2406.06851](https://arxiv.org/abs/2406.06851)
- McLeish (2011) applied truncation directly to MCMC averages $b_k=k^{-1}\sum_{s=0}^{k-1}h(X_s)$. Because ergodic averages converge at the slow Monte Carlo rate, the resulting unbiased estimators "tend to have either a large cost or a large variance" — [Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851). Jacob et al. describe McLeish's estimators as "nearly unbiased" estimators from a single chain — [Jacob et al. arXiv:1708.03625](https://arxiv.org/abs/1708.03625)

**Glynn & Rhee (2014), J. Appl. Probab. 51A:377–389 [PR]**
- General estimator: $Z=\sum_{k=0}^{N}\Delta_k/P(N\ge k)$ with $\mathbb E\Delta_k=\mathbb E(Y_k-Y_{k-1})$ for $k\ge0$ and $Y_{-1}:=0$. $N$ is $\mathbb Z_+$-valued and independent of $(\Delta_k)$, and $\mathbb E\sum_{k\ge0}|\Delta_k|<\infty$ is required (eq. 2.2). Indexing starts at $k=0$, whereas Vihola's starts at $i=1$ — [Glynn & Rhee, arXiv:1409.4302](https://arxiv.org/abs/1409.4302)
- Coupling for contracting chains. The chain is $X_i=\varphi_i(X_{i-1})$ with i.i.d. random maps, so $X_n=(\varphi_n\circ\cdots\circ\varphi_1)(x)$. The coupled copy is $\tilde X_{n-1}=(\varphi_n\circ\cdots\circ\varphi_2)(x)$ for $n\ge2$, with $\tilde X_0=x$. Then $\Delta_k=f(X_k)-f(\tilde X_{k-1})$.
  - Contraction on average, $\mathbb E\rho^2(\varphi_1(y),\varphi_1(z))\le b\,\rho^2(y,z)$ with $b<1$, plus Lipschitz $f$, gives $\mathbb E\Delta_k^2\le\kappa^2 b^{k-1}\mathbb E\rho^2(X_1,x)$.
  - Variance is finite if $\sum_k b^k/P(N\ge k)<\infty$ (eq. 2.8).
  - The cost is $N$. Under a budget $c$, $c^{1/2}(\bar Z_{\Gamma(c)}-\mathbb Ef(X_\infty))\Rightarrow\sqrt{\mathbb EN\cdot\mathrm{var}Z}\,\mathcal N(0,1)$ (Theorem 1).
  - The optimal tail is $P(N\ge k)\propto(\mathbb E\Delta_k^2+2\sum_{j>k}\mathbb E\Delta_k\Delta_j)^{1/2}$ when non-increasing (eq. 2.10) — [Glynn & Rhee](https://arxiv.org/abs/1409.4302)
- Harris-recurrent chains. Coupling via regenerations gives a coupling time $\tau$, and $\Delta_k=(f(X_k)-f(X'_{k-1}))I(\tau>k)$.
  - Because $\Delta_k=0$ for $k>\tau$, simulation cost is at most $2\min(\tau,N)$, "(In fact, we may set N = ∞ a.s. in this setting, if we so wish.)"
  - Theorem 2: the estimator is unbiased for bounded $f$.
  - Theorem 3: square-root convergence if $\mathbb E(T(2)-T(1))^r<\infty$ for some $r>2$. Section 4 proves a Glivenko–Cantelli theorem for the induced signed measure — [Glynn & Rhee](https://arxiv.org/abs/1409.4302)
- Douc et al. show the Glynn–Rhee estimator is "identical to the estimator of Glynn & Rhee (2014), denoted by $H_0(X,Y)$ in Jacob et al. (2020)" when $X_1'\sim P(X_0',\cdot)$ and $(X_0',Y_0')$ is a coupling of $\pi_0$ with itself — [Douc et al. [PP], arXiv:2206.05691](https://arxiv.org/abs/2206.05691)

**Jacob, O'Leary & Atchadé (2020), JRSSB 82(3):543–600, DOI 10.1111/rssb.12336, with discussion [PR]**
- Coupling and initialisation.
  - $\bar P$ is a kernel on $\mathcal X\times\mathcal X$ with $\bar P((x,y),A\times\mathcal X)=P(x,A)$ and $\bar P((x,y),\mathcal X\times A)=P(y,A)$.
  - "We draw $(X_0,Y_0)$ such that $X_0\sim\pi_0$ and $Y_0\sim\pi_0$. Given $(X_0,Y_0)$ we draw $X_1\sim P(X_0,\cdot)$, and then for any $t\ge1$ … we draw $(X_{t+1},Y_t)\sim\bar P((X_t,Y_{t-1}),\cdot)$."
  - Any coupling of $\pi_0$ with itself is allowed, including $X_0=Y_0$, but "X1 ≠ Y0 in general. Thus one cannot force the meeting time to be small by setting X0 = Y0". "In our experiments we use independent draws" — [Jacob et al. arXiv v5](https://arxiv.org/abs/1708.03625)
- Assumptions.
  - Assumption 2.1: $\mathbb E[h(X_t)]\to\mathbb E_\pi[h(X)]$, and $\mathbb E|h(X_t)|^{2+\eta}\le D$ for all $t\ge0$.
  - Assumption 2.2: $\tau:=\inf\{t\ge1:X_t=Y_{t-1}\}$ satisfies $P(\tau>t)\le C\delta^t$ with $\delta\in(0,1)$.
  - Assumption 2.3: $X_t=Y_{t-1}$ for all $t\ge\tau$. Such coupled chains are "termed 'faithful' in Rosenthal [1997]" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Telescoping derivation. For any fixed $k\ge0$:

  $$\mathbb E_\pi h=\mathbb E h(X_k)+\sum_{t=k+1}^\infty(\mathbb E h(X_t)-\mathbb E h(Y_{t-1}))=\mathbb E\Big[h(X_k)+\sum_{t=k+1}^{\tau-1}(h(X_t)-h(Y_{t-1}))\Big].$$

  So $H_k(X,Y)=h(X_k)+\sum_{t=k+1}^{\tau-1}(h(X_t)-h(Y_{t-1}))$, where the sum is zero if $k+1>\tau-1$. It needs "$\tau-1$ calls to $\bar P$ and $\max(1,k+1-\tau)$ calls to $P$" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Time-averaged estimator (Algorithm 1). Loop: "Set t = 1. While t < max(m, τ) … draw $(X_{t+1},Y_t)\sim\bar P((X_t,Y_{t-1}),\cdot)$". Then $H_{k:m}=(m-k+1)^{-1}\sum_{\ell=k}^m H_\ell$, which equals eq. (2.1):

  $$H_{k:m}=\frac{1}{m-k+1}\sum_{\ell=k}^{m}h(X_\ell)+\sum_{\ell=k+1}^{\tau-1}\min\Big(1,\frac{\ell-k}{m-k+1}\Big)\big(h(X_\ell)-h(Y_{\ell-1})\big).$$

  Any weights $w_\ell$ with $\sum_{\ell=k}^m w_\ell=1$ also give an unbiased $\sum_\ell w_\ell H_\ell$ — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Interpretation and cost.
  - The first term "corresponds to a standard MCMC average with m total iterations and a burn-in period of k−1 iterations". The correction is zero if $\tau\le k+1$.
  - $H_{k:m}$ "requires τ − 1 calls to $\bar P$ and max(1, m + 1 − τ) calls to P", giving a cost of "2(τ − 1) + max(1, m + 1 − τ) iterations of the underlying MCMC algorithm" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Practical remarks.
  - The test function need not be specified at run time if the chains are stored.
  - Thinning can be done by replacing each call to $P$ and $\bar P$ by multiple calls.
  - Estimators "can take values outside of the range of the test function h; for instance they can take negative values" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Lag in the original paper. The arXiv v5 text mentions only "a time lag greater than one … with the meeting time defined as the first time t for which $\{X_t=Y_{t-\mathrm{lag}}\}$ occurs", "found to be fruitful in Biswas and Jacob [2019]" — [Jacob et al.](https://arxiv.org/abs/1708.03625)

**Lag-$L$ estimators: origin and exact weights**
- Vanetti & Doucet [DISC] proposed starting Y after L steps of X, with $X_t=Y_{t-L}$ for all $t\ge\tau$. Their identity for $N\ge k+L$ is

  $$\frac1L\sum_{n=N-L+1}^{N}\mathbb E h(X_n)=\frac1L\Big[\sum_{i=k}^{k+L-1}\mathbb E h(X_i)+\sum_{j=k+L}^{N}\{\mathbb E h(X_j)-\mathbb E h(Y_{j-L})\}\Big].$$

  They state it is "similar to the time-averaged estimator … when L = m − k, but the bias correction term incurred is not inflated by a large coefficient" — [Vanetti & Doucet discussion PDF](https://www.stats.ox.ac.uk/~doucet/discussion_VanettiDoucet_unbiasedMCMC.pdf)
- Biswas, Jacob & Vanetti (NeurIPS 2019) [PR].
  - Meeting time: $\tau^{(L)}:=\inf\{t>L:X_t=Y_{t-L}\}$. Faithfulness: $X_t=Y_{t-L}$ for all $t\ge\tau^{(L)}$.
  - Single-time estimator: $H^{(L)}_t=h(X_t)+\sum_{j=1}^{\lceil(\tau^{(L)}-L-t)/L\rceil}(h(X_{t+jL})-h(Y_{t+(j-1)L}))$.
  - Time-averaged estimator: $H^{(L)}_{k:m}=(m-k+1)^{-1}\sum_{t=k}^m H^{(L)}_t$ (their eqs. 8–10) — [Biswas et al. arXiv:1905.09971](https://arxiv.org/abs/1905.09971)
- Atchadé & Jacob (2024) [PP], Algorithm 1. Sample $(X_0,Y_0)\sim\bar\pi_0$. For $t=1,\dots,L$, sample $X_t\sim P(X_{t-1},\cdot)$. Then for $t\ge L$ sample $(X_{t+1},Y_{t-L+1})\sim\bar P((X_t,Y_{t-L}),\cdot)$ "until $X_{t+1}=Y_{t-L+1}$ and $t+1\ge\ell$". Here $\tau=\inf\{t\ge L:X_t=Y_{t-L}\}$ — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Atchadé & Jacob (2024), estimator.
  - $H_k:=h(X_k)+\sum_{j\ge1}(h(X_{k+jL})-h(Y_{k+(j-1)L}))$ (eq. 2.2).
  - $H_{k:\ell}=(\ell-k+1)^{-1}\sum_{t=k}^{\ell}H_t$, which equals (eq. 2.9) $(\ell-k+1)^{-1}\sum_{t=k}^\ell h(X_t)+\sum_{t=k+L}^{\tau-1}v_t(k,\ell,L)\{h(X_t)-h(Y_{t-L})\}$. The sum "is zero if $k+L\ge\tau$".
  - Weights (eq. 2.10): $v_t(k,\ell,L)=\big(\lfloor (t-k)/L\rfloor-\lceil \max(L,t-\ell)/L\rceil+1\big)/(\ell-k+1)$. These are derived from "the number of multiples of L within {a,…,b} equals ⌊b/L⌋ − ⌈a/L⌉ + 1".
  - Cost (eq. 2.11): $\mathrm{cost}(H_{k:\ell})=L+2(\tau-L)+\max(0,\ell-\tau)$. A $\bar P$ step counts as 2 units before meeting and 1 unit after — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Douc et al. [PP], Definition 16.
  - $H_k:=h(X_k)+\sum_{j=1}^{\infty}(h(X_{k+jL})-h(Y_{k+(j-1)L}))$.
  - $(X_{t+L},Y_t)_{t\ge0}$ is Markov with kernel $\bar P$. $(X_t)_{t=0}^{L}$ is a $P$-chain from $\pi_0$, and "$Y_0\sim\pi_0$ independent of $(X_0,\dots,X_L)$. In particular, $(X_L,Y_0)\sim\pi_0P^L\otimes\pi_0$".
  - Meeting time: $\tau=\inf\{t>L:X_t=Y_{t-L}\}$.
  - Weights (eq. 3.6): the integer count $v_t=\lfloor (t-k)/L\rfloor-\lceil\max(L,t-\ell)/L\rceil+1$, applied as $v_t/(\ell-k+1)$.
  - Lagged estimators "can be retrieved … by considering the Poisson equation associated with the iterated Markov kernel $P^L$" — [Douc et al.](https://arxiv.org/abs/2206.05691)
- JRSSB authors' reply (2020) [PR], via OUP HTML. It says "Generalizing equation (2.1) in the paper, we write the lagged, time-averaged estimator as …". The displayed weight on $h(X_l)-h(Y_{l-L})$, summed over $l=k+L,\dots,\tau-1$, transcribes as $\min\big(1,\lceil(l-k)/L\rceil/(m-k+1)\big)$, with $\tau=\inf\{t\ge1:X_t=Y_{t-L}\}$. The tool reconstructed the fraction layout from flattened HTML. The reply adds: "Using a value of L that is larger than 1 is always possible and can yield a large improvement in efficiency." — [JRSSB article page incl. discussion and reply](https://academic.oup.com/jrsssb/article/82/3/543/7056129)

**`unbiasedmcmc` R package [CODE], commit 660c698**
- `H_bar()`.
  - The lag is inferred as `nrow(samples1) - nrow(samples2)`.
  - The MCMC part sums `samples1[(k+1):(m+1),]`, that is, $X_k,\dots,X_m$.
  - If `meetingtime > k + lag`, it adds `coefficient_t * Delta_t` for `time in (k+lag):(meetingtime-1)`. Here `coefficient_t <- floor((time-k)/lag) - ceiling(max(lag, time-m)/lag) + 1` and `Delta_t = h(X_t) - h(Y_{t-lag})`.
  - It returns `H_bar/(m-k+1)` and refuses `k` or `m` greater than the realised horizon — [H_bar.R L42–47](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/R/H_bar.R)
- `sample_coupled_chains()`.
  - Starts with `state1 <- rinit(); state2 <- rinit()`, two independent draws.
  - Advances state1 alone `lag` times.
  - Loop condition: `while ((time < max(meetingtime, m)) && (time < max_iterations))`.
  - Before meeting it calls `coupled_kernel` and sets `meetingtime <- time` when the kernel returns `identical`, so the earliest recordable meeting is `lag+1`.
  - After meeting it runs `state1 <- single_kernel(state1); state2 <- state1`, which enforces faithfulness.
  - It stores $X_0..X_{\max(m,\tau)}$ and $Y_0..Y_{\max(m,\tau)-L}$, and computes `cost <- lag + 2*(meetingtime - lag) + max(0, time - meetingtime)`.
  - `meetingtime` is documented as "equal to Inf if while loop was interrupted" — [coupled_chains.R L50–82](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/R/coupled_chains.R)
- `sample_unbiasedestimator()` is the memory-light, on-the-fly version with the same weights. It errors if `k > m` or `lag > m` — [unbiasedestimator.R L44–99](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/R/unbiasedestimator.R)
- C++ signed measure (`c_chains_to_measure.cpp`).
  - The header comment states the general-lag measure: $(m-k+1)^{-1}\sum_{t=k}^m\delta_{X_t}$ plus $(m-k+1)^{-1}\sum_{t=k+L}^{\tau-1}(\lfloor (t-k)/L\rfloor-\lceil\max(L,t-m)/L\rceil+1)\{\delta_{X_t}-\delta_{Y_{t-L}}\}$.
  - Number of atoms: $(m-k+1)+2\max(0,\tau-(k+L))$.
  - Bias-correction atoms get weights $\pm$coef$/(m-k+1)$.
  - The superseded coefficient remains commented out at lines 71–73: `// weights(index) = ceil(((double) time - k)/((double) lag)); //if ((m - k + 1) < weights(index)){ weights(index) = m - k + 1;` — [c_chains_to_measure.cpp](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/src/c_chains_to_measure.cpp)
- `estimator_bin.cpp` uses the same floor/ceil coefficient and keeps the same commented-out `ceil(...)`-capped formula. Bin membership is tested with strict inequalities `> lower && < upper` — [estimator_bin.cpp L16–35](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/src/estimator_bin.cpp)

### Inferences
- [OWN] Derivation of the weights. $H_{k:m}=(m-k+1)^{-1}\sum_{l=k}^m[h(X_l)+\sum_{j\ge1}\Delta_{l+jL}]$, so the coefficient of $\Delta_t$ counts the pairs $(l,j)$ with $l\in[k,m]$, $j\ge1$ and $l+jL=t$. That count is $\#\{j:\max(1,\lceil (t-m)/L\rceil)\le j\le\lfloor (t-k)/L\rfloor\}$. Since $\max(1,\lceil (t-m)/L\rceil)=\lceil\max(L,t-m)/L\rceil$, this is the floor/ceil formula. Special cases:
  - $L=1$ gives $\min(t-k,\,m-k+1)$, recovering Jacob et al. eq. (2.1).
  - $m=k$ gives $1$ if $L\mid(t-k)$ and $0$ otherwise, recovering $H^{(L)}_k$.
  - $m=k+L-1$ makes every bias-correction coefficient equal to 1. This gives Vanetti–Doucet's estimator $L^{-1}[\sum_{t=k}^{k+L-1}h(X_t)+\sum_{t=k+L}^{\tau-1}\Delta_t]$.
  - Biswas et al.'s inner upper limit $\lceil(\tau-L-t)/L\rceil$ equals $\lfloor(\tau-1-t)/L\rfloor$ for integers, so it agrees with summing up to $\tau-1$.
- [OWN] The rejoinder's lag-$L$ weight is wrong for $L\ge2$, while the floor/ceil weight is correct. I simulated a 3-state chain with a maximal coupling of transition rows, $\pi_0=\delta_0$, $h=(0,1,5)$ and $\pi(h)=1.67857$, using $2\times10^5$ replicates per configuration (script: `/tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/theory/scripts/check_lag_weights.py`).
  - The definition-based average of $H_l$ equalled the floor/ceil formula to $\le 2\times10^{-15}$ in every replicate.
  - Mean of the correct formula versus mean of the rejoinder formula (standard errors ≈0.002–0.006):

    | $(k,m,L)$ | correct | rejoinder |
    |---|---|---|
    | (0,0,2) | 1.6806 | 2.0607 |
    | (0,3,2) | 1.6864 | 1.7844 |
    | (1,6,3) | 1.6778 | 1.6963 |
    | (0,5,4) | 1.6797 | 1.7620 |
    | (2,4,1) | identical | identical |

  - Implement the floor/ceil weights. Also add a unit test asserting the identity with the definition-based $\frac{1}{m-k+1}\sum_l H^{(L)}_l$.
- [OWN] Convention crosswalk to pin down in code and docs:

  | item | Jacob et al. 2020 | Biswas et al. 2019 | Atchadé–Jacob 2024 | Douc et al. 2025 | R code | JRSSB reply |
  |---|---|---|---|---|---|---|
  | last index | $m$ | $m$ | $\ell$ | $\ell$ | `m` | $m$ |
  | meeting time | $\inf\{t\ge1:X_t=Y_{t-1}\}$ | $\inf\{t>L:X_t=Y_{t-L}\}$ | $\inf\{t\ge L:\dots\}$ | $\inf\{t>L:\dots\}$ (for $G_y$, $L=0$: $\inf\{t\ge0:X_t=Y_t\}$) | first $t\ge L+1$ with kernel flag `identical` | $\inf\{t\ge1:X_t=Y_{t-L}\}$ |
  | weight object | $\min(1,\frac{t-k}{m-k+1})$ | via inner sum limit | $v_t$ includes $1/(\ell-k+1)$ | integer $v_t$, divided separately | integer, divided at end | wrong for $L\ge2$ |
  | "$m$" also means | – | – | moment index of $h$ | moment index of $h$ | – | – |
- [OWN] Effect of the meeting-time conventions. The conventions differ only in whether $X_L=Y_0$ can count as a meeting, which happens with positive probability in discrete spaces, for MH under rejection, or with atoms in $\pi_0$. They do not change the value of $H$ provided the coupled kernel is faithful from equal states, because $\Delta_t=0$ whenever $X_t=Y_{t-L}$. They do change the recorded $\tau$, the cost, and any $\tau$-quantile-based tuning. My simulation used the $\inf\{t\ge L\}$ convention and showed min $\tau=L$.
- [OWN] Faithfulness should be enforced in the driver after detecting meeting, as the R code does with `state2 <- state1`, rather than delegated to the coupled kernel. The review's sub-maximal coupling (Algorithm 2 with $\eta<1$) returns $Y\neq X$ with probability $1-\eta$ even when $p=q$, so a kernel built on it is not automatically faithful (from reading Algorithm 2 in [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)). Detect meeting by exact state equality, or by a kernel flag that is guaranteed to agree with equality.
- [OWN] The review's atom count "$N=\max(0,\tau-(k+L))+(\ell-k+1)$" (Sec. 2.4) appears to omit a factor 2. Each $t\in\{k+L,\dots,\tau-1\}$ contributes the two atoms $X_t$ and $Y_{t-L}$, matching Douc et al. Sec. 3.3 and the C++ comment.
- [OWN] The R restriction `lag <= m` in the on-the-fly function is an implementation convenience, not a theoretical requirement; the estimator is well defined for any $L\ge1$ and $m\ge k\ge0$. `telescopic` should define behaviour for $m<L$ explicitly, for example by allowing it and running X until $\max(m,\tau)$.

### Gaps
- I did not read the originals of McLeish (2011, Monte Carlo Methods Appl. 17(4):301–315) or Rhee & Glynn (2015, Oper. Res. 63(5):1026–1043); their formulas above come from Vihola's restatement.
- I saw the rejoinder's lag-$L$ formula only through a fetch tool's transcription of the OUP HTML; it should be checked against the typeset PDF before citing it as an erratum. I found no published erratum notice.
- I did not diff the published JRSSB text against arXiv v5. From the way later papers cite it, the published Propositions 1–4 appear to correspond to arXiv Propositions 3.1–3.4, but that is inferred, not verified.
- A revised version of the Atchadé–Jacob chapter (math.bu.edu/people/atchade/umcmc_rev.pdf) was unreachable (TLS verification failure, then HTTP 503). It might correct the atom-count slip.

## 2. Assumptions for unbiasedness and finite variance; what happens when they fail and how to detect it from output

### Takeaway
The assumption families are:
- Original paper: geometric tails of $\tau$, plus a uniform $2+\eta$ moment of $h$ along the chain, plus faithfulness.
- Middleton et al.: polynomial tails $P(\tau>n)\le Kn^{-\kappa}$ with $\kappa>2(2/\eta+1)$.
- Douc et al. and the review: $\mathbb E_{\pi\otimes\pi}[\tau^\kappa]<\infty$ for chains started at stationarity, plus $d\pi_0/d\pi\le M$.

Under the last family, $H$ is unbiased with $\mathbb E|H|^p<\infty$ whenever $1/p>1/m+1/\kappa$, where $h\in L^m(\pi)$. Finite variance needs $\kappa>2$ and $m>2\kappa/(\kappa-2)$.

When tails are heavy, or a chain can rarely escape to another mode before meeting, the estimator stays unbiased but its variance becomes huge or infinite. The output then looks fine for moderate $R$. The main warning signs are rare huge meeting times and large empirical variance. There is no certificate.

### Cited Findings
- Jacob et al., Proposition 3.1 [PR]: "Under Assumptions 2.1-2.3, for all k ≥ 0 and m ≥ k, the estimator Hk:m(X, Y) has expectation Eπ[h(X)], a finite variance, and a finite expected computing time." The constants $D,\eta,C,\delta$ need not be known — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Jacob et al., Proposition 3.4 (verifying geometric tails). Assume a geometric drift $PV\le\lambda V+b\mathbb 1_C$ with small set $C=\{x:V(x)\le L\}$, $\lambda+b/(1+L)<1$, and $\inf_{(x,y)\in C\times C}\bar P((x,y),D)\ge\epsilon$ with $D$ the diagonal. Then $P(\tau>n)\le C'\pi_0(V)\kappa^n$, so Assumption 2.2 holds if $\pi_0(V)<\infty$. Here $L$ is the drift level-set constant, not the lag — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Douc et al., Proposition 11, is the analogue for $\pi\otimes\pi$ initialisation. It requires $\lambda+2b/(1+\bar v)<1$ and concludes that "Assumption 2 holds for all κ > 1" — [Douc et al.](https://arxiv.org/abs/2206.05691)
- Middleton et al. (EJS 2020) [PR] replace geometric tails with Assumption 2: $\tau=\inf\{n\ge1:Z_n=\tilde Z_{n-1}\}$ and $P(\tau>n)\le Kn^{-\kappa}$ with $\kappa>2(2\eta^{-1}+1)$, with $\eta$ from the $2+\eta$ moment assumption.
  - Theorem 1: unbiased, finite expected cost and finite variance for all $k\le m$.
  - This tail condition gives $\mathbb E[\tau^p]<\infty$ for $p<2(2\eta^{-1}+1)$.
  - Theorem 2: a polynomial drift with exponent $\alpha$ gives $\kappa=1/(1-\alpha)$. With $\eta=1$ one needs $\kappa$ "just above 6", i.e. $\alpha>5/6$ — [Middleton et al. arXiv:1807.08691](https://arxiv.org/abs/1807.08691)
- Douc et al., Assumption 2 [PP]: "The Markov transition kernel P is π-irreducible and for some κ > 1, $E_{\pi\otimes\pi}[\tau^\kappa]<\infty$", with $\tau=\inf\{t\ge0:X_t=Y_t\}$, lag 0, both chains started from π independently.
  - Proposition 8: this is essentially equivalent to polynomial decay of $P_{x,y}(\tau>t)$.
  - Theorem 3: if $h\in L^m(\pi)$ with $m>2\kappa/(\kappa-1)$, the CLT holds with $v(P,h)=2\pi(h_0g_\star)-\pi(h_0^2)<\infty$.
  - Proposition 13: a polynomial drift with exponent $\alpha$ gives the assumption for $\kappa<\alpha/(1-\alpha)$ — [Douc et al.](https://arxiv.org/abs/2206.05691)
- Douc et al., Theorem 17 and Remark 18. Under Assumption 2, $h\in L^m(\pi)$ with $m>\kappa/(\kappa-1)$ and $d\pi_0/d\pi\le M$, for any $k\le\ell$ and $L\ge1$: $\mathbb E[H_{k:\ell}]=\pi(h)$ and $\mathbb E|H_{k:\ell}|^p<\infty$ for $p\ge1$ with $1/p>1/m+1/\kappa$. "It is sufficient that κ > 2 and m > 2κ/(κ − 2) for H to have a finite variance". This is stronger than the CLT condition for the underlying chain, "because finite second moment of the unbiased estimator is shown via finite second moment of the approximation of $g_\star$" — [Douc et al.](https://arxiv.org/abs/2206.05691)
- The Atchadé–Jacob review restates the result as Theorem 2.1. It notes that $d\pi_0/d\pi\le M$ "rules out the choice of π0 as a Dirac mass" for continuous π, "but it allows π0 to be the Uniform distribution on any ball with positive radius". "If τ has Geometric tails, then κ can be taken arbitrary large, and the result holds for all h ∈ L2+η(π)" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Biswas et al.'s proof of unbiasedness and finite variance for $H^{(L)}_t$ uses Hölder's inequality with $p=1+\eta/2$: $\mathbb E[\Delta_t^2]\le\mathbb E[\Delta_t^{2+\eta}]^{1/(1+\eta/2)}P(\tau^{(L)}>(1+t)L)^{\eta/(2+\eta)}$. The partial sums then form a Cauchy sequence in $L^2$ — [Biswas et al.](https://arxiv.org/abs/1905.09971)
- Failure example 1 (Jacob et al. §5.1): a bimodal target with RWM, $\sigma_q^2=1$ and $\pi_0=\mathcal N(10,1)$.
  - Mean $\tau$ was 9, with a 99% quantile of 35.
  - With $k=50$, $m=500$ and $R=1000$, the 95% CI was [0.799, 0.816], "far from the true value of 0.421".
  - Adding 9,000 estimators gave [−0.353, 1.595]: "a few correspond to cases where one chain jumped to the left-most mode before meeting the other".
  - "Upon noticing a large empirical variance one can then decide to use larger values of k and m"; "poor performance of the underlying Markov chains combined with ill-chosen initializations can still produce misleading results for any finite R" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Failure example 2 (Jacob et al. §5.1): with $\sigma_q^2=1$ and $\pi_0=\mathcal N(10,10^2)$, mean $\tau$ was 769 with a 99% quantile of 9,186. Large $k=20{,}000$ and $m=30{,}000$ gave a correct CI [0.397, 0.430] — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Jacob et al. §6: "Unreliable estimators would also result from stopping the chains before their meeting time", and "coupling pairs of Markov chains does not improve their marginal mixing properties" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Atchadé–Jacob, "Limitation" paragraph, on multimodal targets with $\pi_0$ concentrated in one mode: "The user might then observe a small empirical average of the meeting time, even after many independent runs. Yet the expectation of the meeting time could be much larger." The risk "is mitigated by specifying an initial distribution π0 that is spread out relative to the modes of π, or by increasing the lag L" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- JRSSB discussion, via OUP HTML [DISC]:
  - Wilkinson reported an AR(1) Gibbs example with mean coupling time ≈2.6×10³ and maximum 67,000.
  - Ryder, Clarté, Hairault, Lawless & Robert argued that a large $\mathbb E[\tau]$ can warn that the kernel is poorly suited to the target.
  - The reply says large meeting times "can indeed be seen as a feature rather than a bug" and "can sometimes indicate convergence issues with the underlying Markov kernel" — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)
- Sherlock (seconder, via OUP HTML) gave the heuristic $\mathrm{var}(BC_{k:m}\mid\tau)=O\{(\tau-k)^3/(m-k)^2\}$ on $\tau>k+1$, motivating a large $k$ and $m\gg\tau-k$ — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)

### Inferences
- [OWN] Moment arithmetic from Douc et al. Theorem 17, to state in docs:
  - For bounded $h$ ($m=\infty$), $\mathbb E|H|^p<\infty$ needs only $p<\kappa$. So $\kappa>2$, slightly more than two moments of $\tau$ from $\pi\otimes\pi$, suffices for a finite variance.
  - For $h$ with only $2+\eta$ moments, $\kappa$ must be large. Geometric tails cover every $\kappa$.
  - If $\kappa\in(1,2]$, $H$ is still unbiased and the average is still consistent ($\mathbb E|H|<\infty$), but the CLT-based standard errors are invalid and the convergence rate is slower than $R^{-1/2}$.
- [OWN] Output-based warning signs worth implementing, all heuristic since none is a formal test in the sources:
  - Running mean and running sample variance of $H$ against $R$. The variance should stabilise.
  - The ratio $\max_r|H_r - \bar H|^2/\sum_r (H_r-\bar H)^2$. A few replicates dominating signals heavy tails.
  - A log-log survival plot of $\tau$ and $\tau-L$, with a Hill-type tail-index estimate (κ̂ near 2 or below is a red flag).
  - Comparing CIs from disjoint batches of replicates.
  - Sensitivity of $\bar H$ to $(k,m,L)$.
  - Pilot runs from several overdispersed $\pi_0$.
  - None of these can rule out rare escapes, as Jacob et al. §5.1 shows.
- [OWN] A `max_iterations` cap must be reported as a failure, never handled by silently truncating or dropping replicates. Both bias the estimator, consistent with Jacob et al.'s warning about stopping before meeting and Glynn–Heidelberger's analysis of completion constraints (section 6). The R package returns `meetingtime = Inf` in this case.
- [OWN] Choose $\pi_0$ overdispersed relative to π but not degenerate. The theory in Douc et al. and the review needs $d\pi_0/d\pi$ bounded, and Jacob et al. need uniform $2+\eta$ moments along the chain from $\pi_0$.

### Gaps
- I found no published formal procedure, such as a hypothesis test or tail-index estimator, specific to detecting infinite variance or heavy-tailed $\tau$ in unbiased MCMC output. The diagnostics above are my suggestions.
- Jacob et al. state the drift-to-tails result with $\lambda+b/(1+L)<1$ and Douc et al. with $\lambda+2b/(1+\bar v)<1$. I did not check the proofs to see whether one corrects the other.

## 3. Cost and efficiency: exact cost, variance, inefficiency, choice of $k,m,L$, discussion and rejoinder, review, newer results

### Takeaway
One $H^{(L)}_{k:m}$ costs $L+2(\tau-L)+\max(0,m-\tau)$ units, counting a $\bar P$ step before meeting as 2 units. That is $\tau-L$ coupled steps plus $L+\max(0,m-\tau)$ single steps, independent of $k$. For $L=1$ it reduces to $2(\tau-1)+\max(1,m+1-\tau)$.

Efficiency is measured by the inefficiency $\mathbb E[\text{cost}]\times\mathbb V[H]$ (Glynn–Whitt), which is compared with the MCMC asymptotic variance $v(P,h)$. As $m\to\infty$, $\sqrt{m-k+1}(H_{k:m}-\pi(h))\Rightarrow\mathcal N(0,v(P,h))$, and the loss is roughly $1+(k+\mathbb E\tau)/(m-k)$.

Current practical guidance (Atchadé–Jacob 2024):
1. Run a pilot with $L=1$.
2. Set $k$ to the 99% quantile of $\tau-L$.
3. Reset $L=k$.
4. Set $m=10k$.

Douc et al. instead use $L$ equal to a large quantile of $\tau$, $k=L$ and $m=5k$. I found no 2025–2026 paper that optimises $(k,m,L)$ formally.

### Cited Findings
- Exact cost.
  - Jacob et al.: $\tau-1$ calls to $\bar P$ and $\max(1,m+1-\tau)$ calls to $P$, with "calls to $\bar P$ … approximately twice as expensive as calls to P", giving a cost of $2(\tau-1)+\max(1,m+1-\tau)$ — [Jacob et al.](https://arxiv.org/abs/1708.03625)
  - With lag: $L+2(\tau-L)+\max(0,\ell-\tau)$ — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851); the same expression appears in code as `cost <- lag + 2*(meetingtime - lag) + max(0, time - meetingtime)` — [coupled_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/R/coupled_chains.R)
  - Middleton et al. give the same quantity as $T_m:=2(\tau-1)+\max(1,m-\tau+1)$ — [Middleton et al.](https://arxiv.org/abs/1807.08691)
- Biswas et al.: "Obtaining τ(L) requires sampling L times from K and τ(L) − L from K̄." For large L "the distribution of τ(L) − L depends entirely on K̄ and not on L", and "the cost of obtaining τ(L) increases linearly in L" — [Biswas et al.](https://arxiv.org/abs/1905.09971)
- Inefficiency. "Asymptotic inefficiency as the product of the variance and expected cost … the asymptotic variance of $R^{-1}\sum_r H^{(r)}_{k:m}$ as the computational budget … goes to infinity [Glynn and Whitt, 1992]" — [Jacob et al.](https://arxiv.org/abs/1708.03625). Middleton et al. write it as "IF[Hk:m] := E[Tm]V[Hk:m]" — [Middleton et al.](https://arxiv.org/abs/1807.08691)
- Variance decomposition (Jacob et al. §3.1). $H_{k:m}=\mathrm{MCMC}_{k:m}+\mathrm{BC}_{k:m}$, and with $\mathrm{MSE}_{k:m}=\mathbb E(\mathrm{MCMC}_{k:m}-\pi h)^2$:

  $$\mathbb V[H_{k:m}]\le \mathrm{MSE}_{k:m}+2\sqrt{\mathrm{MSE}_{k:m}\,\mathbb E[\mathrm{BC}^2_{k:m}]}+\mathbb E[\mathrm{BC}^2_{k:m}]\quad(3.1).$$

  Proposition 3.3 assumes geometric drift and $|h|\le cV^\beta$ with $\beta<1/2$, and gives $\mathbb E[\mathrm{BC}^2_{k:m}]\le C_{\delta,\beta}\delta_\beta^k/(m-k+1)^2$ with $\delta_\beta=\delta^{1-2\beta}$ — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Approximate efficiency (Jacob et al. §3.1).

  $$\mathbb E[\text{cost}]\,\mathbb V[H_{k:m}]\lesssim (m+\mathbb E\tau)\mathbb V[H]\lesssim (m-k+1)\mathrm{MSE}_{k:m}\Big(1+\tfrac{k+\mathbb E\tau}{m-k+1}\Big)\Big(1+\tfrac{2}{\sqrt{m-k+1}}\sqrt{\tfrac{C_{\delta,\beta}\delta_\beta^k}{(m-k+1)\mathrm{MSE}_{k:m}}}\Big).$$

  Here $(m-k+1)\mathrm{MSE}_{k:m}\to V_{k,m}\to V_\infty$ (the MCMC asymptotic variance). Hence "the loss of efficiency … is approximately 1 + (k + E(τ))/(m − k)" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Jacob et al. guidance: "choose k as a large quantile of the meeting times". "Choosing m as a multiple of k, such as 5k or 10k, makes intuitive sense when considering that k/m is the proportion of iterations that are simply discarded in the event that τ < k." There is a "tradeoff with the desired level of parallelism: one might prefer to keep k and m small" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Empirical example (Jacob et al. Table 1): bimodal target, $\sigma_q^2=9$, mean $\tau=20$, 99% quantile 105. Inefficiency$/V_\infty$ by $(k,m)$:

  | $(k,m)$ | Inefficiency$/V_\infty$ |
  |---|---|
  | (1, k) | 1878.4 |
  | (100, 10k) | 2.9 |
  | (200, 10k) | 1.3 |
  | (200, 20k) | 1.2 |

  — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Douc et al., Proposition 19 [PP]: under the conditions of their Theorem 3, for any $k$ and $L\ge1$, $\sqrt{\ell-k+1}(H_{k:\ell}-\pi(h))\Rightarrow\mathcal N(0,v(P,h))$ as $\ell\to\infty$. They "expect the inefficiency … to be at least ℓ/(ℓ − k) that of ergodic average MCMC". Their tuning: pilot with $L=1$, "re-define L as a large quantile of the meeting times, and we set k = L. We choose ℓ as a multiple of k such as 5k" — [Douc et al.](https://arxiv.org/abs/2206.05691)
- Atchadé–Jacob, concrete guideline: "First generate independent meeting times with lag L = 1 … Then set k as a large (e.g. 99%) quantile of τ − L … Finally, redefine L = k, and set ℓ = 10k."
  - "Setting L = k leads to a minor increase of cost per estimator compared to L = 1, and thus the efficiency is typically improved, sometimes drastically."
  - Their Figure 5 plots relative inefficiency from about $10^4$ for (k=10, L=1) down to about 1 for k=400–800 with ℓ=10k (values read approximately from a log-scale plot) — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Middleton et al., Proposition 1 (polynomial drift with $5/6<\alpha\le1$, $\kappa=1/(1-\alpha)$): $\mathbb E[\mathrm{BC}^2_{k:m}]\le B\{m^{-(\kappa/2-1)}+(m-k+1)^{-2}k^{-(\kappa/2-3)}\}$ — [Middleton et al.](https://arxiv.org/abs/1807.08691)
- Vanetti & Doucet, Table 1 [DISC]: standard deviation of a single estimate for RWM with $\pi=\mathcal N(0,1)$, $\pi_0=\mathcal N(0,5^2)$, $h=x^2$:

  | lag 1, $(k,m)$ | SD | lag $L=m-k$, $(k,L)$ | SD |
  |---|---|---|---|
  | (1, 10) | 430 | (1, 9) | 66.7 |
  | (10, 100) | 34.2 | (10, 90) | 11.9 |
  | (100, 1000) | 0.119 | (100, 900) | 0.119 |

  — [Vanetti & Doucet PDF](https://www.stats.ox.ac.uk/~doucet/discussion_VanettiDoucet_unbiasedMCMC.pdf)
- JRSSB discussion, via OUP HTML [DISC]:
  - Wilkinson (opening) applied Amdahl's law and concluded the approach suits "of the order of 10" processors and is less efficient beyond about $10^2$.
  - Sherlock proposed Rao–Blackwellising the bias correction for coupled MH, reporting an MSE reduction by "a factor of 30 at k = 10^4.25" at less than twice the cost.
  - Craiu & Meng proposed control variates.
  - Lee, Singh & Vihola analysed coupled conditional backward-sampling particle filters, with $P(\tau_T>t)\le C_T\delta^t$ and $\log C_T=O(T)$.
  - Paulin asked about dimension dependence.
  - Robert et al. cautioned that unbiasedness is not the ultimate goal — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)
- JRSSB authors' reply, via OUP HTML [PR]:
  - It distinguishes two kinds of parallel gain, $(2\alpha,T)$ (more precision at equal time) versus $(\alpha,T/2)$ (same precision in half the time): "The class of estimators … achieves the first type of improvement in most settings", while "the second type of gain is realistic only when starting from a moderate or large run time T". Table 12 uses $H\sim\mathrm{Exp}(1)$ with cost $C=1+H$.
  - "The meeting time drives both the cost of the estimator and its variance via the bias correction term."
  - "Common random numbers" are "a good starting point".
  - Lag $L>1$ "can yield a large improvement in efficiency" — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)
- Wang, Blanchet & Glynn (2024) [PP] compare unbiased and biased estimators on completion time and total cost. Unbiased estimators "typically have superior completion times", with the advantage governed by the tail of the running-time distribution, but "may not automatically provide substantial savings in overall computational costs". The framework is applied to MCMC and MLMC — [arXiv:2404.01431](https://arxiv.org/abs/2404.01431)
- Kahalé (2024) [PR] gives unbiased time-average estimators for Markov chains, based on randomized multilevel ideas and a coupling assumption. The modified estimator is square-integrable with finite expected running time and "asymptotically at least as efficient as the original estimator, up to a multiplicative constant arbitrarily close to 1" — [arXiv:2209.09581](https://arxiv.org/abs/2209.09581) (Math. Oper. Res. 49(4):2136–2165, DOI 10.1287/moor.2022.0326)

### Inferences
- [OWN] If $\tau\le m$ almost surely, $\mathbb E[\text{cost}]=m+\mathbb E[\tau]-L$. With $L=k$, $\tau-L$ is approximately the number of coupled steps, which does not depend on $L$ for large $L$ (Biswas et al.). The overhead over plain MCMC with $m$ steps is then about $\mathbb E[\tau-L]$, and the discarded fraction is $k/m$.
- [OWN] With the $L=k$ recipe, the relative inefficiency is at least about $m/(m-k+1)\approx1.11$ for $m=10k$, or $\approx1.25$ for $m=5k$, plus terms from the bias correction and from the variance of $\tau$.
- [OWN] Choose $(k,m,L)$ from an independent pilot batch, never from the replicates being averaged. Data-dependent $k$ breaks exact unbiasedness, in the same way that Craiu–Meng's control-variate coefficients must be estimated leave-one-out (section 5).
- [OWN] Measure efficiency from output as `mean(cost) * var(H)`. Compare it to $v(P,h)$ estimated by UPAVE (section 5), or by spectral or batch-means estimators on a long chain. Report the ratio as "relative inefficiency".
- [OWN] Cost units matter for black-box log-densities. "Coupled step = 2 units" holds when each chain needs one log-density evaluation per step. The review's coupled RWM code evaluates $U$ at both proposals, while the package's example kernel (`check_lags.R`) skips the second evaluation when the proposals coincide. `telescopic` should count log-density evaluations and wall time directly, not assumed units.
- [OWN] The rejoinder's Amdahl argument implies that a requested CI width at a fixed deadline is easier to reach by adding processors than by shortening per-replicate time. Per-replicate time is bounded below by roughly $m+\mathbb E\tau$, and the deadline must exceed the slowest required replicate (section 6).

### Gaps
- I found no peer-reviewed analysis of the optimal $(k,m,L)$ for general chains. The $L=k$, $m=10k$ recipe is heuristic ("In our experience"). Searches for 2025–2026 work on automatic tuning found nothing specific.
- I could not obtain the details of Sherlock's Rao–Blackwellisation or of Lee–Singh–Vihola's analysis beyond the fetch-tool summary.

## 4. Signed-measure representation; unbiased variances, covariances, quantiles, histograms, credible intervals; nonlinear functionals; standard errors and CIs across $R$ replicates; CLT conditions and robust alternatives

### Takeaway
Each replicate yields a signed empirical measure $\hat\pi=\sum_{n=1}^N\omega_n\delta_{Z_n}$. It has:
- $N=(m-k+1)+2\max(0,\tau-k-L)$ atoms;
- weights $1/(m-k+1)$ on $X_k,\dots,X_m$;
- weights $\pm v_t/(m-k+1)$ on the pairs $(X_t,Y_{t-L})$;
- $\sum_n\omega_n=1$.

$\hat\pi(h)$ is unbiased for every $h$ meeting the moment conditions, including indicators, so histogram bin probabilities and CDF values are unbiased. Quantiles from the pooled, non-monotone CDF are consistent but not unbiased.

Nonlinear functionals need products of independent replicates. Variances and covariances have exact U-statistic estimators. A general $g(\pi h)$ needs unbiased MLMC on top.

CIs are $\bar H\pm z\,\hat\sigma/\sqrt R$ under finite variance. Under heavy tails, the review suggests robust aggregators (trimmed means, median-of-means), which trade exact unbiasedness for concentration.

### Cited Findings
- Signed measure (Jacob et al. eq. 2.2, $L=1$): $\hat\pi=(m-k+1)^{-1}\sum_{\ell=k}^m\delta_{X_\ell}+\sum_{\ell=k+1}^{\tau-1}\min(1,\tfrac{\ell-k}{m-k+1})(\delta_{X_\ell}-\delta_{Y_{\ell-1}})$. It has the form $\sum_\ell\omega_\ell\delta_{Z_\ell}$ with $\sum\omega_\ell=1$, and "Some of the weights … may be negative" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Lag-$L$ signed measure: review eq. (2.12) — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851). Douc et al. eqs. (3.5)–(3.7) give "$N=(\ell-k+1)+2\max(0,\tau-(k+L))$". The first $\ell-k+1$ atoms are $X_k..X_\ell$, the rest are $X_{k+L},Y_k,\dots,X_{\tau-1},Y_{\tau-1-L}$, and "There may be duplicate atoms" — [Douc et al.](https://arxiv.org/abs/2206.05691)
- CDF and quantiles (Jacob et al. Proposition 3.2). Under Assumptions 2.2–2.3 and TV convergence, $\hat F^R(s)=\sum_\ell\omega_\ell\mathbb 1(Z_\ell\le s)$ over $R$ pooled replicates satisfies $\sup_s|\hat F^R(s)-F(s)|\to0$ almost surely. Because $\hat F^R$ "is not monotonically increasing", there may be several indices with $\sum_{i<\ell}\omega_i\le q<\sum_{i\le\ell}\omega_i$. Any such $Z_\ell$ is a consistent quantile estimate.
  - Confidence intervals for quantiles, "perhaps by bootstrapping the R independent copies", are left as future research.
  - The paper also mentions projection onto probability measures (a Wasserstein generalisation for signed measures, Mainini 2012) and isotonic regression (Chatterjee et al. 2015) as alternatives — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Histograms. They are "produced by averaging unbiased estimators of expectations of indicator functions, corresponding to consecutive intervals. Confidence intervals at level 95% are obtained from the central limit theorem" — [Jacob et al.](https://arxiv.org/abs/1708.03625). The package computes per-bin `mean(estimators)` and `sd(estimators)/sqrt(nsamples)` and plots ±2 sd — [histogram_c_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/660c69824d3e31b109070b647578d14b045c33d5/R/histogram_c_chains.R)
- The review's density plots were "created using kernel density estimation from the weighted samples" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Subsampling atoms.
  - Draw $I$ uniformly; then $N\omega_I h(Z_I)$ is unbiased. More generally, use selection probabilities $\xi_n$ with weight $\xi_I^{-1}\omega_I$ — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
  - Douc et al. Theorem 20: $S_R=R^{-1}\sum_{i=1}^R\xi_{I_i}^{-1}\omega_{I_i}h(Z_{I_i})$ with $a/N\le\xi_i\le b/N$ has the same unbiasedness and moment conditions as $H_{k:\ell}$. Remark 21: $\mathrm{var}(S_R)=R^{-1}\mathbb E[\mathrm{var}(S_1\mid\hat\pi)]+\mathrm{var}(\hat\pi(h))$ — [Douc et al.](https://arxiv.org/abs/2206.05691)
- Negative weights cannot be eliminated. "We can arbitrarily decrease the proportion of negative weights … by increasing the value of k, but we cannot make it zero." By Jacob & Thiery (2015, Lemma 2.1) "there is no algorithm that takes unbiased estimators of a nonnegative quantity as input (and nothing else), and returns nonnegative unbiased estimators" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Unbiased variance of $h$ under π from two independent signed measures:

  $$\hat v(\pi,h)=\tfrac12\{\hat\pi^{(1)}(h^2)+\hat\pi^{(2)}(h^2)\}-\hat\pi^{(1)}(h)\,\hat\pi^{(2)}(h)\quad\text{(Douc et al. eq. 4.3; review eq. 3.6)}$$

  — [Douc et al.](https://arxiv.org/abs/2206.05691); [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- General nonlinear $g(\pi(h))$: Wang & Wang (JMLR 2023) "combines and generalizes the unbiased MCMC and Multilevel Monte Carlo (MLMC) methods", reaching ε-accuracy at the optimal $O(1/\epsilon^2)$ cost "under mild conditions" and extending to nested expectations — [arXiv:2204.04808](https://arxiv.org/abs/2204.04808); the JMLR venue is from the [review's reference list](https://arxiv.org/abs/2406.06851). Blanchet, Glynn & Pei (2019, [PP]) develop unbiased MLMC for optimisation, regenerative steady-state simulation and quantile estimation — [arXiv:1904.09929](https://arxiv.org/abs/1904.09929)
- Nested or modular targets: unbiased estimators can be nested through the law of total expectation (cut distribution) — [Jacob et al.](https://arxiv.org/abs/1708.03625); [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Standard CI. With $C$ independent copies, mean $\hat\mu$ and standard deviation $\hat\sigma$, $[\hat\mu+q_{\alpha/2}\hat\sigma/\sqrt C,\ \hat\mu+q_{1-\alpha/2}\hat\sigma/\sqrt C]$ is asymptotically valid as $C\to\infty$ given finite variance — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851). Middleton et al.: "a central limit theorem holds provided that V[Hk:m] < ∞" — [Middleton et al.](https://arxiv.org/abs/1807.08691)
- Robust aggregation: "one can readily replace empirical averages by more robust estimators of expectations. Nguyen et al. (2022) consider trimmed means. One could naturally employ median-of-means estimators (Lugosi and Mendelson, 2019), empirical risk optimizers (Sun, 2024) or estimators based on self-normalized sums (Minsker and Ndaoud, 2021) to aggregate unbiased MCMC estimators π̂(h) that have two finite moments" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)

### Inferences
- [OWN] Exact U-statistic estimators for $R\ge2$ independent replicates. Write $a_r=\hat\pi^{(r)}(h)$, $b_r=\hat\pi^{(r)}(h')$, with sample mean $\bar a$, sample variance $s_a^2$ and sample covariance $s_{ab}$ (denominator $R-1$). Independence gives $\mathbb E[a_ra_s]=\pi(h)^2$ for $r\ne s$, which yields:
  - $\widehat{\pi(h)^2}=\frac{1}{R(R-1)}\sum_{r\ne s}a_ra_s=\bar a^2-s_a^2/R$.
  - $\widehat{\mathrm{Var}_\pi(h)}=\frac1R\sum_r\hat\pi^{(r)}(h^2)-(\bar a^2-s_a^2/R)$, where $\hat\pi^{(r)}(h^2)$ applies the same signed weights to $h^2$ at the atoms.
  - $\widehat{\mathrm{Cov}_\pi(h,h')}=\frac1R\sum_r\hat\pi^{(r)}(hh')-(\bar a\bar b-s_{ab}/R)$.
  - These generalise Douc et al.'s two-replicate formula and are exactly unbiased given finite second moments. They can be negative.
  - The pooled plug-in $\sum\omega h^2-(\sum\omega h)^2$ is biased by $-\mathrm{Var}(\bar a)$.
- [OWN] Derived quantities such as standardised moments, correlations or credible-interval endpoints are consistent but not unbiased (delta method, $O(1/R)$ bias). Exact unbiasedness for a general smooth $g$ needs the randomized-MLMC route of Wang & Wang.
- [OWN] Recommended quantile and credible-interval procedure:
  1. Pool all atoms and weights, each replicate's weights divided by $R$.
  2. Sort the atoms and take cumulative sums of the weights.
  3. Monotonise the CDF, for example with a running max clipped to $[0,1]$ or with isotonic regression.
  4. Invert it to get the quantiles.
  5. Get uncertainty by bootstrapping whole replicates, not atoms, since the replicate is the i.i.d. unit.

  This is consistent as $R\to\infty$ but carries no finite-$R$ guarantee. Credible intervals built from it are approximations of a consistent but biased kind.
- [OWN] A weighted KDE with fixed bandwidth $b$ is unbiased for the smoothed density $(\pi*K_b)(x)$, not for $\pi(x)$, and can be negative. Histogram bins with strict inequalities, as in the R code, should use half-open bins $[a,b)$ so that bin probabilities sum exactly to $\sum\omega=1$.
- [OWN] For vector $h$, report the sample covariance matrix of the replicate vectors divided by $R$ as the Monte Carlo covariance of $\bar H$, and use Hotelling-type or marginal intervals.
- [OWN] Median-of-means and trimmed means are biased estimators with better deviation bounds. If they are offered, label them "robust, not unbiased" and keep the plain mean as the default.

### Gaps
- I found no source giving exactly unbiased quantile estimators from coupled chains. Blanchet–Glynn–Pei's quantile construction covers regenerative or other settings, and I read only its abstract.
- I found no source evaluating the coverage of bootstrap CIs for quantiles from signed measures; Jacob et al. leave it as future research.

## 5. Asymptotic-variance estimation with couplings (Douc–Jacob–Lee–Vats), control variates (Craiu–Meng), Rao–Blackwellisation and other variance reduction

### Takeaway
Coupled chains with lag 0 started from $(x,y)$ give unbiased estimates $G_y(x)=\sum_{t=0}^{\tau-1}(h(X_t)-h(Y_t))$ of the Poisson-equation solution $g_y(x)=g_\star(x)-g_\star(y)$. Combined with two independent signed measures, they give UPAVE, an unbiased estimator of $v(P,h)=2\pi(h_0g)-\pi(h_0^2)$. UPAVE converges at the Monte Carlo rate across replicates, unlike batch means, which converge at $T^{-2/3}$.

Variance reduction options:
- the lag $L$ (largest practical gain, per the review and the rejoinder);
- Craiu–Meng control variates using the zero-mean differences $h(X_t)-h(Y_t)$, with gains that shrink as $L$ and $m$ grow;
- Sherlock's Rao–Blackwellisation for coupled MH (discussion only);
- gradient control variates (South–Nemeth–Oates, preliminary);
- atom subsampling, which reduces test-function cost rather than variance.

### Cited Findings
- Fishy functions (Douc et al.).
  - $g_\star:=\sum_{t\ge0}P^th_0$ with $h_0=h-\pi(h)$ solves $(I-P)g=h_0$.
  - $v(P,h)=\mathbb E_\pi[\{g(X_1)-Pg(X_0)\}^2]=2\pi(h_0g)-\pi(h_0^2)$ (eq. 1.3), which equals $\mathrm{var}_\pi(h)+2\sum_{t\ge1}\mathrm{cov}_\pi(h(X_0),h(X_t))$ (eq. 1.5).
  - $g_y(x)=\sum_{t\ge0}(P^th(x)-P^th(y))$, estimated by $G_y(x)=\sum_{t=0}^{\tau-1}(h(X_t)-h(Y_t))$ with $(X_0,Y_0)=(x,y)$ and $\tau=\inf\{t\ge0:X_t=Y_t\}$.
  - Theorem 6: if $h\in L^m(\pi)$ with $m>\kappa/(\kappa-1)$, then $\mathbb E G_y(x)=g_y(x)$ for $\pi\otimes\pi$-almost all $(x,y)$, and $\mathbb E|G_y(x)|^p<\infty$ for $1/p>1/m+1/\kappa$. The cost is about $2\tau$ — [Douc et al.](https://arxiv.org/abs/2206.05691)
- Douc et al. re-derive unbiased MCMC from the identity $\pi(h)=h(x)+Pg(x)-g(x)$: $h(x)+G_x(X_1)$ with $X_1\sim P(x,\cdot)$ is $H_0$ with $\pi_0=\delta_x$ and $L=1$ — [Douc et al.](https://arxiv.org/abs/2206.05691); [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- EPAVE (eq. 4.2) is consistent rather than unbiased: $\hat v_E(P,h)=-v^{MC}(h)+\frac2t\sum_{s=0}^{t-1}(h(X_s)-\pi^{MC}(h))G(X_s)$.
  - The $G(X_s)$ are conditionally independent given the chain, and can be computed only every $D$ steps with $D\approx\lceil C\rceil$.
  - Theorem 22: it is strongly consistent, and satisfies a $\sqrt t$-CLT if $m>4\kappa/(\kappa-3)$ — [Douc et al.](https://arxiv.org/abs/2206.05691)
- UPAVE (Algorithm 4.1, eq. 4.4). Take two independent signed measures $\hat\pi^{(j)}=\sum_n\omega^{(j)}_n\delta_{Z^{(j)}_n}$, $j=1,2$. For each $j$, draw $I^{(j)}_r$ for $r=1..R$ with probabilities $\xi^{(j)}$. Then

  $$\hat v(P,h)=-\hat v(\pi,h)+\frac1R\sum_{i\ne j\in\{1,2\}}\sum_{r=1}^R\frac{\omega^{(j)}_{I^{(j)}_r}}{\xi^{(j)}_{I^{(j)}_r}}\big(h(Z^{(j)}_{I^{(j)}_r})-\hat\pi^{(i)}(h)\big)G\big(Z^{(j)}_{I^{(j)}_r}\big),$$

  with $\hat v(\pi,h)$ from eq. (4.3).
  - Theorem 23: if $h\in L^m(\pi)$ with $m>2\kappa/(\kappa-2)$, $d\pi_0/d\pi\le M$ and uniform $\xi$, then UPAVE is unbiased with $\mathbb E|\hat v|^p<\infty$ for $1/p>2/m+2/\kappa$. When $\tau$ has all moments, slightly more than 2 moments of $h$ give consistency and slightly more than 4 give $O(1/M)$ variance over $M$ averaged copies; the same holds with the roles of $h$ and $\tau$ swapped — [Douc et al.](https://arxiv.org/abs/2206.05691)
- UPAVE tuning (Douc et al. §4.4):
  - $y$ should be "central with respect to π".
  - Optimal selection probabilities $\xi^\star_n\propto\sqrt{\alpha_n}$ with $\alpha_n=\{\omega_n(h(Z_n)-\hat\pi(h))\}^2\mathbb E[G(Z_n)^2\mid Z_n]$ (eq. 4.6).
  - Choose $R$ so that the cost of the two signed measures matches the cost of the $2R$ estimates of $G$.
  - Reservoir sampling avoids storing all atoms.
  - Multivariate entries are $-(\pi(h_ih_j)-\pi(h_i)\pi(h_j))+\pi[(h_i-\pi h_i)g_j+g_i(h_j-\pi h_j)]$ (eq. 4.9) — [Douc et al.](https://arxiv.org/abs/2206.05691)
- The review's simplest form is $\hat v=2(A)-(B)$ with $(A)=N\omega_IG(Z_I,y)(h(Z_I)-\hat\pi^{(2)}(h))$, $I$ uniform on $\hat\pi^{(1)}$, and $(B)$ from eq. (3.6). It contrasts this with batch means, which converge at $T^{-2/3}$ (Flegal & Jones 2010) — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Craiu & Meng (Statistica Sinica 32, 2022) [PR].
  - $\Delta_{k,j}=h(X_{k+jL})-h(Y_{k+jL})$ (same-time differences) has mean zero.
  - For any $\vec\eta$ independent of $(X,Y)$ with $\sum_j\mathbb E|\eta_j|<\infty$, $\tilde H^{(\eta)}_{k,L}=H_{k,L}-\sum_{j\ge1}\eta_j\Delta_{k,j}$ is unbiased.
  - Equivalent backward form: $H_{k,L}=h(X_{k+LJ})+\sum_{j=0}^{J-1}[h(X_{k+jL})-h(Y_{k+jL})]$ with $J=J_{k,L}=\max(0,\lceil(\tau_L-L-k)/L\rceil)$.
  - The optimal-bound choice is $\eta_j=\mathbb 1\{S_j>0.5\}$, the median of $\mathbb 1\{j\le\tilde J\}$ with $\tilde J=J-\mathrm{Bernoulli}(0.5)$. It gives the TV bound $B_{k,L}=\sum_{j\ge1}\min\{P(J\ge j),P(J\le j)\}\le\mathbb E[J]$.
  - The coefficients are estimated by leave-one-out medians over the other replicates (their Algorithm 1). Gains "diminish as L increases" and as the averaging length grows — [Craiu & Meng arXiv:2008.12662](https://arxiv.org/abs/2008.12662)
- The review frames control variates as $\sum_{t\ge0}\eta_t\{h(X_t)-h(Y_t)\}$ added to $H_{k:\ell}$. It also cites Neal & Pinto (2001), Andradóttir et al. (1993) and Alexopoulos et al. (2023) for coupling- and Poisson-based variance reduction — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- South, Nemeth & Oates [DISC] propose gradient-based control variates $g$ with $\pi(g)=0$, chosen to minimise $\sigma(h-g)$ through sample splitting or through an RKHS bound on $\sigma(h-g)$ for $k=m=0$. They report encouraging results in a 302-dimensional logistic regression — [arXiv:1912.10496](https://arxiv.org/abs/1912.10496)
- Other discussion proposals, via OUP HTML [DISC]: Rao–Blackwellisation over coupling events for MH (Sherlock); antithetic or negatively correlated pairs (Chak, Kantas & Pavliotis); reusing each pair's X-chain as the next pair's Y-chain, equivalent to one long chain with lag $RL$, whose unbiasedness was left open (Vanetti & Doucet) — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129); [Vanetti & Doucet PDF](https://www.stats.ox.ac.uk/~doucet/discussion_VanettiDoucet_unbiasedMCMC.pdf)

### Inferences
- [OWN] For `telescopic` v0.1 the cheapest gain is supporting general $L$ and defaulting to the $L=k$ recipe. Craiu–Meng control variates add little when $L$ is large, per the authors.
- [OWN] Their leave-one-out coefficients make replicates mutually dependent. Each replicate stays unbiased, but the naive $\hat\sigma/\sqrt R$ standard error is no longer exactly justified. Prefer coefficients from an independent pilot batch.
- [OWN] UPAVE fits well as an optional "efficiency report" (relative inefficiency $=\mathbb E[\text{cost}]\mathbb V[H]/v(P,h)$) because it needs no long runs. It does need a lag-0 coupled kernel run from arbitrary pairs $(x,y)$, a reference state $y$, and on the order of 4 moments for stable output.

### Gaps
- I could not access Sherlock's Rao–Blackwellisation formulas or quantitative results beyond the summary.
- I found no published variance-reduction method for these estimators beyond those listed.
- I could not confirm a journal version of Douc et al. (arXiv v5, Dec 2025, no journal-ref), so treat it as a preprint.

## 6. Budget-constrained parallel computation: bias of fixed-budget averaging (Heidelberger 1988; Glynn & Heidelberger 1990, 1991) and the recommended unbiased estimator and algorithm

### Takeaway
On one processor with budget $t$, averaging only the completed estimators (0 if none) has expectation $\mathbb E[H\mathbb 1(C\le t)]$, so its bias is $-\mathbb E[H\mathbb 1(C>t)]$.
- The fix is exact: use $\tilde H(t)=\bar H_{\max(1,N(t))}$, i.e. finish the first estimator even if it overruns, but never the one in progress once at least one has completed. This is exactly unbiased whenever $\mathbb E|H|<\infty$ (GH90 Corollary 7).
- "Always complete the run in progress and include it" ($\bar H_{N(t)+1}$) is biased at order $\mathrm{Cov}(H,C)/t$.
- With $P$ processors, average the per-processor $\tilde H_p(t)$ with equal weights. Confidence intervals come from CLTs in $P\to\infty$ or $t\to\infty$.
- Glynn–Heidelberger 1991 give a variant in which only $P_1$ preselected processors may overrun.

### Cited Findings
- Glynn & Heidelberger (1990), Operations Research 38(5):801–814 [PR].
  - Setup: i.i.d. pairs $(X_k,\tau_k)$, where $\tau_k$ is the cost of the $k$-th replicate (any cost measure, e.g. CPU dollars). $N(t)$ is the renewal count and $\bar X_0:=0$.
  - Eq. (1): $\mathbb E[\bar X_{N(t)}]=\mathbb E[X_1;\tau_1\le t]=\mathbb E[X_1]-\mathbb E[X_1;\tau_1>t]$.
  - Corollary 6: if $\mathbb E|X_1e^{\alpha\tau_1}|<\infty$ for some $\alpha>0$, then $\mathbb E[\bar X_{N(t)}]=\mathbb E X_1+o(e^{-\alpha t})$.
  - Corollary 7: $\tilde X_{N(t)}:=\bar X_{N(t)}$ if $N(t)>0$ and $X_1$ if $N(t)=0$ has $\mathbb E[\tilde X_{N(t)}]=\mathbb E[X_1]$ exactly if $\mathbb E|X_1|<\infty$.
  - "If one always completes the replication in progress at time t, then resulting estimate, $\bar X_{N(t)+1}$, has relatively high bias (of order 1/t)", with leading constant $c=\mathrm{Cov}[X_k,\tau_k]$ (Theorem 3).
  - Theorem 4: $\mathrm{Var}[\tilde X_{N(t)}]\le\mathrm{Var}$ of the unbiased adjustment of $\bar X_{N(t)+1}$ — [GH 1990 PDF](https://web.stanford.edu/~glynn/papers/1990/GH90.pdf)
- Glynn & Heidelberger (1991), ACM TOMACS 1(1):3–23 [PR]: "Unless the replication lengths are bounded, one must be willing to simulate beyond any fixed, finite time t on at least some processors in order to always obtain a strongly consistent estimator (as the number of processors increases)." The estimators are:
  - $\bar\mu(P,t)$: the average of per-processor $\bar X_i(N_i(t))$. It is biased, need not converge as $P\to\infty$ at fixed $t$, and does converge as $t\to\infty$ (Proposition 3.1).
  - $\bar\mu_1(P,t)$, eq. (3.2): the average of per-processor $\tilde X_i$. It is unbiased and requires "completing at least one replication on each processor". Specifically, "one must complete the replication in progress on processor i if and only if $N_i(t)=0$" (Proposition 3.2).
  - $\bar\mu_2(P,P_1,t)$, eq. (3.3): unbiased, with a shorter completion time. It "estimate[s] the remainder term $E[X_{i1}\mathbb 1(\tau_{i1}>t)]$" using only $P_1$ preselected processors, and reduces to $\bar\mu_1$ when $P_1=P$ (Proposition 3.3).
  - "The P1 processors must be preselected: an unbiased estimate would not result, for instance, by taking the first P1 uncompleted replications that actually do complete."
  - "A sensible, practical approach is to preselect a fixed fraction a of the processors" — [GH 1991 PDF](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- Jacob et al. §3.3, the paper's adaptation [PR].
  - $\bar H_p(t)$ is 0 if $N_p(t)=0$ and the average of completed estimators otherwise. It is biased: $\mathbb E[\bar H_p(t)]=\mathbb E[H]-\mathbb E[H\mathbb 1(C>t)]$.
  - $\tilde H_p(t)$ uses $\tilde N_p(t)=\max(1,N_p(t))$ and "requires the completion of $H_{p1}$". $\tilde H(P,t)=P^{-1}\sum_p\tilde H_p(t)$ is unbiased if $\mathbb E|H|<\infty$.
  - Consistency holds as $t\to\infty$ or as $P\to\infty$. If $\mathbb E|H|^{1+\delta}<\infty$ and $t_P\to\infty$, then $\tilde H(P,t_P)\to\pi(h)$ in probability.
  - With $\sigma_1^2$ the sample variance of the per-processor $\tilde H_p(t)$, and

    $$\tilde\tau(P,t)=\frac1P\sum_{p=1}^P\frac{1}{\tilde N_p(t)}\sum_{n=1}^{\tilde N_p(t)}C_{pn},\qquad \hat\sigma_2^2(P,t)=\tilde\tau(P,t)\Big(\frac1P\sum_{p=1}^P\frac1{\tilde N_p(t)}\sum_{n=1}^{\tilde N_p(t)}H_{pn}^2-\tilde H(P,t)^2\Big),$$

    there are three CLTs:
    - (3.4) $\sqrt P(\tilde H(P,t)-\pi h)/\hat\sigma_1\to\mathcal N(0,1)$ for fixed $t$ and $P\to\infty$;
    - (3.5) $\sqrt{Pt}(\tilde H-\pi h)/\hat\sigma_2\to\mathcal N(0,1)$ for fixed $P$ and $t\to\infty$;
    - (3.6) the same as (3.5) with $t_P\to\infty$ as $P\to\infty$.
    - "These results require moment conditions such as $E[\tilde H_p(t)^2]<\infty$."
  - At fixed $t$ with $P\to\infty$, the completion time behaves like the maximum of $P$ costs, and under geometric tails its expectation grows like $\log P$ — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Atchadé–Jacob: "Careful: running unbiased MCMC on M ≫ C machines and retaining the C estimators that are first completed would introduce a bias, since the estimator is not independent of its cost." "If each machine is tasked to produce a single estimator, the total time is exactly the (random) cost of the longest run", which grows like $\log(C)$ under geometric tails — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Heidelberger (1988, SIAM J. Sci. Stat. Comput. 9) is cited for the distinction between "fixing the number of replications and letting completion time be random" and "fixing completion time and letting the number of replications be random". In the latter case, completed replications "may not be i.i.d. with the desired distribution" — [summary of a citing paper, arXiv:1506.04986](https://arxiv.org/pdf/1506.04986)

### Inferences
- [OWN] Why Corollary 7 holds: an exchangeability argument. The event $\{N(t)=n\}=\{S_n\le t<S_{n+1}\}$ is symmetric in the first $n$ pairs, so $\mathbb E[\bar H_n\mathbb 1(N(t)=n)]=\mathbb E[H_1\mathbb 1(N(t)=n)]$. Summing over $n\ge1$ and adding $\mathbb E[H_1\mathbb 1(N(t)=0)]$ gives $\mathbb E H_1$. The same argument shows why pooling all completed estimators across processors weighted by count, $\sum_p\sum_iH_{pi}/\sum_pN_p$, is not covered: it is a ratio estimator that over-weights processors with cheap runs, and $H$ and $C$ are correlated through $\tau$. Use equal weights per processor.
- [OWN] Recommended `telescopic` budget algorithm, unbiased for any budget $t$ (GH90 Corollary 7 and GH91 $\bar\mu_1$, as adapted by Jacob et al.):
  1. Fix $(k,m,L)$ and the coupling in advance, from an independent pilot. Give each of $P$ workers an independent RNG stream.
  2. Each worker loops:
     - start a replicate and run it to completion, recording $(H,C)$;
     - if the cumulative cost is at most $t$, accept it and continue;
     - otherwise, if the worker has 0 accepted replicates, finish this replicate and accept it (overrun allowed); if it has at least 1, abort the replicate and discard it, which is safe because its value is never used.
  3. Worker output: $\tilde H_p$, the mean of its accepted replicates, and $\tilde N_p=\max(1,N_p)$.
  4. Estimate: $\tilde H=P^{-1}\sum_p\tilde H_p$.
  5. SE: $\hat\sigma_1/\sqrt P$ for many workers, or $\hat\sigma_2/\sqrt{Pt}$ for few workers with long $t$.
  6. Variant (GH91 $\bar\mu_2$): preselect $P_1=aP$ workers that follow the overrun rule; the others stop hard at $t$. The estimate is $P^{-1}\sum_p\bar H_p(t)+P_1^{-1}\sum_{i\le P_1}H_{i1}\mathbb 1\{N_i(t)=0\}$, where $\bar H_p(t)=0$ if $N_p(t)=0$. This form is reconstructed from GH91's prose, because the displays (3.2)–(3.3) were images in the PDF; it reproduces $\bar\mu_1$ at $P_1=P$, as GH91 state.
- [OWN] Fixed-$R$ mode, the default for an emcee-like API: launch exactly $R$ replicates and wait for all of them. The average is unbiased. Never stop at the first $R$ completions out of more than $R$ launched (review warning).
- [OWN] With cheap replicates relative to $t$, $\bar H_p(t)$ (no overrun) has exponentially small bias under GH90 Corollary 6 when $\mathbb E[H^2]<\infty$ and $\mathbb E[e^{2\alpha C}]<\infty$, which geometric tails of $\tau$ give. With heavy-tailed $\tau$, only the overrun rule is safe.

### Gaps
- I could not access Heidelberger (1988) itself; even its page range is uncertain.
- GH91's displayed formulas (3.2)–(3.3) were images and are reconstructed from the text above.
- I found no source specifically on the asynchronous, work-stealing schedulers that a Python implementation would use. Their unbiasedness follows only if per-worker sequences remain i.i.d. and the stopping rule depends only on each worker's own costs, which is my inference.

## 7. Relation to perfect simulation (CFTP), multilevel Monte Carlo (Vihola 2018) and unbiased MCMC for intractable targets (Middleton et al. 2020)

### Takeaway
Unbiased MCMC sits between ordinary MCMC and perfect sampling ("exact estimation" versus "exact simulation"):
- It needs only two forward-coupled chains that meet; CFTP needs coalescence from all states, or monotonicity, run backwards.
- It produces unbiased signed measures, not exact draws.
- It is the coupled-sum randomized-truncation estimator with $N=\infty$, which exact meeting makes possible. So it shares MLMC's cost–variance trade-offs (Vihola), and stratified or residual allocation schemes from that literature are asymptotically as efficient as MLMC.
- Middleton et al. extend the validity theory to polynomially ergodic samplers for intractable likelihoods, such as pseudo-marginal and exchange algorithms.

### Cited Findings
- Glynn & Rhee argue that "exact simulation is inherently restrictive". Exact simulation algorithms "can typically be constructed only for Markov chains that are φ-irreducible" and require simulating from multiple initial states, whereas exact estimation works even for some non-φ-irreducible contractive chains — [Glynn & Rhee](https://arxiv.org/abs/1409.4302)
- Jacob et al.: "only two chains have to be coupled for the proposed estimator to be unbiased, without further assumptions on the state space or target distribution. Thus the approach applies more broadly than perfect samplers [see Glynn, 2016] while yielding unbiased estimators rather than exact samples" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- Atchadé–Jacob: the requirement "is weaker than that of Coupling From The Past (Propp and Wilson, 1996) … but it does not provide perfect samples". "There exists a world, of a size to be determined, between standard MCMC and perfect sampling, where unbiased estimators can be obtained but not exact samples (Glynn, 2016)." Successful couplings with $|\pi_t-\pi_0|_{TV}=P(\tau>t)$ exist (Pitman 1976) but "may not often be implementable" — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- Craiu & Meng call L-lag coupling "a coupling for the future, breaking from the coupling-from-the-past type of perfect sampling, by reducing the generally unachievable requirement of being perfect to one of being unbiased" — [Craiu & Meng](https://arxiv.org/abs/2008.12662)
- JRSSB, via OUP HTML. The reply: "It is unclear whether the machinery of this paper bring us any closer to perfect sampling". Wang, Pollock, Roberts & Steinsaltz [DISC] proposed regenerative "Restore" samplers combined with CFTP for exact draws. Vats & Jones [DISC] noted that one exact draw (accept–reject, possibly via linchpin variables) used as a starting value removes burn-in bias — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)
- Vihola (2018) [PR] presents "a new general class of unbiased estimators, which admits previous debiasing schemes as special cases" (Theorem 7: $Z=\sum_i\frac{1}{\mathbb EN_i}\sum_{j=1}^{N_i}\Delta^{(j)}_i$ with random level counts $N_i$).
  - "Averaging n independent single term estimators" is MLMC with random $N_i$ (Giles 2015).
  - Stratified (Definition 13), systematic (14) and residual (15) sampling estimators have variance at most $n^{-1}$ times that of single-term or sum estimators (Proposition 16).
  - Under general conditions these schemes are "asymptotically as efficient as MLMC, both in terms of variance and cost" (Theorem 19, Proposition 22, Corollary 23) — [Vihola](https://arxiv.org/abs/1512.01022)
- Jasra, Law & Heng [DISC] proposed combining the approach with single-term and coupled-sum estimators across MLMC levels, using four-way couplings of MH proposals — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129). Wang & Wang (JMLR 2023) combine unbiased MCMC with unbiased MLMC — [arXiv:2204.04808](https://arxiv.org/abs/2204.04808)
- Middleton, Deligiannidis, Doucet & Jacob (EJS 2020, 14(2):2842–2891) [PR] treat targets whose density cannot be evaluated pointwise, such as pseudo-marginal and exchange algorithms. They "establish the theoretical validity of some of these procedures, by extending existing results to cover the case of polynomially ergodic Markov chains". Polynomial tails follow from polynomial drift conditions (Theorem 2, $\kappa=1/(1-\alpha)$), and they verify those conditions for coupled pseudo-marginal kernels. The examples include state-space and Ising models — [Middleton et al.](https://arxiv.org/abs/1807.08691)
- Further cross-links:
  - Agapiou, Roberts & Vollmer (2018, Bernoulli) used contractive couplings with truncation variables for MCMC.
  - Jacob, Lindsten & Schön (2020, JASA) found exactly-meeting couplings of conditional particle filters.
  - Middleton et al. (2019, AISTATS) coupled particle independent MH, "with the curious implication that the bias of self-normalized importance sampling estimators can be removed in finite time".
  - — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)

### Inferences
- [OWN] In the notation of section 1, Jacob et al.'s $H_k$ is Glynn–Rhee's coupled sum with $Y_n\to h(X_{k+n})$, $\Delta_n=h(X_{k+n})-h(Y_{k+n-1})$ and $P(N\ge n)\equiv1$. Exact meeting makes the infinite sum finite, so no truncation distribution needs tuning. A randomized-truncation fallback (Glynn–Rhee contraction setting) is only needed for couplings that contract but never meet exactly; that is outside "published methods with exact meeting" and probably out of scope for v0.1.
- [OWN] Vihola's stratification results suggest that when several unbiased replicates are generated anyway, deterministic or stratified allocation of the truncation levels beats i.i.d. randomisation. For exactly-meeting couplings there is no truncation variable to stratify, so the analogous lever is the choice of $(k,m,L)$.

### Gaps
- I did not read the Propp & Wilson (1996) and Glynn (2016, WSC) originals; the relationships come from the papers above.
- I did not read the details of Middleton et al.'s coupled pseudo-marginal kernels (Proposition 2 conditions); that belongs to couplings research.

## 8. What the "finite-time guarantees" are and are not; limitations

### Takeaway
What is guaranteed:
- For any fixed $k\le m$, $L\ge1$ and $\pi_0$ meeting the theorem's conditions, $\mathbb E[H_{k:m}]=\pi(h)$ exactly.
- Variance and expected cost are finite under tail and moment conditions.
- Averages over $R$ independent replicates give CIs that are valid asymptotically in $R$, not in run length, with bounded work per replicate.
- A fixed-time-budget version is exactly unbiased with the max(1, N) rule.
- Meeting-time samples give estimated upper bounds on the TV and $W_1$ distances from $\pi_k$ to π.

What is not guaranteed:
- small error at finite $R$;
- correctness of the CI at finite $R$ (multimodal or heavy-tailed cases can fool both the CI and the empirical meeting-time distribution);
- better mixing (coupling does not improve the marginal chain);
- non-negative or in-range estimates;
- exact samples;
- robustness to a poor $\pi_0$.

### Cited Findings
- Unbiasedness for all $k\le m$ (Jacob et al. Proposition 3.1; Douc et al. Theorem 17, for any $L$): "Unbiasedness is guaranteed for any choice of k ≥ 0, but both the cost and variance of Hk(X, Y) are sensitive to k" — [Jacob et al.](https://arxiv.org/abs/1708.03625); [Douc et al.](https://arxiv.org/abs/2206.05691)
- CIs "can be constructed with asymptotic guarantees in the number of processors, in contrast with standard MCMC confidence intervals that are justified asymptotically in the number of iterations" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- TV bound with lag 1 (Jacob et al. §6): $d_{TV}(\pi_k,\pi)=\frac12\sup_{|h|\le1}|\mathbb E[\sum_{t=k+1}^{\tau-1}(h(X_t)-h(Y_{t-1}))]|$, hence $d_{TV}(\pi_k,\pi)\le\min(1,\mathbb E[\max(0,\tau-k-1)])$ — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- TV bound with lag $L$ (Biswas et al. Theorem 2.5; review eq. 3.2): $|\pi_k-\pi|_{TV}\le\mathbb E[\max(0,\lceil(\tau^{(L)}-L-k)/L\rceil)]$, estimated by $C^{-1}\sum_c\max(0,\lceil(\tau_c-L-k)/L\rceil)$ from $C$ meeting times.
  - The $W_1$ bound (review eq. 3.4) is $|\pi_k-\pi|_{W_1}\le\mathbb E[\sum_{j=1}^{\lfloor(\tau-k-1)/L\rfloor}|X_{k+jL}-Y_{k+(j-1)L}|]$.
  - Craiu–Meng's bound $B_{k,L}\le\mathbb E[J_{k,L}]$ is never worse. For geometric $\tau-L$ with $P(\tau-L\ge g)=q^g$, $\mathbb E[J_{k,L}]=q^{k+1}/(1-q^L)$ — [Biswas et al.](https://arxiv.org/abs/1905.09971); [Atchadé & Jacob](https://arxiv.org/abs/2406.06851); [Craiu & Meng](https://arxiv.org/abs/2008.12662)
- Limits stated by the authors:
  - "coupling pairs of Markov chains does not improve their marginal mixing properties, and … poor mixing of the underlying chains can lead to poor performance";
  - "The choice of initial distribution π0 can have undesirable effects on the estimators";
  - "Overall the choice of π0 and its relative importance compared to standard MCMC are open questions";
  - couplings "might result in prohibitively large meeting times, either because the marginal chains mix slowly … or because the coupling strategy is ineffective" — [Jacob et al.](https://arxiv.org/abs/1708.03625)
- "Both the computing time and the variance can be prohibitively high" (review §1.2). The TV-bound estimates can be unreliable under multimodality with a concentrated $\pi_0$ (review §3.1, "Limitation"). Unbiased estimation for adaptive MCMC "remains elusive" (review §5.2) — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)
- The reply notes that "using different initial distributions for different runs result in estimators that are unbiased but not identically distributed" (via OUP HTML). Robert et al. [DISC] cautioned that unbiasedness is not the ultimate goal of MCMC — [JRSSB page](https://academic.oup.com/jrsssb/article/82/3/543/7056129)
- Estimates may fall outside the range of $h$ and negative weights cannot be removed (Jacob et al.; review, citing Jacob & Thiery 2015) — [Atchadé & Jacob](https://arxiv.org/abs/2406.06851)

### Inferences
- [OWN] Suggested wording for user-facing guarantees:
  - "Each replicate is an exactly unbiased estimate of $\mathbb E_\pi[h]$ for the stated $(k,m,L)$, provided the coupling meets and stays together and the meeting time has enough moments. The reported interval is a large-$R$ CLT interval and is valid only if the variance is finite."
  - "Unbiased does not mean accurate. A multimodal target with a narrow $\pi_0$ can yield confidently wrong intervals for any practical $R$."
  - "TV and $W_1$ bounds are Monte Carlo estimates of upper bounds, subject to the same caveat."
- [OWN] Pair the guarantees with default diagnostics:
  - the meeting-time distribution and its tail;
  - fraction of replicates with $\tau>k+L$, i.e. with a non-zero correction;
  - the share of negative weight;
  - relative inefficiency;
  - batch-stability checks;
  - a recommendation to run pilots from several overdispersed $\pi_0$.

### Gaps
- I did not access Pierre Jacob's lecture notes ("Couplings and Monte Carlo", sites.google.com/site/pierrejacob/cmclectures), the Statisfaction blog or Christian Robert's blog posts. Their content may add pedagogy but, as far as I know, no new estimators.
- No formal finite-$R$ coverage guarantee exists in the sources. All CI validity is asymptotic in $R$, or in $P$ and $t$ for budgeted runs.
