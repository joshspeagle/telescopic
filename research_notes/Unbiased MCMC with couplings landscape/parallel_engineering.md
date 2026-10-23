# Engineering the execution layer for coupled-chain replicates in Python: budgets, parallelism, random-number streams, memory

Scope: how to run many independent coupled-chain replicates (Jacob, O'Leary & Atchadé 2020) efficiently, reproducibly and without bias in a pure-Python (NumPy/SciPy, optional JAX) package (`telescopic`). Facts are as of 2026-10-23. "Own experiment" findings were measured for this note in a 4-vCPU Intel Xeon @ 2.10 GHz VM with Python 3.13.16, NumPy 2.5.3 and OpenBLAS 0.3.34 (the scipy-openblas build bundled in the NumPy wheel). The scripts and raw outputs are in `/tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/`; that directory is session scratch, so copy it if it should be kept. The toy used for the bias experiments is a coupled random-walk Metropolis sampler with a reflection-maximal coupling and lag 1, on target N(0,1), started from N(3,1), with proposal sd 1, h(x)=x and (k,m)=(2,10). It is exactly unbiased: the mean over 2×10⁶ replicates was −0.0056 with sd(H)=14.4. Its cost C=max(τ,m) has E[C]=11.05, P(C>m)=0.178, max 72 and corr(H,C)=−0.195.

Package versions on PyPI on 2026-10-23 (queried through the PyPI JSON API; pages at `https://pypi.org/project/<name>/`):

| package | version (date) |
|---|---|
| numpy | 2.5.4 (2026-10-10) |
| scipy | 1.18.1 |
| jax | 0.12.0.1 (2026-10-21) |
| numba | 0.68.0 |
| joblib | 1.6.0 |
| loky | 3.7.0 |
| threadpoolctl | 3.7.0 |
| dask / distributed | 2026.8.0 |
| ray | 2.59.0 |
| mpi4py | 4.1.2 |
| schwimmbad | 0.4.2 (2024-03-14) |
| emcee | 3.1.6 (2024-04-19) |
| dynesty | 3.1.0 (2026-07-17) |
| blackjax | 1.7.1 |
| numpyro | 0.22.0 |
| tqdm | 4.70.1 |
| h5py | 3.16.0 |
| zarr | 3.4.1 |
| cupy-cuda12x | 14.2.0 |
| arviz | 1.3.0 |

## Budget-constrained estimation: Glynn & Heidelberger theory, exact estimators, Jacob et al.'s use, algorithms

### Takeaway
Under a time budget, averaging "whatever has finished" is biased, and pooling across processors makes it worse: the pooled estimator converges to the wrong value as the number of processors grows with t fixed. Exact unbiasedness is restored by working per processor. Average each processor's completed replicates; if a processor has completed none, wait for its first replicate. Then average these per-processor values with equal weights. This is the estimator H̃(P,t) that Jacob, O'Leary & Atchadé (2020) adopt. It and two alternatives were reproduced numerically here, and every policy that selects replicates by completion was confirmed to be biased.

### Cited Findings
- **Setting and natural estimator (single processor).** Let the budget be t, let N(t) be the number of replications completed by t (a renewal process), and let the replication outputs and costs (X_i, τ_i) be i.i.d. GH90 derive an exact expression for the sample mean of the N(t) completed replications and a similar one for the mean of N(t)+1 replications. The sample mean over N(t) has bias "at most of order o(1/t)", decreasing "much faster" under moment conditions. The mean over N(t)+1 replications, i.e. completing the one in progress, "has bias of order 1/t". The exact expressions "lead to simple and totally unbiased estimators" — [Glynn & Heidelberger 1990, Oper. Res. 38:801–814 (PDF)](https://web.stanford.edu/~glynn/papers/1990/GH90.pdf); [INFORMS DOI](https://pubsonline.informs.org/doi/10.1287/opre.38.5.801)
- **Exact identity behind this.** Given N(t)=k, the first k outputs are exchangeable. Define X̄_{N(t)} to be 0 when N(t)=0. Then E[X̄_{N(t)}] = E[X₁; τ₁ ≤ t] = E[X₁] − E[X₁; τ₁ > t], so X̄_{max(N(t),1)} is exactly unbiased (Theorem 1 and Corollary 7). If E[|X₁|e^{ατ₁}] < ∞ for some α > 0, the bias of X̄_{N(t)} is o(e^{−αt}) (Corollary 6). GH90 also give Tin-style bias-adjusted estimators for nonlinear functions of means — [GH 1990](https://web.stanford.edu/~glynn/papers/1990/GH90.pdf); [GH 1991 §2](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Parallel processors (Heidelberger 1988, as summarised in GH91).** Heidelberger [SIAM J. Sci. Stat. Comput. 9 (1988) 1114–1132] "showed that some of the most obvious estimates obtainable from parallel replication schemes are guaranteed to produce incorrect results ... converge to the wrong quantity with probability one as P ... increases". Estimators that let incomplete replications finish fix this, at a completion-time penalty that grows like ln(P) — [Glynn & Heidelberger 1991, ACM TOMACS 1(1):3–23 (PDF)](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Pooled estimators.** μ̄₁(P,t) averages all observations completed by t across processors. It satisfies lim_{t→∞} μ̄₁ = μ, but lim_{P→∞} μ̄₁ = μ + O(1/t) ≠ μ. Completing every replication in progress and pooling (μ̄₂) converges to μ as P → ∞ or t → ∞, because N_i(t)+1 is a stopping time and Wald's equation applies to the numerator and the denominator — [GH 1991 §2](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Impossibility result.** No estimator that uses only information from [0,t] can be universally consistent as P → ∞. One "must be willing to simulate beyond time t on at least some processors" — [GH 1991 §2](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Three per-processor estimators (GH91 §3).**
  - μ̄(P,t) averages the per-processor natural averages, with 0 for empty processors. Its expectation is E[X; τ ≤ t]: consistent as t → ∞, but not as P → ∞ with t fixed.
  - μ̃₁(P,t) averages the per-processor X̄_i(max(N_i(t),1)). It is exactly unbiased. It requires completing processor i's in-progress replication "if and only if N_i(t)=0".
  - μ̃₂(P,P₁,t) = μ̄(P,t) + (1/P₁)·Σ_{i≤P₁} X_{i1}·1{τ_{i1} > t}. It is also unbiased, and only P₁ processors chosen in advance may run past t. The paper warns that "the P₁ processors must be preselected: an unbiased estimate would not result, for instance, by taking the first P₁ uncompleted replications that actually do complete."
  - Source: [GH 1991 §3](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **How fast P may grow with t when the biased per-processor estimator μ̄ is used.** Its CLT is correctly centred only if P grows slowly enough: P = O(t^{2k−1}) when E|X|τ^k < ∞, and "under these base assumptions, we require P = O(t³)". With exponential moments the condition is P = O(e^{2θt}/t). For μ̃₂, the variance inflation from preselecting a fraction α = P₁/P "disappears asymptotically" — [GH 1991 Props 3.6–3.7](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Completion time (GH91 §5).** Completion time is max(t, M_{P₁}), where M_{P₁} is the maximum first-replication time over the preselected processors. For exponential replication times, E[M_n] = H_n ≈ ln n — [GH 1991 §5](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **Earlier single-processor result.** For regenerative ratio estimation, completing the cycle in progress reduces bias from order 1/t to 1/t² (Meketon & Heidelberger 1982, cited in GH91) — [GH 1991 §1](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf)
- **How Jacob, O'Leary & Atchadé (2020) apply this (§3.3, "Parallel implementation under budget constraints").**
  - They set H̃_p(t) = H̄_p(t) if N_p(t) > 0 and H̃_p(t) = H_{p1} otherwise, and use H̃(P,t) = P⁻¹Σ_p H̃_p(t), which is "unbiased for π(h)". The natural H̄_p(t) is biased: E[H̄_p(t)] = E[H] − E[H·1(C>t)].
  - They state three CLTs: (3.4) fixed t, P → ∞, using σ̂₁² = sample variance of the H̃_p across processors; (3.5) fixed P, t → ∞; (3.6) t_P → ∞, using σ̂₂², which involves τ̃(P,t), the mean per-processor cost.
  - Under geometric meeting-time tails, "the completion time for H̃(P,t) has expectation behaving as log P".
  - Source: [Jacob, O'Leary & Atchadé 2020, arXiv:1708.03625 §3.3](https://arxiv.org/abs/1708.03625)
- **Their cluster run.** 500 processors with a 30-minute budget; each processor produced 1–7 estimators (mean 3.7, total 1858). Standard errors were σ̂₁(P,t)/√P, from CLT (3.4) — [Jacob et al. 2020 §5.3](https://arxiv.org/abs/1708.03625)
- **Efficiency measure.** Jacob et al. measure efficiency by the asymptotic inefficiency, variance × expected cost (Glynn & Whitt 1992). This is the asymptotic variance as the *computational budget*, rather than R, goes to infinity. They note a trade-off "with the desired level of parallelism: one might prefer to keep k and m small" — [Jacob et al. 2020 §3.1](https://arxiv.org/abs/1708.03625)
- **Atchadé & Jacob (2024 Handbook chapter).**
  - "Careful: running unbiased MCMC on M ≫ C machines and retaining the C estimators that are first completed would introduce a bias, since the estimator is not independent of its cost."
  - When M ≪ C, the speed-up is close to linear. With one estimator per machine, the total time is the cost of the longest run, which grows like log(C) for geometric tails.
  - The cost is cost(H_{k:ℓ}) = L + 2(τ−L) + max(0, ℓ−τ).
  - Source: [Atchadé & Jacob 2024, arXiv:2406.06851 §2.5](https://arxiv.org/abs/2406.06851)
- **Own experiment: policy biases (Monte Carlo, truth = mean of the replicate pool).**
  - Setup: B lanes, each running back-to-back replicates; budget T in lockstep iterations.
  - Pooled "all completed by T":
    - B=64: bias +0.669 (z=+165) at T=1.4·E[C]; +0.412 (z=+102) at 2.7·E[C]; +0.099 (z=+25) at 9·E[C].
    - B=1024 gives the same biases (+0.671, +0.412). This is GH's "converges to the wrong value as P → ∞".
  - Per-lane natural average, dropping empty lanes or counting them as 0: +0.67 / +0.62 at T=1.4·E[C], +0.07 (z≈10) at 2.7·E[C], undetectable at 9·E[C]. This is the exponentially decaying E[H; C>T] bias.
  - "First R completed" with refill: +0.217 (z=58) for B=64, R=256; +0.645 (z=221) for B=R=1024.
  - No detectable bias (|z| ≤ 1.3 in every configuration) for: fixed R chosen by index, per-lane GH (μ̃₁ = Jacob et al.'s H̃), GH91 preselect with P₁=B/4, and pooled-with-completion (μ̄₂).
  - Source: [own experiment exp1_policies.py / exp1_policies.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp1_policies.out)
- **Own experiment: trade-offs among the unbiased or consistent budget estimators.**
  - Mean makespan:
    - B=64, T=15: per-lane GH 25.9 vs pooled-with-completion 36.5.
    - B=1024, T=15: 38.7 vs 49.2.
    - B=1024, T=30: 38.8 vs 67.3.
  - Standard deviation: per-lane GH is 1.15–1.7× larger than pooled-with-completion, because the latter also uses the finished in-progress replicates on every lane. Ratios: 1.40 (T=15), 1.67 (T=30), 1.15 (T=100) at B=64. GH91 preselect with P₁=B/4 had about 2.6× the pooled-with-completion sd at T=15 (1.9× the per-lane GH sd).
  - Source: [own experiment](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp1_policies.out)

### Inferences
- **Implementation algorithms for `telescopic` (derived from GH91 and Jacob et al. §3.3).**
  1. **Fixed-R mode (default).** Assign indices 1..R up front and return the mean of exactly those R replicates, whatever order they finish in. This is exactly unbiased and needs no lane bookkeeping. Replicates beyond R that were started are discarded or kept as warm state for a later extension. Including them only "if finished" reintroduces selection bias.
  2. **Budget mode with "unbiased" policy (default for budgets).**
     - Each lane or worker p records its ordered list of (index, H, C, t_start, t_end) on its own clock.
     - At the deadline t: lanes with N_p ≥ 1 cancel their in-progress replicate, and H̃_p = mean of the completed ones. Lanes with N_p = 0 run their first replicate to completion, and H̃_p = that H.
     - Estimate = mean over p of H̃_p, with SE = sd_p(H̃_p)/√P, from CLT (3.4).
  3. **Budget mode with "hard deadline" policy.** Preselect lanes 1..P₁ before starting. At t, cancel everything except empty preselected lanes. Return μ̃₂ = P⁻¹Σ_p X̄_p (with 0 for empty lanes) + P₁⁻¹Σ_{p≤P₁} X_{p1}·1{C_{p1}>t}.
  4. **Budget mode with "low-variance" policy.** Stop starting new replicates at t, run all in-progress ones to completion, and pool all of them. This is consistent (Wald) and has lower variance, but it is a ratio estimator rather than exactly unbiased, and it overruns the most.
- **What a "lane" is.** In a lockstep batch a lane is a slot, and lockstep iterations are the natural clock. A budget given in iterations makes budget runs deterministic. A wall-clock deadline is still valid if per-step cost does not depend on the states; then the step at which the deadline falls is independent of the replicate values.
- **Assignment of indices to lanes.** Dynamic assignment (the next free lane takes the next index) keeps each lane's sequence of (H, C) i.i.d. Each new replicate is a fresh draw, and its assignment depends only on the past. So the per-lane GH identity still holds. The estimate itself is not reproducible across runs, because the lane-to-index mapping depends on timing.
- **Process pools.** Per-worker bookkeeping (worker id plus timestamps) is needed. The cleanest design is P long-lived "lane" tasks, each looping over replicates until the deadline, rather than relying on executor internals for worker identity. `max_tasks_per_child` respawning would blur lanes.
- **Interrupts.** A run interrupted by Ctrl-C is a budget run with an unplanned t. "Completed so far" is the biased natural estimator, so it should be flagged as such rather than silently returned.

### Gaps
- Heidelberger (1988) itself was not read; its content is known here only through GH91's summary.
- GH91's equations (2.1), (2.2) and (3.1)–(3.3) are images in the PDF. The formulas above were reconstructed from GH91's text and from the proof of its Proposition 3.7, and cross-checked against Jacob et al. §3.3 and by simulation.
- No literature was found on budget rules for heterogeneous worker speeds or for costs that are dependent across lanes, such as shared-memory contention. The per-processor unbiasedness argument assumes an i.i.d. (H, C) sequence within a processor.

## Execution strategies for random-length jobs (lockstep NumPy batches, process pools, JAX vmap/while_loop, GPUs)

### Takeaway
Refilling lanes in a lockstep batch never biases anything by itself; only the rule that decides which replicates enter the average can. Vectorising lanes gives about a 30–100× throughput gain per core for cheap log-densities. Without refilling, a batch that runs until all of its pairs have met (as `vmap(while_loop)` does) wastes a factor E[max C]/E[C], about 2–11× at realistic batch sizes. Process pools add about 0.2 ms per task, plus pickling, and need chunking and an initializer for large data.

### Cited Findings
- **Lockstep throughput (own experiment, d=10 Gaussian target, reflection-maximal coupled random-walk Metropolis, one batched log-density per chain per step).**
  - Per pair-iteration: 38.5 µs at B=1; 4.4 µs at B=10; 0.74 µs at B=100; 0.37 µs at B=1000; 0.43 µs at B=10⁴; 0.60 µs at B=10⁵ (cache effects).
  - The same pair run alone in a Python loop over small NumPy arrays costs 11.7 µs per coupled iteration. A 1-D pure-scalar loop costs 0.7–1.2 µs.
  - Source: [own experiment exp3_overhead.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3_overhead.out); [exp3b_rng_steady.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3b_rng_steady.out)
- **Padding waste without refill (own experiment).** The waste factor E[max over B lanes of C]/E[C]:
  - Toy C=max(τ,m): 1.46 (B=8), 2.34 (B=64), 3.21 (B=512), 4.08 (B=4096), 4.95 (B=32768).
  - τ alone: 2.51 → 8.73 over the same range of B.
  - Geometric meeting times with mean 50: 2.70 (B=8), 4.70 (B=64), 6.75 (B=512), 8.87 (B=4096), 10.95 (B=32768).
  - Source: [own experiment exp2_padding.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp2_padding.out)
- **Lockstep with refill (own experiment).** Lane utilisation with a fixed R (no starts beyond R, so the batch drains at the end):
  - Toy cost, B=64: 0.45 / 0.74 / 0.92 / 0.97 for R/B = 1 / 4 / 16 / 64.
  - Toy cost, B=1024: 0.30 / 0.59 / 0.84 / 0.96.
  - Geometric(1/50), B=1024: 0.12 / 0.37 / 0.73 / 0.91.
  - Compacting the active set during the drain removes the arithmetic waste but not the fixed per-step overhead.
  - Source: [own experiment](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp2_padding.out)
- **Bias of refill and stopping rules.** See the previous section. Index-fixed R and the GH per-lane rules are unbiased with refilling. "First R completed" and "pooled completed by deadline" are biased — [own experiment](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp1_policies.out); [Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851)
- **JAX `while_loop` under `vmap`.**
  - With a batched predicate, the batching rule batches all of the carry: "If the predicate is batched, we have to batch *all* of the carry regardless of if the body needs it".
  - The lowering reduces the predicate with `lax.reduce_or`, so the loop continues while any lane is unfinished. It merges each lane's new carry with `_pred_bcast_select_hlo`, which freezes finished lanes while still computing them.
  - Source: [jax/_src/lax/control_flow/loops.py, `_while_loop_batching_rule` and `_while_lowering` (main branch, fetched 2026-10-23)](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py). Likewise, a batched `cond` becomes a `select` that evaluates both branches — [JAX issue #8409](https://github.com/google/jax/issues/8409)
- **BlackJAX on many chains.** Under `vmap`, NUTS chains take different numbers of integration steps and "each step is as long as the slowest chain". BlackJAX recommends `jax.shard_map` over a device mesh (which "replaces the now-deprecated `jax.pmap`") and per-chain keys from `jax.random.split` — [BlackJAX how-to: sample multiple chains](https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html)
- **NumPyro.** `chain_method` is 'parallel' (default, `pmap`; falls back to 'sequential' when there are fewer devices than chains), 'sequential', or 'vectorized'. 'vectorized' is "an experimental feature which vectorizes the drawing method" on a single device. Turning off the progress bar "will improve the speed for many cases" — [NumPyro MCMC docs](https://num.pyro.ai/en/stable/mcmc.html)
- **Accelerator autobatching of control flow.** Radul et al. (MLSys 2020) batch data-dependent control flow such as NUTS by tracking each batch member's program point. They advance only members at the same point and report orders-of-magnitude speedups on accelerators. The implementation is in TensorFlow Probability — [Radul et al. 2020, arXiv:1910.11141](https://arxiv.org/abs/1910.11141)
- **concurrent.futures (current docs, 3.15 series).**
  - `ProcessPoolExecutor(max_workers, mp_context, initializer, initargs, max_tasks_per_child)`. `max_workers` defaults to `os.process_cpu_count()` (since 3.13).
  - `as_completed` yields futures as they finish. `Executor.map(..., chunksize, buffersize)`; `buffersize` was added in 3.14 to bound in-flight tasks.
  - `shutdown(cancel_futures=True)` (3.9) cancels pending futures. `terminate_workers()` and `kill_workers()` were added in 3.14.
  - "The `__main__` module must be importable by worker subprocesses ... will not work in the interactive interpreter"; "A function defined in a REPL or a lambda should not be expected to work".
  - Source: [Python docs: concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
- **Python 3.14 (released 7 Oct 2025).**
  - "On Unix platforms other than macOS, 'forkserver' is now the default start method (replacing 'fork')". This applies to both multiprocessing and `ProcessPoolExecutor`.
  - `InterpreterPoolExecutor` runs each task in its own subinterpreter with its own GIL, pickling callables and arguments.
  - Free-threaded Python is officially supported (PEP 779), with a single-thread penalty of "roughly 5-10%".
  - Source: [What's New in Python 3.14](https://docs.python.org/3/whatsnew/3.14.html); [concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html)
- **Process-pool overheads (own experiment, 4 forkserver workers).** Trivial tasks cost 218 µs each (submit + as_completed), 199 µs each (`map`, chunksize=1) and 2.9 µs each (`map`, chunksize=64) — [own experiment exp4_pool.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp4_pool.out)
- **joblib.**
  - `Parallel(return_as='list'|'generator'|'generator_unordered')`; with the generator variants, "output order is not deterministic".
  - Arrays above `max_nbytes='1M'` are memory-mapped for the loky and multiprocessing backends.
  - `batch_size='auto'` adapts to task duration, with `pre_dispatch='2*n_jobs'`.
  - Source: [joblib.Parallel API](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html)
- **Dask.** "Every task comes with some overhead. This is somewhere between 200us and 1ms". Dask advises against embedding large client-side objects in task graphs — [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)
- **Ray.** The anti-pattern page's example takes 1.61 s for 10,000 one-number tasks, against 0.165 s serial and 0.030 s with batching. Its advice: "Avoid over-parallelizing" and use batching — [Ray docs: too fine-grained tasks](https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html)
- **mpi4py.futures.**
  - `MPIPoolExecutor` is a concrete `concurrent.futures.Executor` supporting `submit`/`map`/`as_completed`. `map(unordered=True)` yields results as tasks finish.
  - Workers come either from dynamic `Spawn` or from `mpiexec -n N python -m mpi4py.futures script.py`.
  - "only picklable objects can be executed and returned". `use_pkl5` (4.0) sends large buffers out-of-band, and `buffersize` was added in 4.1.
  - Source: [mpi4py.futures docs](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html)
- **schwimmbad.** Version 0.4.2; last commit 2025-03-02.
  - It provides `SerialPool`, `MultiPool`, `JoblibPool` and `MPIPool` with a common `.map(func, iterable, callback=...)`.
  - `MultiPool` builds on the dill-based `multiprocess` package (its dependencies are ["dill", "multiprocess"]). It makes workers ignore SIGINT and polls `r.get(timeout)` so that Ctrl-C reaches the parent, which calls `terminate()`.
  - `MPIPool.map` is a master–worker dynamic scheduler: each task is sent to the next free worker, and results are stored by task id.
  - Source: [schwimmbad source (multiprocessing.py, mpi.py, pyproject.toml)](https://github.com/adrn/schwimmbad)
- **emcee / dynesty `pool` objects.**
  - emcee accepts "An object with a `map` method that follows the same calling sequence as the built-in `map`". With `vectorize=True` the log-probability takes all walkers at once, "Note that `pool` will be ignored if this is `True`" — [emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)
  - dynesty takes `pool` plus `queue_size` (defaulting to the pool's `_processes` or `size`) — [dynesty utils.py `_parse_pool_queue`](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)
- **GPUs.**
  - JAX's Threefry and Philox implementations are marked identical across CPU/GPU/TPU and exact under `jax.vmap` over keys. The RBG variants generate a vmapped batch "from only the first key" — [jax.random docs](https://docs.jax.dev/en/latest/jax.random.html)
  - CuPy (cupy-cuda12x 14.2.0) is current on PyPI, but black-box NumPy log-densities cannot run on it unchanged — [PyPI cupy-cuda12x](https://pypi.org/project/cupy-cuda12x/)

### Inferences
- **(a) Lockstep NumPy batch.** This is the right primary engine when the user supplies a vectorised log-density with signature (n, d) → (n,), as with emcee's `vectorize=True`. It should:
  - make one batched call per step on the stacked proposals of all unmet X and Y chains, plus met chains, so 2·n_unmet + n_met rows;
  - compact the active set;
  - refill finished slots from the next replicate index;
  - stop starting new replicates once index R has been assigned.

  Refill matters most when R/B is large; with R/B ≥ 16, utilisation is ≥ 0.73–0.92 in the experiments.
- **(b) Asynchronous pools.** These suit expensive scalar log-densities, where per-pair cost is in milliseconds or more and pool overhead is negligible. To keep overhead below about 2%, a task should hold at least about 10 ms of work (≥ 50× the ~0.2 ms per-task cost). That means bundling cheap replicates into chunks.
- **`.map`-only user pools (emcee/dynesty style).** These are order-preserving barriers, so they force "waves": map over a chunk of indices and wait for the slowest. Wave waste follows the refill-utilisation numbers: use waves of at least about 16 replicates per worker. Prefer `concurrent.futures`-style executors with `submit`/`as_completed` for streaming, budgets and cancellation, and accept `.map` pools as a fallback.
- **Hybrid.** Each worker process runs its own lockstep batch: process-level parallelism multiplied by per-core vectorisation.
- **(c) JAX.** Do not `vmap` a whole-replicate `while_loop`; it pays the waste factor (2–11× above). Use one of:
  - a "persistent lanes" kernel: a `while_loop` over global steps with B fixed lanes, an on-device replicate counter, per-lane `fold_in(root_key, index)` keys, and results written into an R-sized buffer with `dynamic_update_slice`;
  - chunked `fori_loop`/`scan` of K steps, returning to Python to compact and refill.
- **(d) GPUs.** These only pay off with the JAX (or CuPy/array-API) backend and very large B. That is exactly where the no-refill waste is largest, since it grows like log B, so in-loop refill is essential on GPU.

### Gaps
- No JAX timing was run here (JAX was not installed). The waste factors are computed from the cost distribution, which is exactly what the `reduce_or`/`select` semantics imply, but kernel-launch overheads were not measured.
- Dask, Ray and mpi4py overheads were not measured locally; documented figures only.
- The Python array-API standard (e.g. `array_api_compat`) as a route to CuPy/PyTorch arrays was not evaluated.
- It was not verified whether NumPy 2.5 can be imported in subinterpreters, which `InterpreterPoolExecutor` would require.

## Random numbers: SeedSequence/Generator-per-replicate, counter-based generators, jumping, JAX keys, SPEC 7

### Takeaway
Give every replicate its own stream, addressed by its index: SeedSequence(root_entropy, spawn_key=(namespace, i)) → Generator(PCG64DXSM or Philox). Then results do not depend on worker count or scheduling. Reduce in index order. In a lockstep batch, per-replicate streams cost about 7–8× the batch RNG cost if drawn per step per lane, but only about 1.4–2.6× if each lane pre-draws blocks.

### Cited Findings
- **SeedSequence spawning.** Spawning "mixes in" each child's position in the spawn tree with the user seed. The collision probability is about n²·2⁻¹²⁸; for 2²⁰ streams this is about 2⁻⁸⁸, "solidly-ignorable".
  - NumPy recommends 128-bit entropy roots from `secrets.randbits`.
  - For ad-hoc integer IDs, the varying ID should go *before* the root, e.g. `default_rng([worker_id, root_seed])`, and `root_seed + worker_id` is "UNSAFE".
  - Source: [NumPy parallel RNG docs (v2.5)](https://numpy.org/doc/stable/reference/random/parallel.html)
- **Generator.spawn / BitGenerator.spawn.** Added in NumPy 1.25.0 — [Generator.spawn](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.spawn.html)
- **Index-addressed streams (own check).** `SeedSequence(root).spawn(5)[i]` produces the same stream as `SeedSequence(root, spawn_key=(i,))`, so any replicate's stream can be built directly without spawning all of them. Building one Generator costs about 11.5 µs (PCG64) or 12.4 µs (Philox) — [own experiment exp3_overhead.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3_overhead.out)
- **PCG64 vs PCG64DXSM.**
  - "Uses of the PCG64 BitGenerator in a massively-parallel context have been shown to have statistical weaknesses".
  - Thousands of `SeedSequence.spawn` streams are "negligibly small" risk. With millions of streams, each drawing a lot, the risk is "non-negligible, if still small", and such cases "should consider using PCG64DXSM explicitly or another modern BitGenerator like SFC64 or Philox".
  - PCG64DXSM "will eventually become the new default" for `default_rng`.
  - Source: [NumPy: Upgrading PCG64 with PCG64DXSM](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html)
- **Counter-based and jumped streams.**
  - Philox "generates values by encrypting an incrementing counter", and unique keys give independent streams; its 128-bit key can be set directly.
  - `jumped` is available for MT19937 (jump 2¹²⁸), PCG64 and PCG64DXSM (~2¹²⁷) and Philox (2¹²⁸), with the caveat that one must avoid re-jumping into used streams.
  - Source: [NumPy parallel RNG docs](https://numpy.org/doc/stable/reference/random/parallel.html)
- **Thread use.** NumPy's multithreaded example gives each thread its own Generator from `seq.spawn(threads)`, and notes reproducibility holds only "given that the number of threads does not change" when work is split by thread — [NumPy multithreaded generation](https://numpy.org/doc/stable/reference/random/multithreading.html)
- **JAX keys.**
  - Keys are explicit and functional: "never reuse keys", and `split`/`fold_in` derive new keys. Typed keys are made with `jax.random.key`; legacy keys with `PRNGKey`.
  - The default implementation is threefry2x32. `jax_threefry_partitionable` has been the default since JAX 0.5.0.
  - Threefry is identical across shardings and backends and exact under `vmap` over keys.
  - Source: [jax.random docs](https://docs.jax.dev/en/latest/jax.random.html); [JAX random-numbers tutorial](https://docs.jax.dev/en/latest/random-numbers.html)
- **SPEC 7 ("Seeding pseudo-random number generation").**
  - Standardise an `rng` keyword that accepts a Generator, None or anything `np.random.default_rng` accepts, normalised by `rng = np.random.default_rng(rng)`.
  - Deprecate `random_state` and `seed` in phases, raising an error if both old and new are given, and avoid global state, which "may cause problems in parallel processing scenarios".
  - Endorsed by IPython, NetworkX, NumPy, PySAL, scikit-image and SciPy.
  - Source: [SPEC 7](https://scientific-python.org/specs/spec-0007/)
- **Pigeons.jl.** Pigeons guarantees "strong parallelism invariance: the output for a given seed is identical irrespective of the number of threads and processes" — [Surjanovic et al. 2023, arXiv:2308.09769](https://arxiv.org/abs/2308.09769)
  - It associates "one random number generator to each PT chain" via SplittableRandoms.jl and uses `all_reduce_deterministically()`.
  - It warns that non-associative floating-point reductions can make runs with different worker counts "eventually arbitrarily diverge pointwise".
  - Julia task-local RNGs are reproducible only for a fixed thread count.
  - Source: [Pigeons distributed docs](https://pigeons.run/stable/distributed/)
- **Prior art in other packages.**
  - The R `unbiasedmcmc` scripts run `foreach(irep = 1:nsamples) %dorng% {...}` after `registerDoParallel(cores = ...)` and `set.seed(...)`, using doRNG reproducible parallel streams — [unbiasedmcmc germancredit.run.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/inst/reproducegermancredit/germancredit.run.R); [vignette](https://github.com/pierrejacob/unbiasedmcmc/blob/master/vignettes/introduction-normaltarget.Rmd)
  - dynesty seeds each queued pool task from `SeedSequence(rstate.integers(0, 2**63-1, size=4)).spawn(queue_size)`, and since 1.2.0 claims "100% reproducible results if the rstate object is provided" (a new-style Generator). Its per-task streams depend on `queue_size` but not on which worker runs a task — [dynesty utils.py get_seed_sequence](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py); [sampler.py `_fill_queue`](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/sampler.py); [docs changelog](https://github.com/joshspeagle/dynesty/blob/master/docs/source/index.rst)
  - emcee 3.1.6 still uses a legacy `np.random.mtrand.RandomState` held by the sampler — [emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)
  - johnoleary/mh-max-couplings calls `np.random.seed()  # needed for multiprocessing` in its coupled-kernel constructor. Re-seeding from OS entropy avoids identical forked streams but gives up reproducibility — [mh-max-couplings src/mcmc.py](https://github.com/johnoleary/mh-max-couplings/blob/main/src/mcmc.py)
- **Own experiment: worker-count invariance.**
  - Setup: 400 replicates, each seeded by `SeedSequence(ROOT, spawn_key=(i,))`, run on 1 vs 4 workers.
  - Per-index results were bitwise identical, but completion order differed.
  - Summing in completion order differed in the last digits (−210.88381624233094 vs −210.8838162423308). Summing in index order, or with `math.fsum`, was identical.
  - Source: [own experiment exp4_pool.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp4_pool.out)
- **Own experiment: per-replicate streams inside a lockstep batch.** Per-step RNG cost, d+2=12 normals per lane:
  - B=1000: batch Generator 141 µs; one Generator per lane, drawn every step, 1099 µs (7.8×); per-lane blocks of K=64 steps, 195 µs (1.4×).
  - B=10⁴: 1545 / 11233 / 4061 µs.
  - For comparison, a full d=10 lockstep step was about 370–400 µs at B=1000.
  - Source: [own experiment exp3b_rng_steady.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3b_rng_steady.out)
- **Own experiment: batch-layout reproducibility.**
  - Identical rows within one batched quadratic-form call gave identical bits (0/38 mismatches for d=3–257).
  - The same point evaluated in a batched call versus one at a time differed bitwise in 8–25 of 38 cases for d ≥ 7.
  - Source: [own experiment exp6_rowrepro.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp6_rowrepro.out)

### Inferences
- **Stream design.** Use the stream SeedSequence(entropy=root.entropy, spawn_key=root.spawn_key + (NS, i)). Namespaces NS separate main replicates, pilot or tuning runs, and anything else, so that extending R or adding a pilot never collides with existing streams. Draw the initial states (X₀, Y₀) from the replicate's own stream. Store `root.entropy` in the results so any run, or any single replicate, can be recomputed.
- **Bit generator.** Use PCG64DXSM, or Philox, as the per-replicate BitGenerator, because R can reach 10⁵–10⁶ for cheap targets.
- **Two levels of reproducibility.**
  - Statistical invariance to scheduling holds for all modes.
  - Bitwise invariance across batch sizes or backends additionally needs:
    - per-replicate streams;
    - a fixed amount of randomness consumed per step, so that block pre-drawing works. Reflection-maximal coupling of Gaussian random-walk Metropolis uses a fixed d normals + 2 uniforms; rejection-sampler maximal couplings use a random number of draws;
    - the same log-density call layout. Batched and single evaluation differ in the last bits, and can rarely flip an accept/reject decision.
  - Document "bitwise reproducible for a fixed backend configuration".
  - Offer a "fast" lockstep RNG mode, with one batch Generator, whose results depend on B and are flagged as such.
- **After meeting.** Copy X into Y rather than recomputing Y. When proposals coincide, evaluate the log-density once and share it. This guarantees faithfulness even if a backend gave different bits for identical rows.
- **JAX.** Use key_i = `fold_in(root_key, i)` and per-step `fold_in(key_i, t)` or `split`. This is natively vectorised and invariant to B and sharding with threefry.
- **API.** Follow SPEC 7: an `rng=` keyword, with no `random_state`/`seed`.

### Gaps
- No primary benchmark was found of PCG64DXSM vs Philox vs SFC64 throughput on current NumPy; only the docs' qualitative statement that DXSM "preserv[es] the performance".
- How R's doRNG constructs its L'Ecuyer streams was not checked in detail.

## Memory and storage: streaming accumulators vs stored chains or signed-measure atoms; formats; checkpointing and resuming

### Takeaway
Streaming accumulators cost O(dim h) per replicate and are the right default. Storing signed-measure atoms enables any test function after the fact, at a cost of about (ℓ − k + 1 + max(0, τ − k − L)) × d floats per replicate. Store results replicate by replicate, keyed by index, in HDF5 or Zarr. Because each replicate is a deterministic function of (root, index), a replicate-level checkpoint resumes exactly.

### Cited Findings
- **unbiasedmcmc's two modes.**
  - `sample_coupled_chains` stores both full trajectories: `samples1` has max(m,τ)+1 rows and `samples2` that minus `lag`. Storage is preallocated and doubled as needed. The result reports `meetingtime` (Inf if `max_iterations` interrupts), `iteration`, `elapsedtime` and `cost = lag + 2*(meetingtime - lag) + max(0, time - meetingtime)`.
  - `H_bar(c_chains, h, k, m)` then evaluates any h and any (k, m) up to the stored horizon.
  - `sample_unbiasedestimator` computes the estimator "on the fly, between step k and step m" and "does not record the trajectories, and thus is memory-light".
  - `c_chains_to_measure_as_list` and `histogram_c_chains` build signed-measure weights/atoms and histograms from stored chains.
  - The README warns: "This is not a general-purpose statistical software... Use at your own risk!!"
  - Source: [unbiasedmcmc R/coupled_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/coupled_chains.R); [R/unbiasedestimator.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/unbiasedestimator.R); [R/H_bar.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/H_bar.R); [README](https://github.com/pierrejacob/unbiasedmcmc)
- **Atom count.** The signed measure π̂ = (ℓ−k+1)⁻¹Σ_{t=k}^{ℓ} δ_{X_t} + Σ_{t=k+L}^{τ−1} v_t(δ_{X_t} − δ_{Y_{t−L}}) has N = max(0, τ − (k+L)) + (ℓ − k + 1) atoms, with possibly negative weights — [Atchadé & Jacob 2024 eq. 2.12](https://arxiv.org/abs/2406.06851); [Jacob et al. 2020 §2.4](https://arxiv.org/abs/1708.03625)
- **ColCarroll's `couplings` (Python, 2019).**
  - It preallocates `x` with shape (iters, chains, dim) and `y` with shape (iters−lag, chains, dim) for a lockstep batch (default chains=128).
  - It uses the global `np.random` state and detects meeting with `np.isclose`.
  - Its `short_circuit` option returns when *all* chains have met.
  - Source: [couplings/metropolis_hastings.py](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
- **emcee's HDFBackend.** It stores the chain in resizable HDF5 datasets (`maxshape=(None, nwalkers, ndim)`), grows them with `resize`, supports optional compression, and saves the random state for resumption — [emcee backends/hdf.py](https://github.com/dfm/emcee/blob/main/src/emcee/backends/hdf.py)
- **dynesty checkpointing.** `run_nested(..., checkpoint_file=None, checkpoint_every=60)` saves every N seconds, and `resume=True` / `restore(fname, pool=...)` continue a run. On restore, dynesty keeps the original `queue_size` so results stay consistent — [dynesty dynamicsampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/dynamicsampler.py); [utils.py restore_sampler](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)
- **HDF5 via h5py 3.16.**
  - Datasets resize up to `maxshape`, with `None` meaning unlimited, and "auto-chunking is also silently enabled when using the `compression` or `maxshape` keywords".
  - gzip is portable and lzf "very fast".
  - SWMR (single writer, multiple readers) is supported through `flush()` / `refresh()`.
  - Source: [h5py Datasets docs](https://docs.h5py.org/en/stable/high/dataset.html)
- **Zarr (zarr-python 3.x, format 3).**
  - Sharding is available via `shards=` "if you have large arrays but need small chunks".
  - Arrays "can ... be resized and appended to along any dimension".
  - Concurrent writes are safe when "Different processes write to different chunks" and the store supports atomic writes; threads writing overlapping regions need synchronisation.
  - Source: [zarr performance guide](https://zarr.readthedocs.io/en/stable/user-guide/performance/)
- **.npz.** "A `.npz` file is a zip file containing multiple `.npy` files". Memory-mapping applies to `.npy` (`open_memmap`) — [numpy.lib.format (v2.5)](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
- **Own check: RNG state round-trips.** A pickled `Generator` resumes the identical stream; the blob was 495 bytes for PCG64. So does assigning `bit_generator.state`, a dict with keys `bit_generator`, `state`, `has_uint32`, `uinteger` — [own experiment rng_state_check.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/rng_state_check.out)
- **Pigeons.jl.** Pigeons has a dedicated Checkpoints feature (linked from its parallel docs); not examined in detail — [Pigeons parallel docs](https://pigeons.run/stable/parallel/)

### Inferences
- **Memory arithmetic (from the atom-count formula).** Take d=100, k=100, ℓ=1000, L=1 and typical τ < k. Each replicate needs 901 atoms × 100 × 8 B ≈ 0.72 MB, plus weights, so R=10⁴ needs about 7 GB. Storing full chains from t=0, as unbiasedmcmc does, needs about 2 × 1001 × 100 × 8 B ≈ 1.6 MB per replicate. Streaming needs dim(h) floats; even h = (x, xxᵀ) is only about 5k floats.
- **Storage modes.** Offer:
  1. `store="estimates"` (default): per-replicate H, the MCMC and bias-correction parts separately (useful diagnostics), τ, cost, (k, ℓ, L), index, lane, start/end time;
  2. `store="atoms"`: start storing at k, not 0, as a CSR-like ragged layout — concatenated atoms, a per-replicate offsets array and weights — optionally float32;
  3. `store="chains"`: debugging only.

  Averaging the per-time estimators H_t over any deterministic subset of t ∈ [k, ℓ] (thinning) keeps unbiasedness and cuts atoms, at a variance cost. This is an inference from the time-averaged construction; verify the weights with lag L > 1.
- **Format.** Use one HDF5 (or Zarr) store per run with resizable 1-D ragged datasets plus a JSON manifest. The manifest holds root entropy, config, package versions, and the set of completed indices. Zarr suits multi-process writers, each writing its own chunk or shard range. HDF5 needs a single writer process (the parent), possibly with SWMR so progress can be monitored. Use .npz only for small final summaries.
- **Checkpoint and resume.**
  - The default unit of work is the replicate. Resuming runs exactly the missing indices and is bitwise identical, given the same backend layout.
  - For very long replicates, an optional mid-replicate snapshot pickles (t, X_t, Y_{t−1}, accumulators, Generator).
  - Budget runs additionally need per-lane bookkeeping. Resuming a budget run is better treated as "extend to fixed R" than as re-entering the same budget.
  - `max_iterations` caps are censoring: report censored replicates, never drop them silently. unbiasedmcmc marks them with `meetingtime = Inf`.

### Gaps
- No measured benchmarks of HDF5 vs Zarr append throughput for ragged per-replicate records were found or run.
- ArviZ 1.x has no standard representation for signed-measure (weighted, negative-weight) samples; this was not researched in depth.

## Overheads in pure Python: interpreter, vectorised log-densities, numba/JAX, pickling, BLAS oversubscription

### Takeaway
For cheap targets, overhead is everything. A single coupled pair costs about 12 µs per iteration in NumPy at d=10 (about 1 µs in scalar 1-D code), against about 0.4 µs per pair-iteration in a batch of about 1000. Ship the user's data to workers once, through an initializer, not with every task. Pin BLAS to one thread per worker: oversubscription was catastrophic in this sandbox.

### Cited Findings
- **Interpreter and dispatch overhead (own experiment).**
  - A d=10 lockstep step costs 38.5 µs at B=1, almost all of it NumPy call overhead.
  - Per pair-iteration it falls to 0.37 µs at B=1000.
  - A per-pair loop over small NumPy arrays costs 11.7 µs per iteration; pure 1-D scalar code costs 0.70 µs (stdlib `random`) to 1.21 µs (NumPy Generator scalars).
  - Source: [own experiment exp3_overhead.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3_overhead.out); [exp3b_rng_steady.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp3b_rng_steady.out)
- **emcee `vectorize=True`.** The log-probability receives all positions at once and `pool` is ignored — [emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)
- **emcee's parallel tutorial.**
  - It sets `os.environ["OMP_NUM_THREADS"] = "1"` because MKL or other threading "can cause problems when used with the parallelization methods described here".
  - It explains that the probability function and its args "must be pickled and passed every time the model is called".
  - Making the data a global means "the dataset is only pickled and passed between processes once (when the pool is created)".
  - Source: [emcee parallelization tutorial](https://emcee.readthedocs.io/en/stable/tutorials/parallel/) (source notebook `docs/tutorials/parallel.ipynb` in [dfm/emcee](https://github.com/dfm/emcee))
- **dynesty.pool.Pool.**
  - It creates a `spawn`-context `multiprocessing` pool whose initializer caches `loglike`, `prior_transform` and their args in a module-level singleton, "as this way they will not need to be pickled and unpickled every function call".
  - `map` uses `chunksize=1`, and `__exit__` calls `terminate()`.
  - Source: [dynesty pool.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/pool.py)
- **Own experiment: pickling a closure carrying 8 MB of data.** Pickling took 11.2 ms. Each task then cost 17.7 ms when the closure was shipped with it, versus 0.28 ms with an initializer-held global (4 forkserver workers) — [own experiment exp4_pool.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp4_pool.out)
- **Start methods.** With forkserver (the default on Linux since 3.14) or spawn, workers re-import `__main__`, so lambdas and REPL-defined functions fail with `ProcessPoolExecutor` — [Python docs](https://docs.python.org/3/library/concurrent.futures.html); [What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html). Pools that use dill can pickle such functions; schwimmbad's `MultiPool` builds on `multiprocess` — [schwimmbad](https://github.com/adrn/schwimmbad)
- **joblib/loky thread limiting.** Loky caps BLAS/OpenMP threads in each worker at `max(cpu_count() // effective_n_jobs, 1)`, overridable via `inner_max_num_threads`. Explicitly set environment variables such as OMP_NUM_THREADS take precedence — [joblib parallel guide](https://joblib.readthedocs.io/en/stable/parallel.html); [scikit-learn parallelism guide](https://scikit-learn.org/1.4/computing/parallelism.html)
- **threadpoolctl.** `with threadpool_limits(limits=1, user_api='blas'):`, a `ThreadpoolController` API, and a decorator via `.wrap`.
  - It "will not act on libraries loaded after the instantiation of the ThreadpoolController".
  - Each worker thread must limit its own libraries.
  - It "can fail to limit the number of inner threads when nesting parallel loops".
  - Source: [threadpoolctl README](https://github.com/joblib/threadpoolctl)
- **Own experiment: BLAS oversubscription.**
  - Workload: 4 processes × 8 jobs of 150 matmuls at 400×400.
  - Times: default threads 42.6 s; OPENBLAS_NUM_THREADS=4, 34.0 s; =2, 5.87 s; =1, 1.05–1.29 s.
  - Caveat on magnitude: even a single process with 4 threads was 10× slower than with 1 thread at n=400 (19.7 s vs 1.91 s), although it was 1.8× faster at n=2000 (0.47 s vs 0.84 s). So the size of the effect is specific to this VM; the direction is not.
  - Source: [own experiment exp5_blas.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp5_blas.out); [exp5b_blas.out](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp5b_blas.out)
- **Numba.**
  - "As of version 0.56, users can pass individual NumPy Generator objects into Numba functions and use their methods". It uses the same algorithms as NumPy, though results may differ by many ULPs on some platforms.
  - Numba's Generator support "is not thread-safe" and should not be used with parallel execution logic.
  - Numba is at 0.68.0 on PyPI.
  - Source: [Numba NumPy support](https://numba.readthedocs.io/en/stable/reference/numpysupported.html); [Anaconda engineering blog 2022](https://engineering.anaconda.com/2022/08/numba.html); [PyPI numba](https://pypi.org/project/numba/)
- **Free-threaded builds.** NumPy has had experimental support since 2.1 with CPython 3.13. "The easiest performance gains happen when each worker thread owns its own array" — [NumPy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html). Python 3.14 makes free-threading officially supported, with about a 5–10% single-thread penalty — [What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html)

### Inferences
- Default to lockstep vectorisation whenever the user supplies a vectorised log-density, and make a vectorised log-density the documented "fast path".
- For scalar log-densities that are cheap (< ~50 µs), use one process per core, each running a lockstep batch over a Python-level loop of the user's function. The loop gives no vectorisation gain, but it still amortises pool overhead.
- For expensive log-densities (> ~1 ms), use a plain process pool with one replicate (or a small chunk) per task.
- Workers should:
  - receive the target once (initializer and module-level cache, as dynesty's Pool does);
  - call `threadpool_limits(1)` in the initializer, or have BLAS env vars set before NumPy is imported. Spawn or forkserver children re-import NumPy, so env vars set in the parent before the pool is created are inherited;
  - ignore SIGINT.
- Numba: let users pass `@njit` log-densities. Telescopic's own kernels can stay NumPy. A numba-compiled coupled kernel for scalar targets is a later optimisation; note that passing Generators into numba ties reproducibility to numba's implementation.
- A thread-pool backend becomes attractive on free-threaded Python, or when the log-density releases the GIL (NumPy-heavy, numba `nogil`, JAX): no pickling, shared read-only data, one Generator per replicate.

### Gaps
- Numba- or JAX-accelerated kernels were not timed here.
- The free-threaded-build performance of NumPy-heavy coupled kernels was not measured.
- It is unknown whether NumPy 2.5 is subinterpreter-safe for `InterpreterPoolExecutor`.

## Progress reporting, cancellation and KeyboardInterrupt in long parallel runs

### Takeaway
Use `submit`/`as_completed` (or a lockstep loop) so progress can be reported per completed replicate. Make workers ignore SIGINT and let the parent cancel pending work and terminate workers. Return a clearly flagged partial result that never silently presents the biased "completed so far" average as final.

### Cited Findings
- `Executor.shutdown(wait=True, *, cancel_futures=False)`: with `cancel_futures=True`, futures not yet started are cancelled, while running ones complete. Python 3.14 added `ProcessPoolExecutor.terminate_workers()` and `kill_workers()`, which terminate or kill all live workers and then call `shutdown()` — [concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html); [What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html)
- **Known Ctrl-C pathologies.**
  - On Ctrl-C, `ProcessPoolExecutor` workers treat KeyboardInterrupt as an error of the running call and keep draining the queue; `shutdown(cancel_futures=True)` races with this — [CPython gh-125886](https://github.com/python/cpython/issues/125886)
  - Nesting `ProcessPoolExecutor` inside `multiprocessing.Manager` can hang after an interrupt — [CPython gh-119104](https://github.com/python/cpython/issues/119104)
  - Python 3.14 adds `multiprocessing.Process.interrupt()`, which sends SIGINT to a child — [What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html)
- **schwimmbad's pattern.** The worker initializer calls `signal.signal(signal.SIGINT, signal.SIG_IGN)`. The parent polls `r.get(self.wait_timeout)`, because "a Condition.wait() without a timeout swallows KeyboardInterrupts", and on KeyboardInterrupt calls `self.terminate(); self.join(); raise` — [schwimmbad multiprocessing.py](https://github.com/adrn/schwimmbad/blob/main/src/schwimmbad/multiprocessing.py)
- **dynesty's Pool.** `__exit__` calls `self.pool.terminate()`, then `join()` — [dynesty pool.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/pool.py)
- **Unordered results.** joblib `return_as='generator_unordered'` and mpi4py `map(unordered=True)` yield results as they finish — [joblib.Parallel](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html); [mpi4py.futures](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html)
- **Progress bars in existing samplers.**
  - emcee: `progress=True`, a string such as `'notebook'`, or `progress_kwargs` passed to tqdm — [emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)
  - dynesty: `print_progress=True` — [dynesty sampler.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/sampler.py)
  - NumPyro: disabling the progress bar "will improve the speed for many cases" — [NumPyro MCMC](https://num.pyro.ai/en/stable/mcmc.html)
- **Interrupted or partial runs are biased budget runs.** "Retaining the C estimators that are first completed would introduce a bias" — [Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851). "Completed by t" averages are biased when pooled — [GH 1991](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf); [own experiment](file:///tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/exp1_policies.out)

### Inferences
- **Progress model.**
  - In fixed-R mode, report completed replicates / R.
  - In budget mode, report elapsed / t plus per-lane counts.
  - Live statistics are cheap and diagnostic: running mean and quantiles of τ, the number of active lanes, an ETA based on mean cost, and how many replicates exceed k (the bias-correction term is non-zero only when τ > k + L).
  - In lockstep, update every K steps (or every ≥ 0.1 s) to keep bar overhead below about 1%.
  - Use tqdm when installed (an optional dependency); fall back to logging.
- **Cancellation protocol.**
  - On the first Ctrl-C, set a stop flag, stop submitting, call `shutdown(cancel_futures=True)`, and wait briefly for running replicates. On a second Ctrl-C, or after a timeout, call `terminate_workers()` (3.14+) or `pool.terminate()`.
  - Workers ignore SIGINT (initializer).
  - Return `Result(status="interrupted", ...)` holding all completed replicates by index and the checkpoint path, with `estimate=None` by default. Users can call `result.extend(to=R)` to finish the fixed-R set from the checkpoint. If lane bookkeeping exists, users can request the GH per-lane estimate with t = interrupt time; this may require finishing first replicates on empty lanes.
  - Never pool-average "whatever finished" without a warning.
- In lockstep, check for interrupts between steps (catch KeyboardInterrupt around the step loop); state is consistent at step boundaries, which makes mid-replicate checkpoints trivial.

### Gaps
- No primary source was found on the status of gh-125886, i.e. whether worker draining after Ctrl-C was fixed in 3.15. The behaviour should be tested on the supported Python versions.

## What existing code does for parallelism: unbiasedmcmc (R), emcee, dynesty, Pigeons.jl, BlackJAX/NumPyro, Python coupling code

### Takeaway
Existing coupled-MCMC code either relies on R's `foreach %dorng%` (one replicate per task, reproducible streams, no budget logic) or on fixed-horizon lockstep NumPy batches with global RNG state. No Python library found implements budget-aware, scheduling-invariant execution for coupled chains. The general samplers (emcee, dynesty, Pigeons, BlackJAX) supply reusable patterns rather than solutions.

### Cited Findings
- **unbiasedmcmc (R).**
  - Version 0.3.1 in DESCRIPTION; last commit 2026-09-23 ("resolves a bug in the Polya-Gamma Gibbs sampler's coupling"). It depends on Rcpp, RcppEigen and lubridate, and suggests doParallel and doRNG.
  - Replicates run as `foreach(irep = 1:nsamples) %dorng% { sample_coupled_chains(...) }` or `sample_unbiasedestimator(...)` after `registerDoParallel(cores = detectCores()-2)`.
  - Each replicate is one task and results are averaged over a fixed `nsamples`; there are no budget rules. Confidence intervals are mean ± 1.96·sd/√n.
  - `histogram_c_chains(..., dopar=TRUE)` parallelises over histogram bins.
  - Source: [pierrejacob/unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc); [vignette](https://github.com/pierrejacob/unbiasedmcmc/blob/master/vignettes/introduction-normaltarget.Rmd); [histogram_c_chains.R](https://github.com/pierrejacob/unbiasedmcmc/blob/master/R/histogram_c_chains.R)
- **Pierre Jacob's code page.** It lists R repositories (umcmchandbook, unbiasedmcmc, unbiasedpoisson), Python (colcarroll's `couplings`, johnoleary/mh-max-couplings, tinnguyen96/partition-coupling) and Julia (CoupledConditionalSMC.jl). None is described as JAX or as a general parallel engine — [unbiased-mcmc code page](https://pierrejacob.quarto.pub/unbiased-mcmc/code.html)
- **ColCarroll/couplings (Python, last commit 2019-09-19).** A lockstep NumPy batch of `chains` coupled pairs with a fixed iteration horizon, the global `np.random`, and `short_circuit` (stop when all have met). There is no refill and no per-replicate streams — [couplings](https://github.com/ColCarroll/couplings/blob/master/couplings/metropolis_hastings.py)
- **johnoleary/mh-max-couplings (Python, last commit 2021-02-19).** Re-seeds the global NumPy RNG from entropy in each coupled-kernel constructor "for multiprocessing" — [mh-max-couplings](https://github.com/johnoleary/mh-max-couplings/blob/main/src/mcmc.py)
- **emcee 3.1.6.**
  - `pool` is any object with `.map`, and `vectorize=True` bypasses it.
  - HDF5 backend for incremental storage and resume; tqdm progress; legacy RandomState.
  - Its parallel tutorial teaches the global-data trick and OMP_NUM_THREADS=1.
  - Source: [emcee](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py); [tutorial](https://emcee.readthedocs.io/en/stable/tutorials/parallel/)
- **dynesty 3.1.0.**
  - `pool` + `queue_size`, and per-task seeds spawned from the master Generator.
  - `dynesty.pool.Pool`: a spawn context, an initializer-cached likelihood, `map(chunksize=1)`, and terminate on exit.
  - Time-based checkpointing (`checkpoint_every=60` s) and `restore`.
  - Source: [dynesty](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/pool.py)
- **Pigeons.jl.**
  - Strong parallelism invariance via per-chain splittable RNGs and deterministic reductions.
  - Multithreading (`multithreaded = true`, using min(#chains, nthreads) threads) or MPI, including `ChildProcess` to run with other thread counts.
  - Source: [Pigeons parallel](https://pigeons.run/stable/parallel/); [Pigeons distributed](https://pigeons.run/stable/distributed/); [arXiv:2308.09769](https://arxiv.org/abs/2308.09769)
- **BlackJAX / NumPyro.** Multiple chains via `vmap` (SIMD, where each step lasts as long as the slowest chain) or via sharding/`pmap` (one chain per device) — [BlackJAX how-to](https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html); [NumPyro MCMC](https://num.pyro.ai/en/stable/mcmc.html)

### Inferences
- `telescopic` would be, as far as this search found, the first Python package to combine:
  - a refill-lockstep engine;
  - index-addressed per-replicate streams;
  - GH/Jacob budget estimators;
  - replicate-level checkpointing.

  Patterns worth copying:
  - doRNG's "stream per task index" design;
  - dynesty's initializer cache and terminate-on-exit;
  - schwimmbad's SIGINT handling;
  - Pigeons' invariance guarantee and deterministic reductions;
  - emcee's `vectorize` flag and `.map`-pool compatibility.

### Gaps
- tinnguyen96/partition-coupling, CoupledConditionalSMC.jl and umcmchandbook were not inspected.
- The search for a JAX implementation of unbiased MCMC with couplings found none, but it was not exhaustive (for example, research code attached to individual papers).

## Concrete recommendations for the execution layer of `telescopic`

### Takeaway
Build one core, a "replicate stream", with three executors that all obey the same contract: a lockstep-refill batch (default for vectorised targets), a process or executor pool (for expensive scalar targets), and serial. Under that contract, randomness is a function of (root, index); estimates come from a fixed index set or a GH per-lane budget rule; reductions are in index order; and storage is replicate-granular and resumable.

### Cited Findings
- The recommendations rest on the findings above, chiefly:
  - [GH 1990](https://web.stanford.edu/~glynn/papers/1990/GH90.pdf) / [GH 1991](https://web.stanford.edu/~glynn/papers/1991/GH91a.pdf) / [Jacob et al. 2020 §3.3](https://arxiv.org/abs/1708.03625) / [Atchadé & Jacob 2024](https://arxiv.org/abs/2406.06851) (budget estimators and selection bias);
  - [NumPy parallel RNG](https://numpy.org/doc/stable/reference/random/parallel.html) / [PCG64DXSM note](https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html) / [SPEC 7](https://scientific-python.org/specs/spec-0007/) (streams and API);
  - [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) / [What's New 3.14](https://docs.python.org/3/whatsnew/3.14.html) (pools, forkserver default, cancellation);
  - [jax loops.py](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py) (vmap waste);
  - [dynesty pool.py](https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/pool.py) / [schwimmbad](https://github.com/adrn/schwimmbad) / [threadpoolctl](https://github.com/joblib/threadpoolctl) (worker hygiene);
  - the own experiments in `/tmp/claude-0/-home-user-telescopic/901acb40-2335-5d56-b8d5-e448f1cd801d/scratchpad/research/parallel/experiments/` (bias of policies, waste factors, overheads, invariance).

### Inferences
1. **Estimator contract (non-negotiable).**
   - **Fixed-R mode** (default): use exactly the replicates with indices 0..R−1. This is exactly unbiased, invariant to scheduling, and bitwise reproducible for a fixed backend layout.
   - **Budget mode** (`budget=Budget(seconds=…)` or `Budget(iterations=…)`), with per-lane bookkeeping. `policy="unbiased"` (default) is GH91 μ̃₁ = Jacob et al. H̃(P,t). Alternatives: `"hard_deadline"` (GH91 μ̃₂ with preselected lanes) and `"complete_in_progress"` (pooled μ̄₂, lower variance, longest overrun). Report the SE from CLT (3.4) using lanes as units, and report the overrun beyond t.
   - Never offer "first R completed" or "pooled completed-by-deadline". Interrupted runs return a flagged partial result.
2. **Replicate streams.**
   - Use `rng=` per SPEC 7 and normalise to a root `SeedSequence`; record `root.entropy` and the spawn_key in results.
   - Replicate i uses `Generator(PCG64DXSM(SeedSequence(root.entropy, spawn_key=root.spawn_key + (NS_MAIN, i))))`. Draw the initial states from that same stream. Use separate namespaces for pilot and tuning runs.
   - In the JAX backend, use `fold_in(root_key, i)`.
3. **Lockstep-refill engine (`executor="vectorized"`, default when `vectorize=True`).**
   - B lanes (default about 256–4096, chosen by memory) with active-set compaction, so finished lanes cost nothing.
   - Refill from the index counter; in fixed-R mode, stop starting new replicates at R.
   - One batched log-density call per step on the stacked unmet-X, unmet-Y and met proposals.
   - After meeting, set Y := X by copying (faithfulness).
   - RNG modes:
     - `"per_replicate"` (default): per-lane Generators with block pre-draws of K≈64 steps. Requires a fixed number of draws per step; the measured RNG cost is about 1.4× the batch RNG at B=1000.
     - `"batch"`: fastest, but results depend on B. Document it as not invariant.
   - Make the budget clock lockstep iterations, so budget runs are deterministic, with an optional wall-clock deadline.
4. **Pool engine (`executor=` an `int`, a `concurrent.futures.Executor`, or any object with `.map`).**
   - Prefer the Executor protocol (`submit`/`as_completed`) for streaming, budgets and cancellation.
   - Accept `.map`-only pools (emcee/dynesty/schwimmbad style) in wave mode, with ≥ 16 replicates per worker per wave.
   - Tasks are chunks of indices sized to at least about 10 ms of work. Each worker may itself run a small lockstep batch (hybrid mode).
   - Ship the target once through an initializer cache, as dynesty's Pool does. Pin BLAS to one thread with `threadpoolctl.threadpool_limits(1)`, or env vars set before workers import NumPy. Ignore SIGINT in workers.
   - Document the forkserver/spawn picklability requirements (no lambdas or notebook-defined functions without a cloudpickle/dill-based pool) and the `if __name__ == "__main__"` guard.
   - For budget mode, use P long-lived lane tasks that loop until the deadline and return their own ordered lists.
5. **Reductions.** Sort results by replicate index before summing, or use `math.fsum`. Store per-replicate values so any statistic (CIs, trimmed means, bootstraps) can be recomputed exactly.
6. **Storage.**
   - `store="estimates"` (default): per-replicate H (vector-valued h), MCMC and bias-correction parts, τ, cost, k, ℓ, L, index, lane, timestamps.
   - `store="atoms"`: ragged CSR layout from t=k with weights (memory ≈ (ℓ−k+1+max(0,τ−k−L))·d·8 B per replicate).
   - Persist to HDF5 (single writer in the parent; SWMR for monitoring) or to Zarr v3 (per-worker chunk/shard ranges), with a JSON manifest.
   - Checkpointing is replicate-granular, and resume reruns only missing indices. An optional mid-replicate snapshot pickles the state and Generator.
7. **Interrupts and progress.**
   - The first Ctrl-C triggers a graceful stop (`cancel_futures`, finish running work, checkpoint). The second triggers `terminate_workers()` / `terminate()`.
   - The result is `status="interrupted"` with `estimate=None` unless a valid rule applies.
   - Show a tqdm progress bar when tqdm is available, with live τ quantiles and active-lane counts. In lockstep, update every K steps.
   - `max_iterations` is reported as censoring, never dropped.
8. **JAX backend (later).**
   - A persistent-lanes `while_loop` with in-loop refill (fixed B, an output buffer of size R, `dynamic_update_slice`), or chunked `scan` plus host-side refill.
   - Never use `vmap` over whole replicates (2–11× waste).
   - Use `shard_map` over devices for multi-GPU, with Threefry keys via `fold_in` for B- and sharding-invariance.
9. **Defaults by regime.** Measured crossovers on this VM:

   | Regime | Recommended executor |
   |---|---|
   | Vectorised target | Lockstep (≈0.4 µs per pair-iteration at d=10, B≈1000) |
   | Scalar target < ~50 µs | Processes × lockstep batches |
   | Scalar target ≥ ~1 ms | Plain process pool, 1–few replicates per task |
   | Cluster | `mpi4py.futures.MPIPoolExecutor` via the same Executor protocol |
10. **Testing hooks.** Use a test that runs the same seed on executors {serial, vectorized B=1/7/64, process with 1 and 4 workers} and asserts per-index equality, i.e. the invariance contract. Add a statistical test of the budget estimators against a toy with a known π(h), reusing the exp1 design: biased policies show |z| > 10 in seconds.

### Gaps
- The recommended defaults (B, K, chunk sizes) come from one 4-vCPU VM and one toy target. They should be re-benchmarked on typical user hardware and with expensive targets before they are fixed.
- The variance cost of the exactly unbiased per-lane budget estimator relative to the pooled-completion estimator (1.15–1.7× sd here) suggests offering both. No literature was found comparing their efficiency for coupled-chain costs specifically.
