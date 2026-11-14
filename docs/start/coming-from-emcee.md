# Coming from emcee

| emcee | telescopic |
|---|---|
| walkers, which move together as an ensemble | replicates: independent pairs of chains |
| burn-in steps, discarded | `k`, chosen from pilot meeting times; the bias left at `k` is corrected, not ignored |
| the autocorrelation time, which sets the error bars | meeting times, which set the cost; the error bars come from independent replicates |
| `get_chain(flat=True)` | `result.measure`: atoms whose weights can be negative; they are not samples, so plot them with their weights ([Signed measures](../guide/results.md#signed-measures)) |
| `log_prob_fn(theta, *args)`, `vectorize=True`, `pool=` | the same: `log_prob(theta, *args)`, `vectorize=True`, `pool=`, except that `log_prob` returns the log-density alone, without blobs; derived quantities come from `result.estimate` after the run |

telescopic does not make chains mix faster. Where one chain mixes slowly, as between
distant modes, pairs meet late and replicates cost more. Its gain is error bars that do
not depend on judging convergence, from replicates that run in parallel without
communication. [telescopic beside emcee](../tutorials/beside-emcee.md) runs both on the
line fit of the [Quickstart](quickstart.md).
