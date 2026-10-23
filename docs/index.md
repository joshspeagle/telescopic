# telescopic

*Unbiased Markov chain Monte Carlo with coupled chains and telescoping sums.*

telescopic runs pairs of coupled Markov chains until they meet, and turns each pair into
an exactly unbiased estimate of expectations under the target distribution. Averages
over independent pairs come with error bars that are valid as the number of pairs grows,
not as the chains grow, and the pairs run in parallel without communication. Their
meeting times also bound the distance of the chains from stationarity.

telescopic is in development; version 0.1 is a first draft.

```{toctree}
:maxdepth: 2
:caption: Using telescopic

getting-started
guide/index
tutorials/index
theory
api
```

```{toctree}
:maxdepth: 1
:caption: Project

citing
changelog
contributing
design/index
```
