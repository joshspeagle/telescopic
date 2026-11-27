# Registered meeting comparisons

List the benchmark cases, original target IDs and evaluation caps with:

```bash
PYTHONPATH=src python -m benchmarks.registered
```

Select a case explicitly to run it. The launcher uses the registered tuning
seed and five evaluation seeds, with 100 independent pairs per seed. Tuning
is charged once per method and case. Each selected case keeps its own cap.

```bash
PYTHONPATH=src python -m benchmarks.registered --case mixture_equal_6:2 --start matched
```

The registry includes the three two-mode variants and preserves the original
target IDs. Select the matched or tenfold-wide initial distribution with
`--start`. Meeting measurements and fixed-window estimator coverage use
separate runs. Use the comparison runner with explicit averaging settings
for coverage experiments, following the saved registrations.
