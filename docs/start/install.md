# Install

telescopic needs Python 3.11 or newer, NumPy, SciPy and threadpoolctl. Install telescopic
and its optional dependencies from a clone of the repository:

```bash
pip install ".[all]"
```

The extras are `plot` (matplotlib, for the plots), `progress` (tqdm, for progress
bars), `parallel` (cloudpickle, which sends functions defined in a notebook to worker
processes), `pandas` (summaries as data frames) and `all`.
