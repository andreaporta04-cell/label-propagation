# `notebooks/` — experiments

Each experiment is a Jupyter notebook, numbered in the order it should be read:

```
01_two_moons.ipynb          ← first visual test on synthetic data
02_graph_hyperparameters.ipynb
03_digits_benchmark.ipynb
...
```

## Rules of thumb

- **No heavy logic in notebooks.** Functions go into the `labelprop` package
  (`src/`), notebooks only *call* them, plot and comment the results.
- **Reproducibility.** Fix the random seed at the top of every notebook and
  state which data it uses.
- **Outputs.** Figures and tables that end up in the README or in the report
  are saved to `results/` with the same numeric prefix as the notebook
  (e.g. `results/figures/01_two_moons_propagation.png`).
- **Run top to bottom** ("Restart & Run All") before committing, so that the
  saved outputs match the code.
