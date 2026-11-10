# `src/` — the `labelprop` package

All reusable code lives here, as a Python package called `labelprop`.
Notebooks and scripts **import** from it instead of copying code around:

```python
from labelprop.graph import gaussian_weights
from labelprop.propagation import harmonic_solution
```

## Why a `src/` folder?

Putting the package inside `src/` (the so-called *src layout*) forces us to
install it (`pip install -e .`, see the main README) before using it. This way
the notebooks always run against the same code, wherever they are launched
from, and we catch packaging mistakes early.

## Planned modules

| Module | Content |
|---|---|
| `graph.py` | weight matrices (Gaussian, $k$NN), degree matrix, graph Laplacian |
| `propagation.py` | harmonic solution (closed form + iterative), class mass normalisation |
| `datasets.py` | synthetic data (two moons, circles, blobs), benchmark loaders, label masking |

## Conventions

- Every public function has a docstring explaining inputs, outputs and the
  corresponding formula in the paper.
- Mathematical notation follows [`docs/notes_label_propagation.md`](../docs/notes_label_propagation.md)
  (`W`, `D`, `L`, `U`, `f_U`, …).
