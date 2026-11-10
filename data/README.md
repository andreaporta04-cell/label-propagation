# `data/` — datasets

```
data/
├── raw/         ← data exactly as downloaded (never modified by hand)
└── processed/   ← data after cleaning / feature extraction
```

**The content of these folders is not tracked by git** (see `.gitignore`):
datasets can be large and are easy to re-download. Only the empty folders are
kept, thanks to the placeholder files `.gitkeep`.

## Datasets

| Dataset | Type | How to obtain |
|---|---|---|
| Two moons, circles, blobs | synthetic | generated in code (`labelprop.datasets`), nothing to download |
| Handwritten digits | benchmark | *to be decided* (scikit-learn / OpenML loader) |

When a dataset needs to be downloaded, the code that does it will live in
`labelprop.datasets`, so that anyone can rebuild `data/` from scratch.
