# `report/` — final report

The LaTeX source of the project report will live here:

```
report/
├── main.tex          ← main document
├── references.bib    ← bibliography (BibTeX)
└── figures/          ← (optional) copies of figures from results/
```

## Compiling

- **Locally:** `latexmk -pdf main.tex`
- **Overleaf:** upload the folder, or link the GitHub repository.

LaTeX auxiliary files (`.aux`, `.bbl`, `.synctex.gz`, …) are ignored by git;
only the sources and the final PDF are committed.
