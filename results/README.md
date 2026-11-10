# `results/` — figures and tables

```
results/
├── figures/   ← plots (.png for the README, .pdf for the report)
└── tables/    ← metrics and numerical results (.csv)
```

Unlike `data/`, results **are tracked** by git: they are small and they are
what we show in the README and in the report.

## Naming convention

Every file starts with the number of the notebook that produced it, so it is
always clear where it comes from:

```
figures/01_two_moons_propagation.png
tables/03_digits_accuracy_vs_labels.csv
```
