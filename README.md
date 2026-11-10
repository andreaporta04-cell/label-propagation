# Label Propagation

**Semi-supervised learning with Gaussian fields and harmonic functions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)

Group project for **MATH-412 Statistical Machine Learning** (EPFL, Fall 2026).
We study, implement from scratch and test the label propagation method of
Zhu, Ghahramani & Lafferty (ICML 2003).

---

## Table of contents

- [Semi-supervised learning in a nutshell](#semi-supervised-learning-in-a-nutshell)
- [The method](#the-method)
- [Reference papers](#reference-papers)
- [Project plan](#project-plan)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Team](#team)
- [License](#license)

---

## Semi-supervised learning in a nutshell

In many real problems data are cheap but **labels are expensive**: think of
images that must be annotated by hand, or medical records that need a
specialist. We typically have a few labelled examples and many unlabelled ones.

- *Supervised learning* uses only the labelled examples and ignores the rest.
- *Unsupervised learning* uses all the data but no labels.
- **Semi-supervised learning** uses both: the few labels tell us *what* the
  classes are, the many unlabelled points tell us *how the data are shaped*.

The unlabelled data help only under an assumption that links shape and labels.
The one used here is the **cluster assumption**: points connected by a chain of
nearby points should share the same label. In the classic "two moons" example,
with just one labelled point per moon, a supervised classifier draws a straight
line through the data, while a semi-supervised method follows the moons.

## The method

1. **Build a graph** on all points (labelled and unlabelled), with edge weights
   $w_{ij} = \exp(-\lVert x_i - x_j\rVert^2/\sigma^2)$ or a $k$-nearest-neighbour
   rule.
2. **Look for a smooth function** $f$ on the nodes that matches the known labels
   and varies as little as possible along heavy edges, i.e. minimises
   $E(f) = \tfrac12\sum_{i,j} w_{ij}(f_i - f_j)^2 = f^\top \Delta f$, where
   $\Delta = D - W$ is the graph Laplacian.
3. The minimiser is **harmonic**: every unlabelled node equals the weighted
   average of its neighbours. It has the closed form

   ```math
   f_U = (D_{UU} - W_{UU})^{-1} W_{UL}\, y_L ,
   ```

   and is obtained equivalently by letting the labels *propagate* iteratively
   through the graph until equilibrium.
4. **Classify** each unlabelled point by thresholding $f_i$ at $1/2$ (or with
   class mass normalisation).

The same $f$ can be read as a random-walk hitting probability, an electric
potential, an equilibrium temperature, or the posterior mean of a Gaussian
random field.

📖 **Full explanation:** see the study notes in
[`docs/notes_label_propagation.md`](docs/notes_label_propagation.md).

## Reference papers

- **Main reference** — X. Zhu, Z. Ghahramani, J. Lafferty.
  *Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions.*
  ICML 2003.
- **Follow-up** — X. Zhu, J. Lafferty, Z. Ghahramani.
  *Semi-Supervised Learning: From Gaussian Fields to Gaussian Processes.*
  CMU-CS-03-175, 2003. Turns the model into a full Gaussian process with noisy
  labels and evidence-based selection of the graph hyper-parameters.
- **Known limitation** — B. Nadler, N. Srebro, X. Zhou.
  *Semi-Supervised Learning with the Graph Laplacian: The Limit of Infinite
  Unlabelled Data.* NeurIPS 2009.

## Project plan

What we are going to do:

1. **Implementation from scratch** (NumPy/SciPy) of
   - graph construction: Gaussian weights and $k$NN graphs;
   - the harmonic solution, both in closed form and as an iterative
     propagation;
   - class mass normalisation (CMN).
2. **Experiments on synthetic data** (two moons, concentric circles, Gaussian
   blobs) to visualise how labels propagate and how the choice of $\sigma$ or
   $k$ changes the result.
3. **Experiments on classic benchmarks** (e.g. handwritten digits, as in the
   original paper):
   - accuracy as a function of the number of labelled points $l$;
   - comparison with supervised baselines (1-NN, RBF-kernel SVM) and with the
     scikit-learn implementation;
   - thresholding at 1/2 vs CMN.
4. **Analysis**: sensitivity to the graph hyper-parameters, behaviour with very
   few labels, and the degeneracy described by Nadler et al. (2009).
5. **Final report** in LaTeX.

## Repository structure

Each folder contains its own short README explaining what goes inside.

```
label-propagation/
├── README.md            ← this file
├── LICENSE              ← MIT license
├── .gitignore           ← files git should not track
├── pyproject.toml       ← package configuration and dependencies
├── docs/                ← study notes on the papers
│   └── notes_label_propagation.md
├── src/labelprop/       ← our Python package (reusable code)
├── notebooks/           ← experiments, numbered 01_, 02_, …
├── data/                ← datasets (contents not tracked by git)
│   ├── raw/
│   └── processed/
├── results/             ← outputs of the experiments
│   ├── figures/
│   └── tables/
└── report/              ← LaTeX report
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/andreaporta04-cell/label-propagation.git
cd label-propagation
```

### 2. Create an isolated Python environment

Pick **one** of the two options.

**Option A — conda** (if you use Anaconda; on Windows run it in the *Anaconda Prompt*):

```bash
conda create -n labelprop python=3.11
conda activate labelprop
```

**Option B — venv** (built into Python):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux
```

### 3. Install the package and its dependencies

```bash
pip install -e ".[dev]"
```

`-e` (*editable*) installs `labelprop` as a link to `src/`: every change to the
code is picked up immediately, without reinstalling. `[dev]` adds Jupyter and
pytest. All dependencies are listed in [`pyproject.toml`](pyproject.toml).

### 4. Check that everything works

```bash
python -c "import labelprop; print(labelprop.__version__)"
jupyter lab                     # then open the notebooks/ folder
```

Remember to **activate the environment** (step 2, second line) every time you
open a new terminal.

## Team

- Andrea Porta ([@andreaporta04-cell](https://github.com/andreaporta04-cell))
- *(teammates to be added)*

## License

Released under the [MIT License](LICENSE).
