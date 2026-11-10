# Label Propagation — Notes on the Reference Papers

*Study notes for the MATH-412 group project. They summarise the method of
Zhu, Ghahramani & Lafferty (ICML 2003), its four equivalent interpretations,
its spectral reading, the extensions proposed in the paper, and the follow-up
technical report that connects it to Gaussian processes.*

**Contents**

1. [The semi-supervised problem](#1-the-semi-supervised-problem)
2. [From data to a graph](#2-from-data-to-a-graph)
3. [The harmonic solution](#3-the-harmonic-solution)
4. [Why it is called *label propagation*](#4-why-it-is-called-label-propagation)
5. [Four ways to read the same function](#5-four-ways-to-read-the-same-function)
6. [The spectral view: Fourier analysis on a graph](#6-the-spectral-view-fourier-analysis-on-a-graph)
7. [Why unlabelled data matter](#7-why-unlabelled-data-matter)
8. [Extensions proposed in the paper](#8-extensions-proposed-in-the-paper)
9. [The follow-up: from Gaussian fields to Gaussian processes](#9-the-follow-up-from-gaussian-fields-to-gaussian-processes)
10. [Known limitations](#10-known-limitations)
11. [References](#11-references)

---

## 1. The semi-supervised problem

We observe $n$ points $x_1, \dots, x_n \in \mathbb{R}^d$. Only the first $l$
carry a label $y_i \in \{0, 1\}$ (the set $L$); the remaining $u = n - l$ are
unlabelled (the set $U$). In practice labels are expensive and $l \ll u$.

A purely supervised classifier can only use the $l$ labelled pairs. The idea of
semi-supervised learning is that the unlabelled points are still informative:
they reveal the *shape* of the data distribution. To exploit them we need an
assumption linking that shape to the labels. Label propagation uses the
**cluster / manifold assumption**:

> points that are close along the shape of the data (connected through a chain
> of nearby points) tend to share the same label.

## 2. From data to a graph

Every point becomes a node of a weighted graph. Two common constructions:

- **Fully connected, Gaussian weights**
  $w_{ij} = \exp\!\left(-\lVert x_i - x_j\rVert^2 / \sigma^2\right)$,
  where the bandwidth $\sigma$ sets which distances count as "close";
- **$k$-nearest-neighbour graph**: connect each point to its $k$ nearest
  neighbours (weights equal to 1, or Gaussian), then symmetrise.

From the weight matrix $W = (w_{ij})$ define the degrees
$d_i = \sum_j w_{ij}$, the diagonal matrix $D = \operatorname{diag}(d_i)$ and the
**graph Laplacian**

```math
\Delta = D - W ,
```

the same object that appears in spectral clustering. Its key property is that,
for any function $f$ defined on the nodes,

```math
E(f) \;=\; \frac12 \sum_{i,j} w_{ij}\,(f_i - f_j)^2 \;=\; f^\top \Delta f .
```

$E$ is a **smoothness energy**: it is small when $f$ changes little across
heavy edges, i.e. between points that are close. In particular $\Delta$ is
symmetric positive semi-definite, and the constant vector is in its kernel.

## 3. The harmonic solution

We look for a real-valued function $f$ (not a hard 0/1 labelling, which would
give a hard combinatorial problem) that

- is **clamped** to the known labels on $L$: $f_i = y_i$ for $i \in L$;
- **minimises** the energy $E(f)$ over the free values $f_U$.

Setting the gradient with respect to the unlabelled nodes to zero gives the
**harmonic property**: every unlabelled node equals the weighted average of its
neighbours,

```math
f_i \;=\; \frac{1}{d_i}\sum_j w_{ij} f_j , \qquad i \in U ,
```

i.e. $(\Delta f)_U = 0$. Splitting the matrices into labelled/unlabelled blocks,

```math
W = \begin{pmatrix} W_{LL} & W_{LU} \\ W_{UL} & W_{UU} \end{pmatrix},
\qquad
f = \begin{pmatrix} f_L \\ f_U \end{pmatrix},
```

the solution has the closed form

```math
f_U \;=\; (D_{UU} - W_{UU})^{-1}\, W_{UL}\, f_L \;=\; \Delta_{UU}^{-1}\, W_{UL}\, y_L .
```

Properties:

- **Existence and uniqueness**: $\Delta_{UU}$ is invertible as soon as every
  connected component of the graph contains at least one labelled node.
- **Maximum principle**: $f_U$ takes values in $[0, 1]$, being a weighted
  average of the boundary values.
- **Classification**: predict class 1 if $f_i > 1/2$, class 0 otherwise.
- **Multi-class**: with $C$ classes use one-hot labels $Y_L \in \{0,1\}^{l\times C}$,
  compute $F_U = \Delta_{UU}^{-1} W_{UL} Y_L$ and take the arg-max of each row.

## 4. Why it is called *label propagation*

The same solution is reached iteratively:

1. initialise $f_U$ arbitrarily (e.g. 0);
2. replace every value by the weighted average of its neighbours,
   $f \leftarrow D^{-1} W f$;
3. **clamp** the labelled nodes back to $y_L$;
4. repeat until convergence.

Labels literally *propagate* from the labelled nodes (the "boundary") to the
rest of the graph until equilibrium. The iteration is a Jacobi-type scheme for
the linear system $\Delta_{UU} f_U = W_{UL} y_L$ and converges to the harmonic
solution. It is useful when $u$ is large and inverting $\Delta_{UU}$ (cost
$O(u^3)$) is too expensive, especially with sparse $k$NN graphs.

## 5. Four ways to read the same function

The harmonic function $f$ has four equivalent interpretations, each giving a
different intuition.

- **Random walk.** Start at node $i$ and jump to neighbour $j$ with probability
  $w_{ij}/d_i$. Then $f_i$ is the probability of hitting a node of class 1
  *before* hitting a node of class 0.
- **Electric network.** Edges are resistors with conductance $w_{ij}$; class-1
  nodes are held at 1 V, class-0 nodes are grounded. $f$ is the resulting
  voltage at every node (Kirchhoff's laws are exactly the harmonic property).
- **Heat diffusion.** Labelled nodes are thermostats at fixed temperatures;
  $f$ is the equilibrium temperature. Unlike heat-kernel methods there is no
  diffusion time $t$ to tune: we look at the steady state.
- **Gaussian random field.** Put the prior
  $p(f) \propto \exp(-\beta\, f^\top \Delta f)$ on functions over the nodes.
  Conditioning on the observed labels, $f_U$ is exactly the posterior mean.
  This is a Gaussian process whose kernel is built from the graph (Section 9).

## 6. The spectral view: Fourier analysis on a graph

Let $\Delta = \sum_k \lambda_k \phi_k \phi_k^\top$ with
$0 = \lambda_1 \le \lambda_2 \le \dots$ The eigenvectors $\phi_k$ are the
**waves of the graph** and $\lambda_k = \phi_k^\top \Delta \phi_k$ measures how
much they oscillate.

- On a ring graph $\Delta$ is the discrete second derivative and the $\phi_k$
  are exactly sines and cosines.
- On a graph with two clusters, the first non-constant eigenvector is
  approximately a step function that separates the clusters (this is why
  spectral clustering works).

Writing $f = \sum_k a_k \phi_k$, the Gaussian prior of Section 5 becomes
independent coefficients $a_k \sim \mathcal{N}\big(0, 1/(\beta\lambda_k)\big)$:
**slow waves** (cluster indicators) have large variance, **rough waves** have
almost none. The method therefore explains the labels mostly with the slow,
cluster-level waves.

Changing the variance assigned to each wave gives the whole family of **graph
kernels**: $e^{-t\lambda_k}$ (diffusion kernel), $1/(\lambda_k + 1/\sigma^2)$
(regularised Laplacian), keeping only the first $p$ eigenvectors (spectral
truncation), and so on.

## 7. Why unlabelled data matter

With a standard RBF kernel, the covariance between two points depends only on
those two points: adding unlabelled data changes nothing. Here the **precision**
matrix $\Delta$ is local, but the **covariance** $\Delta^{-1}$ (restricted to the
appropriate block) depends on *all paths* through the graph. Two points at the
same Euclidean distance can be very similar if a chain of data connects them,
and very different if there is a gap in between. The unlabelled points are what
build those chains.

## 8. Extensions proposed in the paper

- **Class Mass Normalisation (CMN).** Thresholding at 1/2 often produces
  unbalanced predictions. If the expected proportion of class 1 is $q$
  (estimated from the labels with Laplace smoothing, $q = (n_1 + 1)/(l + 2)$),
  predict class 1 when
  ```math
  q\,\frac{f_i}{\sum_j f_j} \;>\; (1-q)\,\frac{1 - f_i}{\sum_j (1 - f_j)} .
  ```
  The ranking of the points does not change, only the decision threshold.
- **External classifiers ("dongles").** To combine the graph with a classifier
  $h$, attach to every unlabelled node a new labelled leaf node with value
  $h_i$; a parameter $\eta$ controls how much each node listens to its leaf
  rather than to its graph neighbours. The result interpolates between the
  graph solution and the classifier's predictions.
- **Learning the graph.** Use one bandwidth per feature,
  $w_{ij} = \exp\!\big(-\sum_d (x_{id}-x_{jd})^2/\sigma_d^2\big)$, and choose the
  $\sigma_d$ by minimising the entropy of the predictions on $U$ (by gradient
  descent, with smoothing to avoid the trivial solution $\sigma \to 0$). A
  value $\sigma_d \to \infty$ means feature $d$ is ignored: a form of feature
  selection.

## 9. The follow-up: from Gaussian fields to Gaussian processes

The companion technical report (Zhu, Lafferty & Ghahramani, CMU 2003) turns the
model into a proper Gaussian process:

- the kernel becomes $K = \big[\beta\,(\Delta + I/\sigma^2)\big]^{-1}$; the term
  $I/\sigma^2$ makes the prior proper (otherwise constants have infinite
  variance);
- labels are treated as **noisy** through a sigmoid likelihood, so a label in
  strong disagreement with the graph can be overruled;
- hyper-parameters of the graph are chosen by maximising the **evidence**
  (marginal likelihood) instead of entropy;
- predictions extend to new points not in the graph.

In the limit $\sigma^2 \to \infty$ with noise-free labels, the harmonic
solution of Section 3 is recovered. A practical conclusion of the report is
that an unweighted $k$NN graph with a well-chosen $k$ is already a very strong
baseline.

## 10. Known limitations

- **Graph construction is everything**: results depend heavily on $\sigma$ or
  $k$, and there is little labelled data to tune them.
- **Computational cost**: dense $W$ needs $O(n^2)$ memory, the closed form
  needs $O(u^3)$ time; sparse $k$NN graphs and iterative solvers are needed for
  large $n$.
- **Degeneracy in high dimension**: with very few labels and a very large
  number of unlabelled points in dimension $d \ge 2$, the harmonic solution
  degenerates to an almost constant function with sharp spikes at the labelled
  points (Nadler, Srebro & Zhou, 2009).
- **Transductive**: the basic method predicts only on the given points; new
  points require rebuilding or extending the graph.

## 11. References

1. X. Zhu, Z. Ghahramani, J. Lafferty. *Semi-Supervised Learning Using Gaussian
   Fields and Harmonic Functions.* ICML 2003. **(main reference)**
2. X. Zhu, J. Lafferty, Z. Ghahramani. *Semi-Supervised Learning: From Gaussian
   Fields to Gaussian Processes.* Technical report CMU-CS-03-175, Carnegie
   Mellon University, 2003.
3. X. Zhu, Z. Ghahramani. *Learning from Labeled and Unlabeled Data with Label
   Propagation.* Technical report CMU-CALD-02-107, Carnegie Mellon University,
   2002.
4. B. Nadler, N. Srebro, X. Zhou. *Semi-Supervised Learning with the Graph
   Laplacian: The Limit of Infinite Unlabelled Data.* NeurIPS 2009.
