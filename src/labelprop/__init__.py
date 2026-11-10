"""labelprop: semi-supervised learning with label propagation.

Implementation from scratch of the method of Zhu, Ghahramani & Lafferty,
"Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions"
(ICML 2003), developed for the MATH-412 project at EPFL.

Planned modules:
    graph        -- build the weight matrix W and the graph Laplacian
    propagation  -- harmonic solution (closed form and iterative) and CMN
    datasets     -- synthetic data generators and loaders for benchmarks
"""

__version__ = "0.1.0"
