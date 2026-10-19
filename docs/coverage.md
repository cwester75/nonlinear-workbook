# Release coverage matrix

Current release: v0.2.0, covering all 13 chapters. See
[release validation](release-v0.2.0.md) for the synthetic-only evidence and pinning contract.

This matrix distinguishes implemented numerical examples from explanatory theory.
Each chapter guide has the section-level inventory, printed-page references,
corrections and exact validation commands. A completed chapter means its declared
computational scope is implemented and tested, not that numerical experiments
prove the chapter's theorems or every source assertion is correct.

| Chapter | Status | Computational scope | Evidence |
|---|---|---|---|
| 1 | Implemented | Scalar/multidimensional maps, Fourier methods, stability, dimensions and Newton methods | chapter01.md; Stage A/B tests and examples |
| 2 | Implemented | Correlation, embedding, Wolf/Kantz/local Jacobians, R/S and LZ | chapter02.md; analytical/synthetic diagnostics |
| 3 | Implemented | Planar flows, equilibria, pendulum, limit cycles and invariants | chapter03.md; exact homoclinic and conservation checks |
| 4 | Implemented | Hamiltonians, periodic/open Toda, Lax, Mathieu and sections | chapter04.md; conservation and variational checks; long quartic failure documented |
| 5 | Implemented | Dissipative models, full flow QR and exterior area growth | chapter05.md; analytical rates, trace identities and sensitivity |
| 6 | Implemented | Driven systems, phase-aware sampling, Floquet, signals | chapter06.md; linear forcing and spectral normalization |
| 7 | Implemented | Bounded control, entrainment, conditional/transverse spectra | chapter07.md; local feedback and synchronization controls |
| 8 | Implemented | Fractal constructions, escape images, finite sums/flights | chapter08.md; exact geometry and estimator sensitivity |
| 9 | Implemented | Elementary/2D automata, Life, cycles and puzzle | chapter09.md; exact finite-state references and GUI smoke |
| 10 | Implemented | Euler, Fehlberg, supplied Taylor and separable symplectic methods | chapter10.md; order checks and numerical failure regressions |
| 11 | Implemented | Hopfield, Kohonen, perceptrons and sigmoid networks | chapter11.md; exact fixtures and gradient checks |
| 12 | Implemented | Genetic search, TSP, knapsack, annealing and islands | chapter12.md; exact small references and evaluation accounting |
| 13 | Implemented | Membership operators, defuzzification, exact finite extension, alpha-cut/LR arithmetic and rule inference | chapter13.md; exact fixtures, approximation/grid/operator sensitivity |

Shared release additions provide a causal, shadow-only diagnostic window adapter,
periodic/stochastic synthetic controls, explicit unavailable results, and baseline
feature comparisons. No held-out market archive is consumed by this release.
