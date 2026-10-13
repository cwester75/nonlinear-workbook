# Nonlinear Workbook

## Chapter 3

Implemented planar continuous-time classification, homoclinic oscillator, pendulum,
Van der Pol, printed sine damping, radial equilibria and Lotka-Volterra. See
[guide](docs/chapter03.md) and [validation](docs/chapter03-validation.md).
Run `python examples/chapter03/validation.py` for phase portraits, invariant checks
and shared solver-to-Chapter2 sampling/noise sweeps.

## Chapter 10 and shared validation

Euler, fixed/adaptive Fehlberg4(5), supplied-derivative Taylor expansion, ghost
recurrences and separable symplectic Euler/Verlet are implemented. See
[Chapter10 guide and validation](docs/chapter10.md). The shared observation harness
connects known equations to Chapter2 diagnostic sweeps with controlled sampling
and seeded noise. Run `python examples/chapter10/validation.py`.

## Chapter 12

Implemented binary genetic search, schema/encoding utilities, exact-reference
knapsack, PMX/composition TSP search, simulated annealing and persistent serial
islands. See [guide](docs/chapter12.md) and [validation](docs/chapter12-validation.md).
Run `python examples/chapter12/search.py` with the existing frozen examples
environment to produce seeded JSON/CSV diagnostics and a headless plot.

Reproducible, independently written implementations of the book's dynamics and
examples, starting with Chapter 2. Python with NumPy is the proposed foundation:
it supports concise numerical methods, tests, and future notebook/plot workflows.
C++ ports can follow after numerical conventions and benchmarks are established.

## Structure

- src/nonlinear_workbook/: importable algorithms for reuse by outlierhunter.
- examples/: deterministic chapter demonstrations.
- tests/: analytic and synthetic reference checks using standard-library unittest.
- docs/: book corrections, conventions, and validation results.

## Run (PowerShell, Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python examples/chapter02.py
```

From outlierhunter's environment, install this checkout with:
`python -m pip install -e C:/Users/cw_se/quant/nonlinear-workbook`.
For reproducible releases, pin a reviewed Git commit instead of an editable path.

## Chapter 2 scope

Implemented: Pearson correlation; logistic-map generator and known-equation
Lyapunov reference; Kantz neighborhood-divergence curve; classical R/S Hurst
estimation; LZ76 symbolic complexity. Wolf replacement, direct divergence, and local-Jacobian QR spectra
are also implemented. See docs/book-corrections.md before translating printed code.

Examples use seed 20261010. Validation compares logistic divergence with ln(2),
a stable logistic fixed point with ln(0.5), IID Gaussian R/S scaling with an
approximate 0.5 baseline, and LZ counts with the book's simple symbolic parses.
R/S finite-sample bias and Kantz fit-window sensitivity remain relevant; these
methods do not independently certify chaos or provide anomaly thresholds.

No license has been selected yet. The source book PDF is kept outside Git.

## Research implementation

Chapter 2 now includes direct nearest/neighborhood divergence, Wolf replacement,
and local-Jacobian QR spectra. Existing functions and return values are preserved.
See [Chapter 2 guide](docs/chapter02.md) for coverage, parameter units, diagnostics,
limitations, and pinned validation commands. Run
`python examples/chapter02/research_validation.py` for deterministic parameter
sweeps. The reference runtime is CPython 3.12.14 with NumPy 2.3.5; requirements
are pinned in requirements-validation.txt. Derivative projects remain future work.

## Chapter 1

Completed both one-dimensional and multidimensional stages. Import methods from
`nonlinear_workbook.chapter01`; see [coverage and APIs](docs/chapter01.md),
[validation](docs/chapter01-validation.md), and [book corrections](docs/book-corrections.md).
The optional examples dependency is frozen with its complete dependency set in
requirements-examples.txt. Headless Stage A/B scripts generate numerical reports,
CSV data, and plots under ignored outputs/chapter01/. Chapter 2 APIs are preserved.

## Chapter 11

Implemented the full neural-network example families: Hopfield recall and cycle
handling, similarity/Fourier features, Kohonen learning and exact-reference TSP,
perceptrons, and online/batch sigmoid backpropagation. Import from
`nonlinear_workbook.chapter11`. See [Chapter 11 guide](docs/chapter11.md) and
[validation](docs/chapter11-validation.md). The three example scripts generate
JSON/CSV diagnostics and seven PNGs under ignored outputs/chapter11/. Existing
Chapter 1/2 APIs and pinned dependencies are preserved.
