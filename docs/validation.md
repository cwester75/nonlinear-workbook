# Initial validation

Executed 2026-10-10 with bundled Python and NumPy, standard-library unittest.
All six test cases pass, covering affine Pearson relationships, constant-input
rejection, logistic-map analytic exponents, Kantz divergence and embedding bounds,
IID Gaussian R/S scaling, and symbolic LZ reference counts.

Example seed: 20261010.

| Benchmark | Observed | Reference / interpretation |
|---|---:|---|
| Logistic r=4, derivative exponent, 100000 samples | 0.6931506023 | ln(2) = 0.6931471806; tolerance 0.005 |
| Logistic r=2.5 stable fixed point | -0.6931471806 | ln(0.5); checked to 10 decimal places |
| Kantz first three lag slope, 2500 samples | 0.6951743814 | ln(2); tolerance 0.15 |
| IID Gaussian R/S H, 65536 samples | 0.5228834454 | Approximate 0.5 baseline; tolerance 0.08 |
| Alternating binary sequence LZ count | 3 | Book's symbolic example |
| Constant binary sequence LZ count | 2 | Book's symbolic example |
| Random binary normalized LZ, 4000 symbols | 1.0320488946 | Finite-sample values may exceed one |

Tolerances are benchmark checks, not confidence intervals. No noisy-chaos,
long-memory, forecasting, or anomaly-detection performance is established.
The known-map derivative exponent is deliberately separate from data-only Kantz
estimation. Examples report the full curve; slope selection is explicit.

The initial suite has been extended below with Wolf and Jacobian validation.


## Completed Chapter 2 validation (2026-10-10)

Runtime: CPython 3.12.14, NumPy 2.3.5, setuptools 81.0.0, Windows
64-bit. Pinned requirements and Python version are stored in the repository.
All 15 unittest cases pass, including the unchanged six initial tests. The
original example also runs. Two final research-example runs produced identical
SHA256: 23D1746E0B8F0CE0DDFE272108385F3B70D62202311C10C241C9B024498D0908.
Reproduction commands appear in [Chapter 2 guide](chapter02.md).

| Check | Observed | Predeclared acceptance / interpretation |
|---|---:|---|
| Wolf logistic, 4000 samples | 0.6933917406 | ln(2), absolute tolerance 0.15 for finite neighborhoods/folds/replacement |
| Linear Jacobian QR, nonsymmetric triangular map | ln(1.01), ln(0.99) | Absolute tolerance 1e-10; checks orientation and both signs |
| Controlled Wolf segment | ln(2) in one step | Floating-point equality via assertAlmostEqual; verifies log-distance accounting |
| Direct nearest and neighborhood estimates | [0.6927723864, 0.6930265758] | Tests use 2500 samples, tolerance 0.08 against ln(2) |
| Wolf sensitivity range, 1600 samples | 0.689513 to 0.693926 | Diagnostic, no optimal parameter selected |
| Scalar local spectrum sensitivity | 0.691941 to 0.694681 | Diagnostic against ln(2) |
| IID Gaussian R/S across three seeds | 0.532329 to 0.549841 | Finite-sample bias; original 65536-sample test retains 0.08 budget |
| Short-range AR(1), phi=0.9 | 0.707470 to 0.715102 | Elevated H does not imply long memory |

The sweeps use dimensions 1/2, Theiler exclusions 0/20, Wolf radii .005/.02,
max_angle .3 radians, min_distance 1e-6, max_steps 4, local neighbor counts 8/24,
and QR burn-in 20. Logistic r=4, x0=.618, burn-in 1000. Units are per observation
step (sample_interval=1). Runtime and all controls are included in emitted JSON.

Kantz uses radius .01 and 12 lags. Fit slices [0:3], [2:5], [7:11] show early
exponential growth and late saturation. With dimension=1 and Theiler=20, the
noiseless early slope is 0.699880, while the late slope is 0.000682; observation
noise std=.01 increases the early slope to 1.146712. Full curves and eligible
reference counts are emitted, not just fitted slopes.

Two-dimensional logistic local fits produce a largest QR estimate around 1.39
rather than ln(2), with poorly constrained transverse dynamics. These outputs
are deliberately documented failure examples, not accepted spectrum benchmarks.
The analytic linear-map test validates Jacobian algebra without that manifold
ambiguity. Rank-deficient fits terminate; tests confirm partial diagnostics and
that no later trajectory gap is bridged.

LZ examples give counts 2 (constant), 3 (periodic), and 6 (book parse).
Seed-42 random binary normalization is 1.035040; the logistic critical-point
partition gives 1.026066. This differs from the initial random example because
that example draws from a different seeded RNG stream. Neither value is clipped.
Constant normalization is zero under the observed-alphabet convention.

Failure tests cover duplicates, unavailable neighbors/replacements, zero evolved
distance, insufficient embedding history, invalid controls, excessive burn-in,
rank deficiency, elapsed-time scaling, and preserved successor bounds.
These error budgets are implementation benchmarks, not statistical confidence
intervals. No automatic chaos decision or performance on external data is claimed.

## Chapter 1 validation

See [Chapter 1 validation](chapter01-validation.md) for the staged analytic checks,
finite-sample sensitivity findings, runtime pins, reproduction commands, and figure
verification. Chapter 2's 15 tests are included in both stage regression runs.

## Chapter 11 validation

See [Chapter 11 validation](chapter11-validation.md) for finite-state memory checks,
Kohonen worked updates, exact TSP comparison, all 16 Boolean functions, full gradient
checks, online/batch XOR training, and reproducible example artifacts. The original
31 Chapter 1/2 tests remain included in each stage regression run.
