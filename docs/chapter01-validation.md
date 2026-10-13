# Chapter 1 validation

Executed 2026-10-11 in CPython 3.12.14 / NumPy 2.3.5 /
Matplotlib 3.11.2 on Windows 64-bit. Full example dependencies are
frozen in requirements-examples.txt; pip check reports no broken requirements.
Editable package installation with no build isolation and no dependency resolution
also succeeds. Numerical core dependencies remain unchanged.

## Acceptance and regression

Stage A: 8 new tests plus the existing 15 Chapter 2 tests passed before Stage B
implementation. Final: all 31 tests pass (16 Chapter 1, 15 Chapter 2). Syntax
compilation of source, examples, and tests succeeds. Original Chapter 2 public
interfaces remain intact; Chapter 1 imports the existing logistic generator.

Acceptance budgets were specified in tests before running benchmarks: exact
fractions/integers require equality; Fourier checks use 1e-13 to 1e-12 absolute
error; Jacobian finite differences use 1e-8; logistic exponent error budget is
0.005; density bin-mass error is below 0.005; the first two period-doubling roots
use 1e-9; the final displayed Feigenbaum ratio uses a 0.05 asymptotic-error budget.
Monte Carlo acceptance uses five estimated IID standard errors. These are
benchmark budgets, not guarantees or statistical confidence intervals.

| Benchmark | Observed | Target / interpretation |
|---|---:|---|
| Logistic derivative exponent, 100000 observations | 0.6931506023 | ln(2) = 0.6931471806 |
| Logistic invariant bin masses, 40 bins | Maximum error 0.001299 | Analytical arcsine CDF; density integral one |
| Cosine custom FFT vs NumPy | Maximum error 1.72e-16 | Forward normalization 1/N |
| First period-doubling threshold | 3.000000000000 | 3 |
| Second threshold | 3.449489742783 | 1+sqrt(6) |
| Following thresholds | 3.544090359552, 3.564407266095, 3.568759419544 | Multiplier=-1 and full-period closure |
| Feigenbaum ratios | 4.751446218, 4.656251018, 4.668242236 | Approach 4.669201609; finite cascade |
| IID mean integration of sin(x) | 0.460464148 +/- 0.000782660 SE | 1-cos(1) = 0.459697694 |
| Hit/miss integration | 0.461150000 +/- 0.001576367 SE | Same target |
| Circle rotation, zero forcing | 0.370000000000 | omega=0.37 |
| Bernoulli/tent float64 collapse | Iterations 53 / 54 | Numerical precision demonstration, x0=.618 |
| Henon QR spectrum | 0.417992143, -1.621964948 | Largest approximately .42; sum ln(.3) |
| Henon exponent-sum error | 1.55e-15 | Absolute budget 1e-11 |
| Standard-map spectrum sum | -1.39e-17 | Jacobian determinant one |
| Uncoupled r=4 logistic spectrum | 0.692532725, 0.693232260 | Both approximately ln(2) |
| Vector Newton from book initial point | 10 updates, residual 2.22e-16 | Negative root (-1/sqrt(2), -1/sqrt(2)) |

The deterministic exact fixtures reproduce Collatz from 56, logistic from 1/3,
Bernoulli from 1/17, and Gauss from 23/101. Bungalow with a=1/3 equals tent on
branch boundaries. Fourier fixtures include complex inputs, known cosine bins,
and inverse round trips. The long exact tent example uses a seeded 4096-bit
rational initial condition; its periodic rational trajectory avoids float64
collapse but does not constitute an irrational orbit or a proof of ergodicity.

## Sensitivity and limitations

Henon correlation fits range from 1.227167 to 1.316219
across sample sizes 1000/2500, temporal exclusions 0/20, and slices [2:9]/[4:12].
Box-count fits range from 1.127104 to 1.218397 across supplied grid
origins [-2,-1]/[-1.999,-.999] and those fit slices. The box curve bends toward
sample saturation at fine scales. The book's approximate 1.2/1.26 values are
comparison context, not exact acceptance targets. No automatic fit region is chosen.

The coupled example r=3.7, coupling=.06 gives finite-time exponents
0.380677 and
0.313586. The RTN sequential
example gives -0.000293 and
-0.618000, including a near-zero direction;
this trajectory does not by itself establish the chapter's theoretical route to
chaos. Analytic Jacobians for all six map families are checked against centered
finite differences, including chain-rule terms in the sequential RTN map.

Boundedness scans classify 400 updates at radius 20 on a 31-by-21 parameter grid.
'Completed' only means that neither coordinate escaped within that horizon.
Newton's 121-by-121 grid distinguishes converged (14400),
escaped (8), max_steps
(232), and the singular origin (1).
Unresolved basin labels remain -1; they are not assigned a guessed root.

## Reproducibility and artifacts

Follow the exact commands in [Chapter 1 guide](chapter01.md). The scripts record
all fixed seeds and major controls. Stage A uses seed 20261011, logistic r=4,
x0=.618, 1000 burn-in updates, and the explicit parameter grids in its report.
Stage B reports all map parameters, initial states, iteration counts, fit slices,
scales, grid origins, and Newton tolerances.

Two full runs produced byte-identical results for all 26 artifacts: JSON reports,
CSV data, and seven PNG plots. Their SHA256 values are recorded in
[output hashes](chapter01-output-hashes.json). Generated artifacts remain in ignored
outputs/chapter01/. All seven final plots were visually inspected: readable axes,
legends, colorbars, and no clipped labels.

Tests also cover invalid controls, map bounds, histogram endpoints, zero
derivatives, nonfinite/escaped iterations, Newton singularities and exhaustion,
empty eligible pair sets, explicit fitting ranges, standard-map angle wrapping,
Lozi's nondifferentiability, and bounded scan update counts. No external data,
automatic chaos classification, or application/trading claims are introduced.
