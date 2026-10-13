# Chapter 2: Time Series Analysis

Source: printed pp.107-134, PDF pp.119-146. All 28 pages visually reviewed
2026-10-10. Algorithms are independently implemented; the PDF remains outside Git.

## Key ideas and coverage

| Section / pages | Idea and book example | Implementation | Validation |
|---|---|---|---|
| 2.1 / 107 | Reconstruction and quantitative descriptors; autocorrelation and fractal dimensions refer back to Chapter 1 | Context only | Not Chapter 2 algorithms |
| 2.2 / 108-109 | Pearson linear correlation; positive/negative proportional vectors | correlation | Affine and constant-input tests |
| 2.3.1 / 111-112 | Local Jacobian least squares and orthogonal tangent propagation | local_lyapunov_spectrum | Nonsymmetric linear map, expansion/contraction, rank failure |
| 2.3.2 / 113-117 | Nearest-neighbor and neighborhood one-step divergence; logistic map | direct_lyapunov | Both variants against ln(2) |
| 2.3.2 / 118 | Wolf trajectory evolution with aligned neighbor replacement | wolf_lyapunov | Logistic ln(2), controlled doubling, time accounting, failed replacement |
| 2.3.2 / 119-121 | Kantz neighborhood divergence; logistic map | kantz_curve | Known growth, bounds, noise and fit-window sweeps |
| 2.4 / 122-129 | Rescaled-range scaling; logistic map demonstration | hurst_rs | IID Gaussian baseline, AR(1), logistic scale curves |
| 2.5 / 130-134 | LZ exhaustive-history parsing; binary examples and logistic symbolic partition | lz_complexity | Counts 2, 3, 6; seeded random and logistic symbols |

## Reading the estimates

Pearson measures a linear relationship, not independence. Delay reconstruction
uses past samples: row t is [x[t], x[t-delay], ...]. Neighborhood distance is
Euclidean and temporal exclusion is measured in original sample indices.
Scaling the observations changes distance thresholds; choose them explicitly.

Lyapunov estimates measure logarithmic separation growth. The logistic derivative
reference uses the known equation; direct, Wolf, Kantz, and local-Jacobian methods
use observations. All new exponent APIs use natural logs per sample_interval.
Wolf max_steps is a per-segment limit, not the total trajectory duration.
Replacement angle is directed and measured in radians (0 to pi). Failed
replacement retains the evolved pair until the next segment or boundary.

The Jacobian estimator solves row displacements Y B = Z and propagates B.T.
QR direction order is retained, not sorted. qr_burn_in updates the tangent basis
without accumulating exponents. fit_ranks and condition_numbers include failed
fit attempts; usable_steps counts successful QR updates. Partial results include
a termination reason; failure before any post-burn-in update raises ValueError.
Numerical full rank does not imply a well-constrained tangent map. In particular,
two-dimensional reconstruction of the logistic map can produce spurious transverse
exponents: its observed trajectory lies on a lower-dimensional curved manifold.
Inspect conditioning and parameter sensitivity rather than interpreting all
returned exponents as a physical spectrum.

R/S fits log mean rescaled range against log block size. IID finite-sample bias
and short-range AR(1) correlations can increase the estimate. H near 0.5 does not
prove independence, and an elevated H does not establish long memory.

LZ counts new exhaustive-history phrases with overlapping copying. For the book
example the parse is 0 | 001 | 10 | 100 | 1000 | 1. The terminal phrase counts
as a phrase even if it is reproducible. Normalization uses the observed alphabet;
a constant sequence has a degenerate one-symbol alphabet and returns zero
normalized complexity by convention. Binary-alphabet normalization would give a
small nonzero value instead. Finite-length normalized values can exceed one.

## Reproduce

From the repository root in PowerShell, using CPython 3.12.14:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-validation.txt
.\.venv\Scripts\python -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python examples/chapter02.py
.\.venv\Scripts\python examples/chapter02/research_validation.py
```

The last script prints JSON containing parameters, runtime versions, curves,
reference counts, ranks, running spectra, and sensitivity results. It selects no
optimal parameters. See [corrections](book-corrections.md) and
[validation](validation.md). The original example path and APIs remain available.
No estimator independently proves chaos or supplies anomaly/trading thresholds.
