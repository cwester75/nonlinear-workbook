# Chapter 1: Nonlinear and Chaotic Maps

All printed pages 1-106 (PDF pages 13-118) were visually reviewed on 2026-10-11.
Stage A was implemented, tested with Chapter 2, and run before Stage B began.
The source PDF remains outside Git. Implementations are independently written.

## Ideas and chapter coverage

The chapter progresses from trajectories and local stability to invariant measures,
frequency analysis, bifurcations, and symbolic descriptions. Multidimensional maps
add tangent spectra, attractor geometry, and basins. Numerical results depend on
initial conditions, finite precision, sample size, and scale/window choices.

Every substantive example family is covered below. Historical C++/Java/Symbolic++
listings and their GUI/file-I/O scaffolding are replaced by Python functions and
headless example scripts; they are not translated line by line.

| Section | Printed pages | Implementation / example | Validation |
|---|---|---|---|
| 1.1 introduction | 1-3 | Orbit, periodic point, and hyperbolicity context | Explanatory definitions |
| 1.1.1 exact/numerical trajectories | 4-20 | Collatz; logistic; Bernoulli; tent; bungalow-tent; Gauss; Fraction iteration; CSV output | Exact book fractions, Collatz bound, numeric collapse |
| 1.1.2 fixed points/stability | 21-22 | stability; logistic fixed points | Stable/unstable/nonhyperbolic derivatives |
| 1.1.3 invariant density | 23-29 | invariant_density; logistic, sine, bungalow, Gauss | Analytical logistic bin masses; tent invariance; density normalization |
| 1.1.4 Lyapunov exponent | 30-34 | scalar_lyapunov; logistic and sine | Logistic ln(2), zero derivative |
| 1.1.5 autocorrelation | 35-37 | autocovariance; moments | Exact small-series sums; logistic variance 1/8 and near-zero later lags |
| 1.1.6 DFT | 38-41 | dft and inverse; cosine and logistic data | Known coefficients, complex round trip |
| 1.1.7 FFT | 42-45 | Independent radix2_fft | DFT/NumPy equivalence, inverse round trip |
| 1.1.8 logistic exponent sweep | 46-47 | logistic_sweep | ln(2) and stable fixed-point benchmarks; sweep diagnostic |
| 1.1.9 bifurcation diagram | 48-50 | Trajectories across r in [2,4] | Multiplier-crossing roots; periodic windows plotted |
| 1.1.10 random maps/density | 51-53 | lcg_step; polynomial_random_step; histogram | Exact short LCG cycle; map bounds |
| 1.1.11 Monte Carlo integration | 54-56 | monte_carlo; book-map averaging | Integral 1-cos(1), IID uncertainty; map mean is diagnostic only |
| 1.1.12 circle/rotation | 57-58 | circle_step lift; rotation_number | Zero-forcing omega, zero-rotation locking |
| 1.1.13 Newton | 59-61 | newton; sine fixed point | Residual-based convergence, singular derivative, iteration limit |
| 1.1.14 Feigenbaum | 62-63 | period_doubling; feigenbaum_ratios for logistic family | Exact first two thresholds; ratio convergence |
| 1.1.15 symbolic dynamics | 64-66 | quadratic_step; symbolic_partition; inverse_symbolic | Forward/inverse composition and boundary ownership |
| 1.2.1 multidimensional introduction | 67-68 | Hyperbolicity/manifold/Hopf context | Theoretical context, not numerical proof |
| 1.2.2 phase portraits | 69-82 | Henon, Lozi, standard, Ikeda, coupled logistic | Analytic Jacobians vs finite differences; book parameter portraits |
| 1.2.3 fixed points/stability | 83-84 | fixed_point_classification; Henon fixed points | Root residuals, eigenvalue classification |
| 1.2.4 Lyapunov exponents | 85-86 | known_lyapunov_spectrum (QR) | Linear expansion/contraction; Henon determinant sum |
| 1.2.5 correlation integral | 87-88 | correlation_integral; fit_scaling | Exact eligible-pair counts; Theiler and sample-size sensitivity |
| 1.2.6 capacity | 89-91 | box_counts; fit_scaling | Line/square dimensions; explicit grid-origin sensitivity |
| 1.2.7 hyperchaos | 92-96 | Coupled logistic QR spectra and parameter sweep | Uncoupled ln(2) exponents; diagnostic two-positive finite-time example |
| 1.2.8 attraction domain | 97-98 | boundedness_scan | Completed/escaped labels, update limits; no asymptotic certification |
| 1.2.9 complex Newton | 99-101 | complex_newton_basins for z^4-1 | Four known roots and singular origin |
| 1.2.10 vector Newton | 102-103 | vector_newton for x^2+y^2=1, x=y | Both roots; linear solves; singular Jacobian |
| 1.2.11 RTN scenario | 104-106 | rtn_step and composed Jacobian; trajectory/spectrum | Sequential update and finite-difference Jacobian; theory remains context |

Feigenbaum's section illustrates three algebraically related quadratic families;
the numerical implementation uses the logistic family, not duplicate solvers for
each conjugate parametrization. The source's general statements about universality,
chaos, and invariant manifolds are not claims established by these examples.

## Public interfaces and conventions

Import from `nonlinear_workbook.chapter01`. The core remains NumPy-only; examples
require the optional Matplotlib environment. Chapter 2 APIs and return values are
unchanged, and chapter01.logistic_map is the same imported generator object.

- Maps are single-step callables. `iterate(step, x0, n=..., burn_in=...)` returns
  `IterationResult(values, status, updates)`. It includes f^burn_in(x0) as the first
  observation, then n-1 updates. Escape/nonfinite outcomes return a partial result.
  `exact=True` retains Python int/Fraction objects. Collatz takes positive Python
  integers and always has an explicit observation bound.
- The existing logistic_map generator retains its older Chapter 2 convention:
  it omits the initial state and returns n updates after burn-in. Use iterate for
  Chapter 1's initial-inclusive convention. Neither API is silently changed.
- Bernoulli uses modulo one; tent's midpoint belongs to the descending branch.
  Bungalow branch boundaries belong to the branch on their right. Exact arithmetic
  is provided by passing Fraction states and parameters, not by converting floats.
- DFT/FFT forward transforms divide by N; inverses do not. Frequency indices are
  sample bins, not physical frequencies. radix2_fft requires a power-of-two length.
  DFT has quadratic cost and is intended for small reference examples.
- Histograms use supplied increasing edges, include the final right endpoint,
  and normalize by retained samples. They report counts so exclusions are visible.
  Compare singular invariant densities through integrated bin masses.
- Autocovariance demeans with the full-series mean and divides by N-lag. It returns
  lags, covariances, and pair counts, not a normalized Pearson coefficient.
- Scalar exponents average natural logs of absolute derivatives per sample_interval.
  Zero derivative returns negative infinity. Known Jacobian QR returns exponents,
  running estimates, usable/accumulated steps, and status; a singular update stops.
  If no post-burn-in QR updates exist, it raises ValueError. QR ordering is retained.
  These Jacobians come from equations, unlike Chapter 2's fitted local Jacobians.
- Standard-map state is [momentum, angle]; only angle wraps to [0,2*pi). Its Jacobian
  describes the smooth angular lift. Lozi's Jacobian raises at x=0. RTN updates x
  first and uses that new x to update y; its Jacobian differentiates the composition.
- `correlation_integral` returns radii, values, raw counts, and eligible unordered
  pairs. Distance equality counts; temporal exclusion acts on sample row indices.
  Duplicate states count if temporally eligible. Streaming computation uses O(N^2)
  work without an N-by-N allocation, so examples keep bounded samples.
- `box_counts` uses half-open grid cells floor((point-origin)/scale) with a supplied
  origin. `fit_scaling` requires an explicit contiguous fit_slice and positive
  values; inverse_scale=True fits box counts against log inverse scale. It returns
  slope, intercept, R-squared, and selected indices. No scaling window is inferred.
- Newton requires an explicit derivative/Jacobian and residual tolerance; it reports
  convergence, singularity, escape, nonfinite values, or max_steps. Vector Newton
  solves linear systems. Complex basin labels are 0..3 for [1,-1,i,-i]; -1 means
  unresolved and is accompanied by statuses and iteration counts.
- Monte Carlo integrates over [0,1]. Mean and hit/miss methods use seeded IID NumPy
  samples; hit/miss requires 0<=f<=1. Standard errors apply to that sampler. The
  polynomial map and LCG are demonstrations, not interchangeable IID samplers.
- Circle rotation uses an unwrapped lift and displacement after burn-in. Boundedness
  scans report outcomes over exactly horizon updates and a supplied escape radius;
  'completed' is not proof of attraction or long-term boundedness.
- Symbol boundaries belong to the right interval. inverse_symbolic composes the
  supplied L/R branches in reverse order for 1-r*x^2, with an explicit terminal state.
- Period-doubling solves orbit closure plus multiplier=-1, requires continuation
  guesses, rejects lower-period roots, and reports convergence. It is a local solver,
  not a global root enumerator or an arbitrary-precision universality proof.

## Reproduce both stages

Reference: CPython 3.12.14, NumPy 2.3.5. requirements-examples.txt freezes all
resolved example dependencies, including Matplotlib 3.11.2. Core validation pins
remain in requirements-validation.txt. From the repository root in PowerShell:

```powershell
python -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-examples.txt
.\.venv-examples\Scripts\python -m pip install --no-build-isolation --no-deps -e .
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/chapter01/stage_a.py
.\.venv-examples\Scripts\python examples/chapter01/stage_b.py
```

Alternatively set `$env:PYTHONPATH='src'` instead of installing the package.
Each example accepts `--output PATH`. Defaults are outputs/chapter01/stage_a and
outputs/chapter01/stage_b. These ignored folders contain report.json, numerical
CSV files, and three/four PNG figures respectively. Reports contain versions,
seeds, parameter choices, statuses, curves, and diagnostics. Nonfinite numerical
values become JSON null; consult the accompanying statuses and original CSVs.

See [validation](chapter01-validation.md) and [book corrections](book-corrections.md).
