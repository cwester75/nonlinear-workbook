# Chapter 13 validation

Reference run: 2026-10-16, CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2,
Windows, existing requirements-release.txt environment. Core dependencies and
all prior public APIs are unchanged. Seed: 20261016 for bounded operator checks;
other Chapter 13 fixtures are deterministic and use no random draws.

## Acceptance budgets established before recording results

- Exact equality: Boolean reductions, extension output keys/collision maxima,
  Fraction fixtures, explicit book shoe/pricing indices and availability statuses.
- Absolute 1e-12: bounded membership landmarks, lattice identities, entropy,
  analytical Gaussian compatibility and finite interval endpoint arithmetic.
- Centroid quadrature: asymmetric triangular membership on [0,3], peak 1,
  analytical centroid 4/3. At 1601 points error <2e-6 and at least a factor 100
  smaller than the 101-point error. These are finite-grid acceptance bounds,
  not exactness claims for trapezoidal quadrature.
- Printed grades rounded to six decimals: compare to within 1e-6 only when
  the source input and convention are unambiguous. The malformed p.576 input
  is reported with two explicit interpretations instead of forced agreement.
- Approximation sensitivity: halving both positive product spreads in the
  chosen fixture must reduce its endpoint error by four, within 1e-12.
  No general nonlinear approximation error guarantee follows from that fixture.

## Observed results

All 170 tests pass from both the source checkout and an extracted built wheel:
the previous 144 plus 26 Chapter 13 tests. The wheel includes all four new package
modules; no build/runtime dependency was added. Local pip check passes. Coverage includes
invalid/nonfinite/complex inputs, zero-height and zero-mass failures, mismatched
universes, unsupported inference conventions, crisp zero spreads, negative
scaling, singular division, overflow and preallocation bounds. Independent
nested-loop finite extensions, explicit corner arithmetic and scalar inference
oracles support the numerical comparisons.

| Check | Analytical/book target | Observed |
|---|---|---|
| p.509 discrete centroid | 3.3402985074626863 | agreement within 1e-12 |
| p.537 sampled continuous centroid | 3.260869565217391 | 3.2608695652173916 |
| Same grades, discrete centroid | sum(x*mu)/sum(mu) | 3.428571428571429; intentionally different |
| p.537 cardinality | 2.8 | 2.8 |
| Corrected relative cardinality | 2.8/5 | .56; printed .7 divides by coordinate width |
| Fuzzy entropy / contrasted entropy | independent binary-entropy sums | 3.3857573786840622 / 2.6131212958535337 |
| Shoe winners (zero-based) | 3 then 2 | 3 then 2 |
| Pricing, minimum aggregation, 3001 integer prices | 3649 / 3146 Rand | 3649 / 3146 |
| p.576 first two Java grades, 200 points | .316121 / .622325 (rounded) | .3161205300 / .6223245121 |
| Product (2,.5,.5)*(3,.5,.5), compressed triangle | (6,2.5,2.5) | exact parameter agreement; not exact membership |
| Corrected approximate quotient | (2/3,11/42,.3) | (.6666666667,.2619047619,.3) |
| Power approximation | (1200,124,124) watt | agreement; exact independent-support endpoints are 1078/1326 watt |

The exact square/add/product finite tables reproduce pp.547-549. The product
output is nonconvex and is retained, not forced into a triangular fuzzy number.
The gas example enumerates 125 combinations using Fraction inputs and exact
rational output keys. These possibilities assume independent input variation;
no statistical uncertainty or physical dependence is inferred.

## Sensitivity results

- Asymmetric centroid errors at 101/401/1601 points:
  3.3003300e-5 / 2.0781380e-6 / 1.3012700e-7.
- Product approximation maximum alpha-cut endpoint error at input spreads
  .5/.25/.125: .25/.0625/.015625. The missing quadratic term explains this
  fixture's factor-four reduction; reciprocal compression remains asymmetric.
- Minimum-aggregation price grids with 101/401/1601/3001 points select
  3650/3650/3650/3649 Rand in scenario 0 and
  3140/3147.5/3145.625/3146 Rand in scenario 1. Discrete maxima need not move
  monotonically under refinement.
- At 3001 points, changing gamma from 0 to .5 to 1 selects
  3397/3500/3649 Rand (scenario 0), and 3339/3304/3146 (scenario 1).
  Non-minimum aggregations are left-associated and order-dependent; these are
  alternative supplied decision models, not discovered optimal settings.
- Language predicates use 101/200/401/1601 points. The 200-point fixture matches
  the first two printed outputs. The third extend token is printed as `0.0.5`:
  input zero produces zero Java grade; input .5 produces .4728483329. Neither
  confidence grade is a calibrated class probability.
- Pendulum output histories retain 201 angle inputs per implication strategy.
  Of 402 clip/product cases, 44 have no firing outside/at the open supports.
  They retain None outputs instead of substituting zero control. Output-grid
  truncation and shoulder definitions affect centroids. This is force inference,
  not a simulated or proved stabilizing controller.

Four headless plots were visually inspected: memberships/hedges/spreads,
exact-versus-compressed arithmetic, decision sensitivity, and rule-output curves.
Full Chapter 13 examples were repeated; all nine JSON/CSV/PNG files are byte-identical.
No environments, generated plots or source PDF are committed.

## Reproduction and regression scope

```powershell
$env:PYTHONPATH='src'
.venv-examples/Scripts/python.exe -m unittest discover -s tests -v
.venv-examples/Scripts/python.exe examples/chapter13/validation.py
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --output outputs/chapter13-repeat
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --smoke --output outputs/chapter13-smoke
.venv-examples/Scripts/python.exe -m pip check
```

The full regression also reruns the existing executable example entrypoints:

```powershell
$examples = @(
 'chapter01/stage_a.py', 'chapter01/stage_b.py',
 'chapter02.py', 'chapter02/research_validation.py',
 'chapter03/validation.py', 'chapter04/validation.py',
 'chapter05/validation.py', 'chapter06/validation.py',
 'chapter07/validation.py', 'chapter08/validation.py',
 'chapter09/validation.py', 'chapter10/validation.py',
 'chapter11/memory.py', 'chapter11/kohonen.py', 'chapter11/feedforward.py',
 'chapter12/search.py', 'release_smoke.py', 'diagnostic_sensitivity.py'
)
foreach ($example in $examples) {
 .venv-examples/Scripts/python.exe "examples/$example"
 if ($LASTEXITCODE -ne 0) { throw "Example failed: $example" }
}
.venv-examples/Scripts/python.exe examples/chapter09/puzzle.py --backend TkAgg --smoke
```

The optional interface's callbacks passed the TkAgg smoke check; no unattended
interactive game is left running. common.py files are helper modules, not example
entrypoints. CI adds the bounded Chapter 13 smoke command alongside the complete
unit suite; hosted CI is not claimed to have run. Existing v0.1.0 and outlierhunter
pins remain untouched, and no external dataset or detection threshold is added.

All 18 existing noninteractive entrypoints completed successfully, followed by
the full Chapter 13 demonstration and its smoke command. The optional TkAgg
interface smoke also passed. SHA256 comparisons found all 335 comparable legacy
artifacts unchanged, including all 94 Chapter 7 files. Common numerical paths
therefore retain their saved deterministic results; the new chapter's nine
artifacts independently reproduced byte for byte.
