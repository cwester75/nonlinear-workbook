# Chapter 8: Fractals

All printed pp.275-298 / PDF pp.287-310 were visually reviewed before implementation.
The printed gasket fixture (p.284), Julia parameter (p.294) and Lévy formulas
(p.297) were rechecked at full resolution. Core remains NumPy only; existing
chapter APIs and dependency pins are unchanged.

| Section / printed pages | Executable coverage | Validation |
|---|---|---|
| 8.1, 275-276 | Explicit finite-curve fitting; reuse of Chapter1 box/pair curves | Exact combinatorial scaling ratios; noise/grid/sample sensitivity |
| 8.2.1, 277 | Deterministic affine IFS, seeded chaos game, matrix contraction norms | One-step fixtures and repeatable map choices |
| 8.2.2, 278-280 | Exact Fraction Cantor intervals and explicit float endpoints | Counts, lengths, measure and disjoint interiors |
| 8.2.3, 281-283 | Ordered complex dragon polyline | Endpoints, segment count and lengths |
| 8.2.4, 284-286 | Gasket triangles, normalized and printed affine fixtures | Triangle count, side and area scaling |
| 8.2.5, 287-289 | Open Koch curve and closed outward snowflake | Perimeters, orientation and finite-depth area formula |
| 8.3, 290-292 | Mandelbrot escape arrays | Scalar/vector agreement, strict radius boundary and analytical orbits |
| 8.4, 293-295 | Quadratic filled-Julia escape arrays | Initial escape, c0 fixture and scalar/vector agreement |
| 8.5, 296-298 | Finite Weierstrass sums and bounded discrete Lévy flight | Three-term expression, tail bound, probability/characteristic normalization |

Hausdorff-dimension definitions, IFS fixed-set existence, Julia-set theory and
infinite-series regularity remain explanatory. Finite fitted dimensions are not
Hausdorff computations; no current theorem status is inferred from historical prose.

## Construction APIs

Import `from nonlinear_workbook import chapter08 as c`.

- `cantor_intervals(depth)` returns a tuple of closed Fraction intervals on[0,1].
  `cantor_endpoints(intervals)` explicitly converts to sorted unique float columns.
- `deterministic_ifs(matrices,translations,initial_points,depth=...)` applies each
  affine map to every point in map-major order. Duplicate points remain present.
  `chaos_game` requires probabilities summing to1, initial_point, samples, burn_in
  and seed; its first retained point follows burn_in+1 map applications. Returned
  map_indices correspond to retained updates; seed/burn reproduce earlier updates.
  `contraction_norms` reports Euclidean operator norms, without enforcing contraction.
- `dragon(depth)` returns complex vertices starting0 and ending1, with2^depth
  segments. It applies (1+i)z/2 and 1+(-1+i)z/2, reversing second-branch traversal.
- `sierpinski_triangles(depth,vertices=...)` returns shape(3^depth,3,2). Default is
  an equilateral triangle of side1. `gasket_ifs(printed=True)` preserves p.284:
  all matrices .5I and translations(1,1),(1,50),(50,50). Its fixed triangle has
  vertices(2,2),(2,100),(100,100), a right-triangle coordinate version of the gasket.
- `koch_curve` is open with explicit start/end and side+1(left)/-1(right).
  `koch_snowflake` closes a supplied three-vertex triangle, choosing outward
  refinement from its orientation. The standard equilateral default supplies the
  documented perimeter/area benchmarks; arbitrary triangles need not share those
  formulas and may produce intersecting boundaries. `polygon_area` uses shoelace
  area (signed on request); `polyline_length` sums ordered segment lengths.

Depth zero is the unrefined object. max_elements defaults to1,000,000; depth is
also bounded by64. Exponential construction sizes are checked before generation.
The element budget bounds principal output sizes (interval endpoints, triangle
coordinates, polyline vertices, IFS coordinates), not exact peak process memory.
Escape functions instead limit the number of grid points. Invalid controls,
degenerate triangles and nonfinite iterations raise ValueError.

## Series and finite random walks

`weierstrass(x,s=...,lam=...,terms=...)` returns values, retained angular
frequencies/amplitudes and the uniform omitted-series bound. With
r=lam^(-(2-s)), the bound is r^(terms+1)/(1-r). Require1<s<2 and lam>1;
reject frequency/phase overflow and an unresolved denominator. Frequencies use
floating powers explicitly, avoiding silent integer overflow. Finite sums are
smooth; their rough appearance is not a proof of infinite-series regularity.

`levy_flight` requires steps,lam,b,max_level,seed; lam,b>1. It returns increments,
positions (including initial0), levels, signs, probabilities, jump sizes and
omitted_probability=lam^(-(max_level+1)). Retained levels are renormalized, not
clipped. `levy_distribution` exposes this finite law; `levy_characteristic` returns
sum_j p_j*cos(k*b^j). Signs are equiprobable, so the population imaginary part is0.
The selected example lam=b=2 is a valid finite distribution but departs from the
book's asymptotic condition b>lam>1. No untruncated stable-process claim is made.

## Escape arrays and fitted curves

`mandelbrot(grid,max_iterations=...)` uses z0=0 and strict abs(z)>2.
`filled_julia(grid,c=...,max_iterations=...)` uses radius max(2,abs(c)). Grid shape
is preserved, including scalar inputs. Counts record actual updates; initial Julia
escape has count0. Escaped points stop updating. nonfinite is a separate flag and
is excluded from escaped; final_values preserve the last calculated state.
Unescaped finite points are only **not escaped within budget**. Filled-Julia
images approximate the filled set; its boundary is the Julia set. No membership
or dimension certificate is inferred from a raster mask.

Chapter1 `box_counts` and `correlation_integral` are re-exported, not duplicated.
`fit_dimension(scales,values,fit_slice=slice(start,stop),kind=...)` requires at least
two positive fitted values and returns slope, intercept, residuals, indices and
the entire raw/log curve. Box fits use log(1/scale); correlation fits log(radius).
Zero values outside the selected slice remain in the full curve, with -inf logs.
No automatic fit selection or clipping to an expected dimension.

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter08/validation.py
.\.venv-examples\Scripts\python.exe examples/chapter08/validation.py --output outputs/chapter08-repeat
```

Use the unchanged requirements-examples.txt environment. Outputs are ignored
JSON/CSV/PNG files. See [validation](chapter08-validation.md) and [corrections](book-corrections.md).
