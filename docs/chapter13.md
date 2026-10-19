# Chapter 13: Fuzzy Sets and Fuzzy Logic

Source: printed pp.495-576 / PDF pp.507-588. All 82 pages were visually reviewed
on 2026-10-16, with suspected corrections rechecked at higher resolution.
No source PDF is redistributed. Import `from nonlinear_workbook import chapter13 as c`.

| Section / pages | Computational coverage | Reference validation |
|---|---|---|
| 13.1, 495-501 | Finite membership sets, rectangles/triangles/trapezoids, support, normalization, inclusion, alpha cuts, cardinality | Membership endpoints; crisp reductions; RAM and speed fixtures; sampled quasiconcavity |
| 13.2.1-2, 502-507 | Logical/algebraic/bounded/gamma operators, powers, Cartesian minimum | Exhaustive Boolean cases; De Morgan/distributive identities; gamma endpoints |
| 13.2.3, 508-509 | Discrete/continuous-grid centroid, min/max grade with ties, alpha and top-beta cuts | p.509 centroid 3.34029850746; asymmetric-triangle refinement |
| 13.2.4-6, 510-514 | Small/large ramps, logistic/peak-one Gaussian, Gaussian max-min compatibility, hedges/contrast, fuzzy entropy | Analytical compatibility; entropy complement symmetry and maximum; spread examples |
| 13.2.7, 515-537 | Explicit NumPy finite sets replace C++ storage/operators; interpolation, clipping, comparisons and means | Independently calculated p.536 input and p.537 output fixtures; safe endpoint interpolation |
| 13.2.8, 538-545 | Shoe and pricing decisions; grid/operator sensitivity | Shoe indices 3 then 2; book minimum-aggregation pricing 3649 then 3146 Rand |
| 13.3.1-2, 546-551 | Bounded exact-key extension principle; square/add/product; max-min vectors | Independent nested loops, Fraction collisions and nonconvex output tables |
| 13.3.3-5, 552-564 | LR intervals/spread functions; triangular numbers; finite alpha-cut arithmetic; separately named LR approximations | Exact interval endpoint enumeration; negative scaling; reciprocal corrections; approximation refinement |
| 13.3.6, 565 | Power and ideal-gas examples | Power approximation (1200,124,124) watt; exact support and rational-key gas extension |
| 13.4.1-2, 566-571 | Explicit rule construction, min/max/product antecedents, clip/product implication, max aggregation, centroid/maximum/singleton outputs | Scalar reference oracle, rule histories, uncovered inputs and zero mass |
| 13.4.3, 572-573 | Pendulum force-inference sweep | Rule coverage and implication sensitivity; no missing plant invented |
| 13.4.4, 574-576 | Language predicates on the listing's 200-point grid | First two printed grades to rounding; grid sensitivity and reported malformed third input |

Definitions, applications mentioned without sufficient equations, and general
claims about control remain explanatory. Membership choices alone do not prove
closed-loop stability or demonstrate calibrated decision confidence.

## Public conventions

`FuzzySet(universe, membership)` copies finite data onto a strictly increasing
numeric universe. Grades must lie in [0,1]; mismatched universes are rejected.
Use `at(x)` explicitly to resample by piecewise-linear interpolation. Outside
queries raise unless `outside='zero'` is supplied. Arrays are read-only.

`alpha_cut(0)` includes the entire supplied finite universe, as its definition
uses >=. `support` uses >0. Continuous `Triangular.cuts` and linear
`LRInterval.cuts` instead use the support closure at level zero; that convention
avoids claiming that a continuous zero cut is a bounded subset of all real numbers.
`sampled_quasiconcave` checks the stored grades only, not arbitrary unsampled geometry.

`relative_cardinality` divides by the number of universe points. It is not a
coordinate integral and changes with sampling. Normalize only deliberately;
zero-height normalization raises. Discrete centroids use sums; `mode='continuous'`
uses trapezoidal samples of the numerator and denominator on the supplied grid.
It does not integrate an interpolated polynomial numerator exactly. Zero mass
raises. Extrema accept first/last/mean ties; top-beta uses a stable smaller-coordinate
first tie rule. Rectangle boundaries are closed, triangular peaks are included,
and zero-width triangles represent crisp singletons. Degenerate shoulders are supported.

`combine` names operators explicitly; `gamma` is required only for the weighted
operators and defaults to .5. Gamma-weighted AND/OR generally are not associative.
A logical minimum aggregation reproduces the pricing listing; varying gamma is a
separate sensitivity experiment. `hedge` requires a finite positive exponent;
the limiting infinite-exponent linguistic label remains explanatory. Gaussian
and logistic normalized differences retain subtract-first cancellation accuracy,
with a scale-first fallback for finite subtraction overflow. Genuine tail
saturation is retained without invalid arithmetic. No global
hedge-name defaults hide the disagreement between the book's tables and code.
`entropy` computes the sum of binary membership entropies, base >1, with 0log0=0;
these are not probabilities normalized across the universe.

## Arithmetic

`extension(universes, memberships, function, max_combinations=...)` computes the
sup-min result by bounded enumeration. Exact integer/Fraction keys are preserved;
float equality is literal, with no automatic rounding/binning. NumPy signed and
unsigned integer universe values are converted to Python integers before callbacks
to avoid fixed-width overflow; Fraction values retain their original exact type.
Callbacks remain responsible for any explicit narrowing casts they introduce. Output can be
nonconvex and need not be a fuzzy number. All supplied map outputs must be finite.

`CutIntervals(levels, bounds)` stores finite nested intervals at explicitly
supplied strictly increasing alpha levels. Arithmetic returns the full interval
curve, not a fitted triangle. Four corner products/quotients determine interval
bounds. Division rejects any supplied denominator cut containing zero; callers
must include level zero to check the whole bounded support. Linear LR intervals
have direct cuts; nonlinear LR shapes expose membership, with callers supplying
other cut intervals explicitly. Gaussian/exponential zero cuts are unbounded and
are not silently truncated into finite intervals.

`Triangular` has mode and nonnegative left/right spreads. Addition/subtraction
and signed scaling preserve the triangular representation. `approximate_product`
uses first-order flank propagation with derivative signs; it is not exact
extension multiplication. `approximate_reciprocal` matches mode/support endpoints
but compresses curved membership into a triangle; `approximate_divide` compounds
these approximations. It rejects supports touching zero. A crisp operand has zero
spreads, including an explicit membership-one peak without division by zero.

Extension and interval arithmetic assume operands can vary independently in the
Cartesian possibility set. They do not preserve unspecified shared-variable
dependence: an expression such as X-X requires additional information to force
an identically zero result. Electrical power and gas examples record this assumption.

## Rules and computation bounds

`Rule` stores named (variable, term) antecedents, connective, consequent and weight.
`infer` takes explicit inputs, term functions/FuzzySets, aligned output FuzzySets,
and rules. Results contain every antecedent grade, weighted firing strength,
activated consequent, aggregate membership, conventions, status and crisp value.
No rule firing and zero output mass return distinct statuses with value None.
Outputs are not probabilities and are not automatically normalized.

The pendulum example reports degrees and arbitrary force units. Its finite output
shoulders and grid are documented choices. Temperature membership shapes are a
synthetic extension: error = current minus setpoint (Celsius), output = duty percent.
Neither example adds plant dynamics or claims stabilizing feedback.

Finite sets/cut arrays and retained rule histories have a one-million-element
bound. Extension and Cartesian constructions check explicit combination/element
limits on sized inputs before array conversion or expanding products (at most 32 input axes). Inputs are validated and copied; no automatic
clipping, learning, classification or outlierhunter runtime integration is added.

## Reproduction

Use the existing CPython 3.12.14 environment and frozen requirements-release.txt;
NumPy remains the only core numerical dependency. Matplotlib remains examples-only.

```powershell
$env:PYTHONPATH='src'
.venv-examples/Scripts/python.exe -m unittest discover -s tests -v
.venv-examples/Scripts/python.exe examples/chapter13/validation.py
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --output outputs/chapter13-repeat
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --smoke --output outputs/chapter13-smoke
```

The full demonstration writes results.json, four CSVs and four headless PNGs
under ignored outputs/chapter13. Runtime versions, seed 20261016, grids, policies,
units and model assumptions accompany the histories. See
[validation](chapter13-validation.md) and [source corrections](book-corrections.md).
