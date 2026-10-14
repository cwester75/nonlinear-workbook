# Chapter 8 validation

CPython3.12.14, NumPy2.3.5, Matplotlib3.11.2; unchanged frozen dependencies.
Seed20261014. All103 tests pass: the previous95 plus8 Chapter8 tests.
All examples run, and eight headless PNGs were visually inspected. Reproduction
commands are in chapter08.md. Generated outputs and source PDF remain outside Git.
The two complete final runs produced25 byte-identical JSON/CSV/PNG artifacts.

## Analytical acceptance

- Cantor Fractions give exactly2^n intervals, length3^-n, retained measure(2/3)^n,
  disjoint interiors and known first-stage endpoints.
- Dragon gives2^n segments with fixed endpoints0/1 and lengths2^(-n/2) to1e-12.
- Gasket3^n triangle counts, side2^-n, and total area ratio(3/4)^n are checked.
- Koch4^n segments per original edge, perimeter(4/3)^n and outward snowflake area
  A_n/A_0=1+3/5*(1-(4/9)^n) are checked to12 decimal places at bounded depths.
- Escape fixtures include c0,-1,-2, escaping c.5, strict radius equality, last
  update escape, initial Julia escape and nonfinite arithmetic. Independent Python
  scalar loops agree with vectorized counts/flags for both image types.
- Weierstrass three-term expression(.5 sin4x+.25 sin16x+.125 sin64x), tail bound,
  large floating powers and nonfinite/invalid controls are checked.
- Lévy normalization, exact accounting, reproducibility, characteristic(0)=1;
  populated level counts (expected>=25) remain within5 binomial standard errors.
- Combinatorial scaling fits reproduce log2/log3,log3/log2,log4/log3 within1e-12.
  Numerical point-cloud slopes are sensitivity observations, not acceptance targets.

## Examples and observed results

Cantor depths0..6, dragon0..12, gasket0..6 and Koch0..5 are shown. IFS comparisons
use20,000 retained chaos-game points after100 burn updates, equal probabilities,
initial(0,0). Deterministic gasket uses depth6 and all three initial vertices.
Euclidean contraction norms are(.5,.5,.5). Printed coordinates are kept separately.

Snowflake depth5 has perimeter12.64197531 and area.6883148614; starting area is
sqrt3/4 and infinite-area limit(8/5)*sqrt3/4=.6928203230.

| Escape image | Grid | Not escaped at32 | At128 | At512 |
|---|---|---|---|---|
| Mandelbrot | 600x400,[-2,1]x[-1,1] | 65204 | 61194 | 60313 |
| Filled Julia, c=-.12256117+.74486177i | 500x500,[-1.5,1.5]^2 | 36520 | 36174 | 36174 |

No nonfinite pixels occur in these runs. Equal Julia masks at128/512 do not prove
membership or exact boundary resolution. Image grids include endpoints. Escape
colors share a log(1+updates) scale across budgets; black denotes unresolved
finite points within the iteration budget.

Weierstrass s1.5,lam4: terms3/6/9 have highest angular frequencies64/4096/262144
and tail bounds.125/.015625/.001953125. Coarse1025 samples on[0,2*pi] alias retained
high frequencies. The six-term reference uses131073 points,32 samples per shortest
retained wavelength; its spacing is2*pi/131072. This resolves the finite sum,
not the infinite series.

The10,000-step Lévy example uses lam=b=2,max_level12. Removed infinite-law mass is
1/8192, and the retained law is renormalized. Maximum empirical real-characteristic
error over101 points on[0,1] is about.00751; the empirical imaginary part fluctuates
around0. Rare high levels are underpopulated and are not subjected to Gaussian
frequency acceptance checks.

## Dimension sensitivity

240 records: Cantor depths6/9, gasket4/6, Koch3/5, line and square controls;
requested sample counts512/2048 are capped by available unique vertices/endpoints.
Samples are drawn without replacement. The square starts from a65x65 grid; line
from2048 points. Noise std0,1e-4,1e-2 is added in normalized coordinate units.

Box scale families2^(-1..-8) and3^(-1..-6) use origins0 or.137 in every coordinate,
always fit slice[1:5]. Correlation uses radii2^(-8..-1), Theiler0, slice[2:6] and
at most2048 points. Full pair counts/eligible totals, curves and fit residuals are
retained. Ten zero-count scaling slices are reported unavailable instead of moved.

| Object | Box-fit range | Correlation-fit range | Similarity/control reference |
|---|---|---|---|
| Cantor | .5340.. .8668 | .6347.. .7928 | .63092975 |
| Gasket | .2404..1.5750 | 1.5755..1.8028 | 1.58496250 |
| Koch | .3001..1.3588 | 1.2478..2.1459 | 1.26185951 |
| Line | .8603..1.4022 | .9737..1.2113 | 1 |
| Square | .4709..1.7795 | 1.8411..2.2166 | 2 |

Ranges mix intentionally different depths, sample counts, noise and grids; they
are not confidence intervals. Fine boxes saturate at the point count; coarse
boxes have strong boundary/origin effects. Small finite Koch/gasket point clouds
cannot resolve their infinite limit at arbitrarily fine scales. Added noise
changes small-scale geometry. No automatic optimum or Hausdorff estimate is claimed.
