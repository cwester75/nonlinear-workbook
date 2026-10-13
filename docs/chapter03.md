# Chapter 3: Autonomous Systems in the Plane

Printed pp.135-152 / PDF pp.147-164 were visually reviewed in full before implementation.
The chapter is independently implemented in `nonlinear_workbook.chapter03` using
Chapter10 solvers; core dependencies remain NumPy only.

| Section / printed pages | Coverage | Reference checks |
|---|---|---|
| 3.1, 135-137 | Continuous-time eigenvalue classification, Van der Pol equilibrium, radial equilibrium circle | Canonical nodes, defective/star nodes, saddle, spiral, center and zero eigenvalues; equilibrium residuals |
| 3.2, 138-140 | Homoclinic oscillator, Jacobian, equilibria, energy and analytic separatrix | Independently differentiated reference and integration over shifted [-10,10] |
| 3.3, 141-143 | Dimensionless pendulum, Jacobian, equilibria, energy; unwrapped rotations | Small-amplitude period, invariant drift, symplectic/Fehlberg comparison |
| 3.4, 144-149 | Van der Pol, printed sine damping and separately named listing-derived update | Jacobians; finite-horizon crossing/recurrence diagnostics; source mismatch documented |
| 3.5, 150-152 | Positive-parameter Lotka-Volterra, Jacobian, equilibria and logarithmic invariant | Positive trajectories, invariant drift and solver refinement |

## APIs and conventions

The six RHS functions accept `(t,state)` with two finite coordinates. Parameters
are keyword-only: `mu` for `van_der_pol` and `sine_damping`, positive `a,b,c` for
`lotka_volterra`. Each has a corresponding `_jacobian` function with the same
signature. Autonomous systems still validate the supplied time.

`equilibria(system, ...)` returns finite fixtures. For pendulum, callers supply
integer multiples of pi; for the radial system, callers supply angles on the
equilibrium circle. These finite arrays do not exhaust infinite equilibrium sets.
Lotka-Volterra returns both the boundary origin and the positive interior equilibrium.
Its RHS permits finite coordinates so solver stages can be evaluated; logarithmic
invariants reject nonpositive populations explicitly.

`classify_planar(matrix, tolerance=1e-10)` returns eigenvalues, trace, determinant,
discriminant, stability and geometric type. Stability refers to continuous linear
dynamics and uses real parts, unlike Chapter1 map multipliers. Saddle means unstable;
imaginary or zero-real-part eigenvalues mean nonhyperbolic. A linear center is not
a conclusion about a nonlinear system. Repeated scalar matrices are star nodes;
nonscalar repeated-root matrices are defective nodes. Tolerance is absolute.

`homoclinic_energy`, `pendulum_energy` and `lotka_volterra_invariant` are scalar
diagnostics. Pendulum time is scaled by sqrt(g/L), velocity is d(angle)/d(scaled time),
and angle is in radians. Energy uses p^2/2+1-cos(q). Do not wrap angles before
integration. Lotka invariant is c*log(x)+a*log(y)-b*(x+y), retaining the sign used
in the book's a=b=c=1 example.

`homoclinic_reference(t)` returns shape (...,2) for the positive time-centered
separatrix. It uses a stable sech representation of 3/(2+sqrt(13)*cosh(t/sqrt(2))).
The analytic expression has been differentiated and checked against the ODE and
zero energy. Numerical comparison shifts forward solver time by ten units.

`van_der_pol_listing_step(state, step=..., mu=...)` reproduces p.148 separately:
drift x first, then multiply velocity by an exponential frozen-position factor and
apply the kick. It is dissipative and is not the printed sine-damping equation.

`section_crossings` requires an explicit coordinate, level and direction. It returns
linearly interpolated crossing times/states and successive intervals. It does not
refine event roots; recurrence distances include sample interpolation error. Rotation
with an unwrapped angle can have no repeated crossings of the same section level.

## Runnable examples

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter03/validation.py
.\.venv-examples\Scripts\python.exe examples/chapter03/validation.py --output outputs/chapter03-repeat
```

Use the unchanged frozen requirements-examples.txt environment. Examples generate
JSON/CSV diagnostics and six headless PNGs under ignored outputs/chapter03.
They show fixed-point portraits, equilibrium circles, homoclinic orbit, pendulum
trajectories/energy, limit-cycle observations and Lotka invariant drift.

The shared `validation.kantz_sampling_sweep` retains clean/noisy observations,
solver work, all Kantz curves and eligible counts, and caller-supplied fit slices.
No automatic neighborhood, embedding or fit-window choice is made. Empty
neighborhoods yield `no_valid_neighbors` with no fitted slopes. Other parameter
and integration failures propagate rather than being converted into empty fits.
The plot shows four labelled sampling/noise cases; JSON retains all 48 cases.

See [validation results](chapter03-validation.md) and [source corrections](book-corrections.md).
