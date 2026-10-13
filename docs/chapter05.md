# Chapter 5: Nonlinear Dissipative Systems

Printed pp.177-204 / PDF pp.189-216 were visually reviewed in full before implementation.
Core remains NumPy only. Existing Chapters1/2/3/10/11/12 APIs are preserved.

| Section / pages | Coverage | Validation |
|---|---|---|
| 5.1, 177-182 | General continuous-time equilibrium classification; Lorenz equilibria | Residuals, stable/unstable/center dimensions, analytic stability threshold |
| 5.2, 183-187 | Lorenz trajectories using named parameters and Chapter10 adaptive Fehlberg | Book listing fixture, solver/horizon sensitivity |
| 5.3, 188-190 | State0/state1 phase portrait | Seeded/reproducible numerical data and plotted trajectories |
| 5.4, 191-193 | Full continuous variational QR spectrum with integrated trace | Linear analytical exponents, Lorenz divergence identity |
| 5.5, 194-196 | Generalized Lotka-Volterra and analytic Jacobian | Finite-horizon positivity, transient behavior, trace comparison |
| 5.6, 197-200 | Four-dimensional system; independent second-compound area propagation | Wedge derivative identity, expanding/contracting linear fixtures, finite-time QR pair-sum comparison |
| 5.7, 201-202 | Chemical model, algebraic equilibrium branches, bracketed eigenvalue crossings | Equilibrium residuals, Hopf-candidate search, ambiguous/real/no-crossing cases |
| 5.8, 203-204 | Lorenz time-dependent integral at r=0,b=1 | Algebraic cancellation and numerical relative drift |

Stability and invariant-manifold theorems, ergodicity and the nonlinear Hopf
theorem remain explanatory mathematics; executable diagnostics are not proofs.

## Model APIs

Import `from nonlinear_workbook import chapter05 as c`. Functions accept `(t,state)`
and keyword-only parameters; each has a matching `_jacobian` function:

- `lorenz`: sigma>0, r>=0, b>0. `lorenz_equilibria` returns the origin and two
  additional branches for r>1. `lorenz_hopf_threshold` requires sigma>b+1.
- `generalized_lotka_volterra`: a,b,c>0; three coordinates. Finite states are
  allowed for solver stages; positivity is monitored explicitly in examples.
- `hyperchaotic`: four coordinates and finite a,b,c,d. The name identifies the
  book's model, not an automatic classification of every parameter or trajectory.
- `belousov_zhabotinsky`: s,q,f,w>0. `chemical_equilibria` returns all three real
  branches and a strictly-positive-coordinate mask. The origin is not positive.

`classify_equilibrium(matrix,tolerance=1e-10)` reports eigenvalues, dimensions of
negative/positive/near-zero-real-part eigenspaces, hyperbolicity and linear stability.
An unstable direction establishes linear instability even when center directions
also exist. Otherwise center directions are labelled nonhyperbolic/inconclusive.
This does not change Chapter3's planar classification contract.

`hopf_candidate(jacobian,branch,bracket=...,tolerance=...,...)` takes callbacks
jacobian(parameter,state) and branch(parameter). It requires one isolated complex
pair, rejects multiple pairs/additional center directions, and bisects a real-part
sign bracket. It returns hopf_candidate, not_bracketed, no_complex_pair or
max_iterations. The caller selects and verifies the equilibrium branch; a crossing
does not check nonlinear nondegeneracy, transversality or cycle stability.

`lorenz_time_integral` requires exact r=0,b=1, positive sigma and finite state/time.
It rejects overflow/nonfinite values rather than returning a misleading invariant.

## Continuous-flow diagnostics

Import `flow_lyapunov_spectrum`, `exterior_area_growth`, `second_compound` and
`wedge` from `nonlinear_workbook.validation`. Implementations are isolated in
flow_validation.py and re-exported through the shared validation layer.

Both estimators integrate state and variational quantities together with adaptive
Fehlberg. Supply burn_in, duration, normalization interval, initial tangent data and
solver_parameters explicitly. The QR initial basis must be full-rank; it is
orthonormalized before use. Area initial_vectors has shape(2,n) and nonzero wedge.

QR evolves Q'=JQ, normalizes diagonal signs, and accumulates log(abs(diag(R))).
Burn-in evolves/aligns the basis without measuring growth. Rates use natural logs
per physical time. Columns retain their order; no running column sorting occurs.
The trace is integrated as an augmented scalar and averaged over the same measured
intervals. Full-rank checks use NumPy's numerical rank convention; overly long QR
intervals can lose contracting directions before exact singularity.

Area evolution uses the additive second-compound matrix, constructed independently
from antisymmetric basis matrices in lexicographic pair order(01,02,...). It uses
Euclidean wedge norm, periodic normalization and no QR. A generic two-plane tends
toward the largest pair sum; an invariant initial plane can select another pair.

Results include running rates/times, final state and normalized tangent data,
measurement elapsed time, actual augmented RHS calls, accepted/rejected internal
steps and status. Each augmented RHS call evaluates both f and J. Failure statuses
include solver statuses, singular_tangent or zero_area. Only completed normalization
intervals enter the measurement. On failure, final state/tangent data refer to the
last complete interval; work counters include the failed trial. Zero measured time
yields NaN estimates. Solver escape_radius applies to the entire augmented state.
max_steps is the per-normalization-interval trial bound.

## Reproduction

Use the unchanged complete requirements-examples.txt environment:

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter05/validation.py
.\.venv-examples\Scripts\python.exe examples/chapter05/validation.py --output outputs/chapter05-repeat
```

The example writes JSON/CSV histories and six headless PNGs under ignored
outputs/chapter05. Full long-horizon experiments take longer than unit tests.
See [validation](chapter05-validation.md) and [corrections](book-corrections.md).
No automatic chaos/hyperchaos label or application threshold is returned.
