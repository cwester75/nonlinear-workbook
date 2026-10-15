# Chapter 4: Nonlinear Hamilton Systems

Printed pp.153–176 (PDF pp.165–188) were visually reviewed before implementation.
The quartic integral on p.173 and Toda Lax output on p.163 were checked at full
resolution. See book-corrections.md for confirmed errors versus conventions.

| Section / printed pages | Implementation | Validation |
|---|---|---|
| 4.1, 153–156 | Canonical RHS, energies, gradients and Hessians; separable trajectory driver | Energy gradients and Hamiltonian Jacobian identity |
| 4.1.1, 157–158 | Analytic Hénon–Heiles variational Jacobian | Centered differences; reuse of existing variational methods |
| 4.2.1, 159–160 | Periodic three-body Toda energy, momentum and cubic integral | Analytic directional derivative and invariant drift |
| 4.2.2, 161–164 | Separately identified open Toda; Flaschka coordinates, Lax pair and spectral invariants | Canonical/Lax consistency, commutator residual and eigenvalue drift |
| 4.2.3, 165–166 | Mathieu monodromy and Floquet diagnostics | Constant-coefficient analytical matrices, determinant and defective boundary fixtures |
| 4.3.1, 167–169 | Hénon–Heiles trajectory | Book initial conditions, energy error and solver refinement |
| 4.3.2, 170–172 | Hénon–Heiles energy shells and directional sections | Both book E=1/12 fixtures; refined crossing tests |
| 4.3.3, 173–176 | Quartic Hamiltonian, corrected extra integral, canonical rotation and sections | Polynomial cancellation, printed-form counterexample, separated energies |

Hamiltonian mechanics, involution/integrability, invariant tori and Floquet theory
remain explanatory mathematics. Numerical conservation and portraits are checks,
not proofs of integrability, chaos, or asymptotic boundedness.

## Models and conventions

Import `from nonlinear_workbook import chapter04 as c`. Every Hamiltonian state
uses `(q1,...,qn,p1,...,pn)`, unit masses and dimensionless physical time. Public
functions require finite states. `henon_heiles_*`, `quartic_*` and `toda_*` provide
potentials, gradients, Hessians, energies, RHS functions and state Jacobians.
Quartic functions require keyword `r` in [0,1]. Toda functions require an explicit
boolean `periodic`; exponential overflow raises `FloatingPointError`.

Periodic Toda has three exponential bonds, including exp(q3−q1). Its additional
integrals are P=sum(p) and
`p1*p2*p3-p1*exp(q2-q3)-p2*exp(q3-q1)-p3*exp(q1-q2)`.
The open model has only the first two bonds. `open_toda_lax(state)` returns a,b,L,A,
the analytical L derivative, commutator residual, symmetric-matrix eigenvalues,
and `(trace(L),trace(L²),det(L))`. Here a=exp(Δq/2)/2, b=p/2 and
`L'=A@L-L@A`. In particular trace(L²)=H_open/2. Open Toda positions need not stay
bounded; its scattering dynamics must not be confused with periodic Toda.

`quartic_integral(state,r=0)` rejects r≠0. `quartic_rotation(state)` applies
Q±=(q1±q2)/sqrt(2), P±=(p1±p2)/sqrt(2). At r=0,
H=H+ + H−, H±=P±²/2+Q±⁴/6 and I=3(H+−H−).

`henon_heiles_shell(energy=...,q2=...,p2=...,momentum_sign=...)` sets q1=0
and determines p1 with explicit sign ±1; negative kinetic radicands are rejected.
The saddle energy 1/6 identifies opening escape channels, not a universal chaos
threshold for every orbit.

## Shared numerical tools

`integrate_separable(state,grad_potential=...,t_span=...,step=...,method=...)`
wraps Chapter 10 kick-drift Euler or kick-drift-kick Verlet for T=|p|²/2.
Both endpoints are retained; the last step is shortened when necessary. The
IntegrationResult evaluation count counts potential-gradient calls (one/two per
successful Euler/Verlet step), not RHS calls. Explicit max_steps and escape_radius
bound computation. Status distinguishes completion, exhaustion, step underflow,
escape and nonfinite arithmetic. Invalid shapes remain input errors.

`refined_sections(rhs,times,states,coordinate=...,level=...,direction=...,
max_crossings=...,solver_parameters=...)` accepts any state dimension. Direction
is +1 or −1. Left endpoints are excluded, right endpoints included, and transverse
velocity must match direction. It reintegrates from each stored left state and
bisects time until coordinate residual ≤ tolerance (default 1e−10). Supply adaptive
Fehlberg controls. A bracket lost under reintegration is an explicit failure;
there is no interpolation fallback. It returns full states, times, original
brackets, residuals, failure reasons and refinement RHS work. Refinement does not
repair global error in the trajectory. Coarse trajectories can miss crossings;
compare step sizes. These tools are also re-exported from `validation`; Chapter 3's
existing two-state section API is unchanged.

`mathieu_floquet(delta=...,epsilon=...,solver_parameters=...)` integrates one
coefficient period π using Chapter 6's variational map and multiplier classifier.
The report includes trace, determinant error, classification and status. A trace
within tolerance of ±2 is boundary_unresolved: repeated unit multipliers may
have secular growth. `scalar_boundary` distinguishes a matrix near ±I from a
defective boundary candidate; it is not a nonlinear stability conclusion.

## Reproduction

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-examples.txt
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/chapter04/validation.py
.\.venv-examples\Scripts\python examples/chapter04/validation.py --output outputs/chapter04-repeat
```

The headless example is deterministic and uses no random draws (seed=null).
Parameters, versions, tolerances and work are in results.json. Section CSVs retain
every crossing; trajectory CSVs retain every twentieth stored state plus the
endpoint, while energy maxima and crossing searches use every state. Source PDF,
environments and outputs are excluded from Git. See chapter04-validation.md for
observed results and limitations.
