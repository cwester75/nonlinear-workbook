# Chapter 7: Controlling and Synchronization of Chaos

All printed pp.243–274 / PDF pp.255–286 were visually reviewed before
implementation. Suspected source errors on pp.246,250–251,262,264,269,271 were
checked at full resolution. The NumPy-only core and frozen examples environment
are unchanged; existing APIs and tests are preserved.

| Section / printed pages | Executable coverage | Validation |
|---|---|---|
| 7.1, 243 | Feedback versus prescribed forcing; finite-run diagnostics | Explicit control/state/error histories |
| 7.2.1, 244–248 | Logistic fixed-point/period-two OGY | Exact target cycles, local cancellation, bounds, noise and nominal resets |
| 7.2.2, 249–253 | Additive-parameter Hénon, SISO pole placement, bounded feedback | Assigned poles, nonlinear remainder and printed-gain counterexample |
| 7.3, 254–257 | Periodically modulated Lorenz | Zero-forcing reduction, periodicity, Jacobian and exact-time sampling |
| 7.4, 258 | Generic feedforward force and synthetic anharmonic target | Target consistency, independent forced-linear analytical solution |
| 7.5.1, 259–262 | Hénon driver/response unstable-direction control | Simultaneous updates, invariant synchronization manifold, unavailable/singular projection handling |
| 7.5.2, 263–267 | Corrected Rössler master and two-state response | Full/conditional Jacobians and conditional QR spectrum |
| 7.6, 268–274 | Generic diffusive coupling; Lorenz, Van der Pol and logistic examples | Block decomposition, transverse flow shifts and exact map multipliers |

Control and synchronization theorems remain explanatory; finite capture and
negative estimated conditional rates do not establish global synchronization.
No physical phase estimator is introduced: full-state error approaching zero is
complete-state synchronization, stronger than phase locking.

## Maps and bounded control

Import `from nonlinear_workbook import chapter07 as c`.

`logistic_targets(r=...,period=1|2)` returns the nonzero fixed-point branch or
ordered, distinct period-two points. The fixed-point branch must lie in [0,1];
period two requires 3<r≤4. `logistic_ogy(initial,r=...,period=...,updates=...,
neighborhood=...,max_perturbation=...,noise_std=...,seed=...)` reuses Chapter 1's
single-step map. Neighborhoods are strict. The target is the nearest cycle point
to the current observation, with first-index tie breaking; its successor is the
next point in the target cycle. The feedback is
`δr=−r(1−2s)(observed−s)/(s(1−s))`. Require the allowed parameter range to remain
within [0,4]. A zero denominator is reported as singular, not divided through.

The separately named `henon_additive_step` implements `(a−x²+by,x)`. This is the
Chapter 7 parameterization, not Chapter 1's `(1+y−a*x²,b*x)`. They must not share
parameter perturbations by an implicit coordinate substitution. Fixed points and
the analytic Jacobian are supplied. `place_poles_2d(A,B,poles=...)` returns the
real gain K for A−BK using a controllability solve; poles must be real or a
complex conjugate pair. `henon_ogy` requires an actual target fixed point, gain,
strict max-norm neighborhood, update limit, perturbation bound and escape bound.

`henon_synchronize` controls only the response's additive parameter. Both maps
update simultaneously from old states. Current driver x_n supplies J_n; the
nominal next driver supplies an instantaneous left unstable covector l_(n+1).
The correction is `δa=−l_(n+1) J_n (response−driver)/(l_(n+1) B)`, B=(1,0).
This local projection is not a computation of covariant Lyapunov vectors.
The L1 neighborhood is strict; missing unique unstable directions or a small
normalized denominator yield zero applied control and explicit reasons.

All controllers reset the parameter to nominal at every inactive step. They
reject, rather than clip, proposals exceeding max_perturbation. Observational
Gaussian noise affects decisions only; the true map state is never clipped or
replaced by the measurement. Positive noise requires an explicit seed.

ControlResult contains true states including initial/final state, per-update
measurements, parameters, proposals, applied controls, target phases, measured
errors, activation flags, reasons and status. Histories describe completed finite
updates; a nonfinite trial is not appended. Escaped finite states are retained.
The final state is supplied separately from per-update measurements. Status
distinguishes completed, escaped and nonfinite arithmetic; an orbit that does
not acquire the target still completes its bounded simulation normally.

## Continuous systems and shared diagnostics

`modulated_lorenz` adds k*cos(omega*t+phase)*x to Chapter 5's baseline second
equation; its Jacobian is analytic. Baseline Lorenz parameters keep their existing
validation. Modulation is signed and may make the effective r negative; omega
must be positive. Absolute time and the printed omega=6.28318 are preserved.

`feedforward_force(rhs,target,target_derivative,t)` returns target′−f(t,target).
Then target is an exact solution of x′=f(t,x)+F(t), but attraction is a separate
question. The §7.4 synthetic example uses the existing anharmonic oscillator
a=b=1,c=.1, target q=.5 cos(t), p=−.5 sin(t), and only mechanical acceleration
forcing F=(0,−.5 sin(t)+.0125 cos³(t)). No extra error-feedback term is added.
The book expresses general forcing through adjustable parameters; this adapter
specifically assumes additive RHS actuation. It does not solve an arbitrary
inverse parameter-control problem.

`rossler` uses x′=−y−z, y′=x+ay, z′=b+z(x−c), positive a,b,c. The combined
state in `rossler_drive_response` is (x,y,z,response_x,response_z); the response
shares master y. `rossler_response_jacobian` is the two-dimensional derivative
with master coordinates held fixed, whereas `rossler_drive_response_jacobian`
is the full five-dimensional derivative.

`diffusive_rhs` and `diffusive_jacobian` accept an existing RHS/Jacobian plus
concatenated (u,v) and nonnegative coupling. Along u=v=s, the longitudinal and
transverse Jacobians are J(s) and J(s)−2cI. A constant orthogonal
symmetric/antisymmetric transformation proves the block decomposition, even
when J has complex eigenvalues. Flow transverse exponents shift by −2c along
the same reference trajectory. For maps the factors instead change to
f′(s_n)−2c; `logistic_transverse_curve` returns factors, running log averages,
exponent and sample count. A zero factor yields −inf, which is preserved.

Shared `conditional_flow_spectrum` in flow_validation (also re-exported by
validation) requires response_dimension, response_jacobian(t,full_state),
initial_basis and the existing physical-time QR controls. It integrates the full
state but propagates only the response tangent matrix. The existing full-state
QR function delegates to the same implementation. Rates remain in tangent-column
order, signs are normalized, burn-in aligns without accumulating growth, and
incomplete intervals are not measured. Trace averages refer to the supplied
conditional Jacobian. Failure with no elapsed measurement returns NaN estimates
with an explicit status. The example JSON encodes unavailable nonfinite values as
null; CSV curves and statuses retain their meaning.

Chapter 6's stroboscopic sampler now accepts arbitrary vector dimensions. Its
arguments, phase conventions, first retained time and two-state behavior are
unchanged. Periodic recurrence residuals compare samples separated by 1,2,4 drive
periods; no tolerance is silently converted into a period classification.

## Reproduction

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-examples.txt
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/chapter07/validation.py
.\.venv-examples\Scripts\python examples/chapter07/validation.py --output outputs/chapter07-repeat
```

The example records seed 20261015, parameters, controls, versions, work and failure
statuses. CSVs preserve measurements separately from true map states, conditional
curves, trajectories and phases. The diagnostic capture threshold is explicitly
1e−6; entries/losses count threshold crossings and final persistence refers only
to the last 100 updates. These are finite-run error summaries. See
chapter07-validation.md for acceptance targets and observed sensitivity.
