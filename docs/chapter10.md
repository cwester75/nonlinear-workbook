# Chapter 10: Solving Differential Equations

All printed pp.313-336 (PDF pp.325-348) visually reviewed 2026-10-13.
Independently implemented in src/nonlinear_workbook/chapter10.py.

| Section / pages | Implementation and validation |
|---|---|
| Introduction, 313 | Explicit initial-value problem, state/time units supplied by caller |
| Euler, 314-316 | euler_step and integrate; logistic exact solution and first-order convergence |
| Lie series, 317-321 | taylor_step takes supplied time derivatives; second-order logistic fixture. Holomorphic convergence theory and REDUCE complex Lorenz machinery remain explanatory, not an automatic symbolic engine |
| Fehlberg, 322-323 | Six-stage fifth-order step, embedded fourth-order error, optional adaptive controller; exponential order, polynomial time forcing, rejection/accounting tests |
| Ghost solutions, 324-328 | central_difference with two explicit starting states; unstable parasitic mode demonstration |
| Symplectic methods, 329-334 | Kick-drift Euler and kick-drift-kick Verlet for separable Hamiltonians; determinant and long-horizon oscillator energy checks. General generating functions/BCH remain theory |
| Invisible chaos, 335-336 | quartic_discrete_step reproduces the scaled map; numerical map behavior is distinct from continuous quartic dynamics |

## Numerical contracts

rhs(t,y) returns the same real vector shape as y. Scalar initial states are promoted
to length-one vectors. Integration includes the initial state, clips the final step
to the requested end time, and supports forward time only. It returns accepted
times/states, accepted step sizes and normalized embedded errors, rejected count,
actual RHS evaluation count, and a termination status. Each Euler trial costs one
call; each Fehlberg trial costs six. max_steps bounds all trials, including rejects.

Methods: euler, rkf45 (fixed fifth-order Fehlberg), rkf45_adaptive. Adaptive control
uses max_i |error_i|/(atol_i+rtol*max(|old_i|,|new_i|)), safety .9, factor limits .1
and5, exponent1/5. Tolerances, min/max steps and escape radius are explicit.
Reported local errors do not certify global accuracy. This explicit solver is not
intended as a stiff solver. Fixed Euler error norms are zero placeholders, not
estimated error bounds. Exceptions signal invalid input; statuses signal incomplete
integration. Nonfinite trials are stopped rather than silently retried.

Symplectic methods require separable H=T(p)+V(q), supplied gradients and fixed
steps. They preserve symplectic structure, not exact energy. Do not apply them
unchanged to arbitrary coupled Hamiltonians or dissipative equations.

## Shared validation harness

validation.sampled_observations advances an equation to uniform observation times,
applies a scalar observation callback, then optional seeded IID Gaussian noise.
It records clean/noisy observations and solver work. It restarts the adaptive
solver at sample boundaries rather than interpolating; this convention is recorded
and may increase cost. Internal integration step and observation interval are
separate controls.

examples/chapter10/validation.py sweeps sample length256/512, interval.05/.2,
noise0/.02 and dimension2/3 for harmonic oscillator observations, then reports full
Chapter2 Kantz curves, eligible-reference counts and explicit fit slices[0:3]/[1:6]
in physical time. Other controls: delay2, Theiler20, radius.15, horizon8, seed20261013.
The known periodic system is a negative control: finite-window positive fitted
slopes do not establish deterministic chaos. No estimator optimum is inferred.

## Reproduction and observed results

Use the existing pinned requirements-examples.txt environment (Python3.12.14,
NumPy2.3.5, Matplotlib3.11.2):

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter10/validation.py
```

67 tests pass. Analytic acceptance tests require exponential Euler error ratio
1.8-2.2 on halving the step, Fehlberg ratio25-40, polynomial forcing error<1e-12,
adaptive exponential endpoint error<1e-8, symplectic determinant agreement to
12 decimal places and Verlet energy error<.0013 over5000 steps of.1.

Logistic max errors over t=0..10, u0=.1: Euler h=.2/.1/.05 gives
.0255674/.0127956/.00639390; fixed Fehlberg gives1.35e-8/4.78e-10/1.59e-11.
Adaptive atol=rtol=1e-10 gives1.10e-10 to1.51e-10. These observations are not
universal global error bounds. Harmonic energy error over3000 steps of.1:
Euler4.60e12, symplectic Euler.0263158, Verlet.0012500.
Central difference h=.05, starts.5/.5125, reaches max error1.50 by t49.95;
onset time is not treated as an exact benchmark. Outputs: ignored JSON/CSV/PNG.
The artifact plot was visually inspected.

The tested pre-Chapter10 baseline is commit
0797908f30ec685c25a68335f5cef498c89217c3. Next priorities are Chapters3,5,6,
with their source equations reviewed before extending the continuous benchmarks.
