# Chapter 6: Nonlinear Driven Systems

Printed pp.205-242 / PDF pp.217-254 were visually reviewed in full before implementation.
Suspected source errors were rechecked at full resolution. Numerical core remains
NumPy only; no prior chapter API or dependency pin is changed.

| Section / pages | Coverage | Checks |
|---|---|---|
| 6.1, 205-207 | Fixed-phase return maps, monodromy/shooting, Floquet modulus classification and P/N,D/I labels | Analytical forced-linear periodic orbit and multiplier fixtures; topological propositions remain theory |
| 6.2.1-6.2.2, 208-212 | Driven cubic oscillator and stroboscopic sampling | State/lifted Jacobians, exact period endpoints and forcing phases |
| 6.2.3, 213-215 | Conditional and phase-lifted QR spectra | Conditional sum=-a, explicit neutral phase direction |
| 6.2.4, 216-220 | Normalized finite-pair autocorrelation | Exact arrays, physical lags, counts and constant rejection |
| 6.2.5, 221 | One-sided periodogram, rectangular/Hann windows | Sinusoids, Parseval, DC and Nyquist |
| 6.3, 222-228 | Externally driven pendulum | Unwrapped dynamics, section data, spectra of observations |
| 6.4, 229-232 | Parametric pendulum | Correct RHS/Jacobian, output-only angle wrapping |
| 6.5, 233-238 | Driven Van der Pol | Reused Chapter3 unforced model, conditional QR trace identity |
| 6.6, 239-242 | Two-frequency pendulum | Autonomous two-phase lift and phase-aware sampling; no projected closed-map claim |

## Models and phases

Import `from nonlinear_workbook import chapter06 as c`. The functions `anharmonic`,
`driven_pendulum`, `parametric_pendulum`, `driven_van_der_pol` and
`two_drive_pendulum` accept `(t,state)` and named parameters; each has a matching
`_jacobian`. The first four use omega/phi; the last uses omega1/omega2/phi1/phi2.
For anharmonic, parameters are a,b,c,k1,k2; other pendulum models follow their
printed equations with a,b,k or a,k. Van der Pol uses a,k. Parameters and states
must be finite; damping a>=0 and frequencies>0; stiffness/forcing may be signed.

Time is absolute and phases are radians. Integration never wraps angle. The
explicit `wrap_angle` function maps output to [-pi,pi). In two-drive forcing,
approximate decimal frequency ratios do not establish a common physical period.

`phase_lift(system, **parameters)` returns RHS/Jacobian callables, frequencies,
offsets, and an initial_state(state,time=...) helper. The lifted coordinates are
actual phases theta=omega*t+phi, so the forcing derivative with respect to theta
has no extra omega. Conditional two-state tangent dynamics keep the driving phase
fixed; a lifted spectrum includes the extra phase directions. These are different
finite-time experiments and need not follow identical numerical chaotic paths.

## Sampling and periodic orbits

`stroboscopic_sample` takes initial_time, period, burn_in_periods, samples,
drive_frequencies, drive_phases and solver_parameters explicitly. Initial state is
the state at initial_time. The first retained state is at
initial_time+burn_in_periods*period. With zero burn-in it is the initial state.
Exactly samples observations are returned on completion; failures return a prefix
with status and actual solver work. No interpolation or time-window selection.
Phases are (omega*absolute_time+offset) modulo2*pi; values near2*pi equal phase0
on the circle. Supplied frequencies/phases are metadata and must match the RHS.

`variational_map` integrates two-state dynamics and its 2x2 fundamental matrix
from a fixed initial_time for a supplied period. Its monodromy is with respect to
initial state at that fixed phase. `shoot_periodic` solves P^m(y)-y=0 using bounded
Newton iterations and linear solves. It takes an explicit positive period_multiple,
residual tolerance, iteration limit and state escape radius. It returns residual
history, evaluated endpoint/monodromy and actual augmented evaluations. Statuses:
converged, singular_solve, max_iterations, escaped, nonfinite or integration_*.
On an unevaluated failed Newton proposal, endpoint/monodromy remain the last
evaluated map. Residuals use unwrapped state equality; rotating pendulum orbits
modulo2*pi are not silently treated as fixed points. Callers must supply a true
forcing period for periodic-orbit interpretation, especially with two drives.

`classify_multipliers` uses modulus and returns eigenspace counts, hyperbolicity,
linear stability and the chapter label. P/N records even/odd unstable dimension;
D/I records orientation on the unstable subspace. Unit-modulus boundaries receive
no P/N,D/I label. Singular maps also receive no such label. This is independent
of continuous-time classification by eigenvalue real parts.

## Signal analysis

`normalized_autocorrelation` reuses Chapter1 demeaned autocovariance divided by
each lag's available pair count, then normalizes by lag-zero variance. It returns
physical lags and pair counts; zero variance is rejected. This finite-pair
estimator can exceed magnitude1 at large lags and is not clipped.

`periodogram` requires explicit sample_interval, demean and rectangular/hann
window. Frequencies are cycles/time (not angular frequency), density signal^2*time.
The symmetric Hann window is np.hanning; two samples give zero window energy
and are rejected. Density is dt/sum(w^2)*abs(rfft(w*x))^2, with only interior
positive-frequency bins doubled. For odd sample counts the last bin is doubled;
for even counts Nyquist is not. DC is never doubled. Integrated density equals
the window-weighted mean square of the processed signal. Demeaning does not
remove a trend, and unwrapped rotation observations can be nonstationary.

## Reproduction

Use the unchanged complete requirements-examples.txt environment:

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter06/validation.py
.\.venv-examples\Scripts\python.exe examples/chapter06/validation.py --output outputs/chapter06-repeat
```

Examples write seeded JSON/CSV and five headless PNGs under ignored outputs/chapter06.
One continuous section trajectory supplies the overlapping burn100/burn200 windows
for single-drive fixtures, retaining512 samples in each. Two-drive512 samples
retain the second phase and are not labelled a closed projected return map.
Phase portraits show100 time units after100 drive periods. All parameters,
observation phases and solver costs are retained in report.json.
See [validation](chapter06-validation.md) and [source corrections](book-corrections.md).
