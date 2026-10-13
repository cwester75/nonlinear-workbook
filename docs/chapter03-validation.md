# Chapter 3 validation

Runtime: CPython3.12.14, NumPy2.3.5, Matplotlib3.11.2. Frozen example dependencies
are unchanged. Seed20261013. All **77 tests pass**: the previous67 plus10 Chapter3
tests. All examples ran; six PNGs were visually inspected. Two runs produced
26 byte-identical JSON/CSV/PNG artifacts. Reproduction commands are in chapter03.md.

## Acceptance checks

Tolerances were specified in the approved plan before implementation:
analytic Jacobian centered-difference tolerance1e-6, equilibrium residual1e-14,
homoclinic trajectory error<1e-6 over[-10,10], pendulum and Lotka invariant drift
<1e-6 over20 time units with adaptive atol=rtol=1e-10. Small-amplitude pendulum
period acceptance uses amplitude.001, max_step.02 and error<1e-5 from2*pi.
The homoclinic algebra/energy residual checks use tolerance1e-14.
Fixed Fehlberg invariant errors decrease by at least a factor8 when h=.2 is halved.

Tests cover all six Jacobians, finite/invalid inputs, all equilibria, canonical
linear classifications including defective and scalar repeated roots, section
validation, deterministic noisy observations, failed integration propagation,
explicit physical-time fits and empty neighborhoods.

## Observed results

Default example solver: adaptive Fehlberg, initial step.05, max step.05,
atol=rtol=1e-10; dimensionless time. Integration must return completed.

| Benchmark | Observed |
|---|---|
| Homoclinic shifted reference, t0=-10 to10 | Maximum coordinate error1.9154e-8 |
| Pendulum libration, (q,p)=(1,0), horizon40 | Max energy drift2.8320e-9; periods6.6999744-6.6999767 |
| Pendulum rotation, (0,2.5), horizon40 | Max energy drift5.9792e-8; no repeated crossings of unwrapped q=0 |
| Pendulum near separatrix, (0,1.999), horizon40 | Max energy drift3.8160e-9; one measured interval19.3650321 |
| Lotka-Volterra a=b=c=1, (2,1.5), horizon40 | Max invariant drift1.9947e-9; populations remain positive |
| Van der Pol mu1, initial(.1,.2)/(3,0), horizon120/burn60 | Mean section intervals6.6632798/6.6632865 |
| Van der Pol mu5, same initial states/horizons | Mean intervals11.6122300/11.6122287 |

Van der Pol maximum successive crossing-state distances are about1.24e-4 atmu1
and1.77e-5 atmu5. These finite-time measurements include interpolation error and
do not establish a theorem about unique limit cycles.

At fixed h=.05 over200 time units, pendulum maximum energy drifts are:
Fehlberg4.2597e-8, symplectic Euler.0110603, Verlet.000265287.
Symplectic structure does not imply exact conservation or superiority at every
finite horizon. Near-separatrix periods are particularly sensitive to energy error.

Printed sine damping mu5, initial(.05,.01), approaches the stable origin
(final about7.23e-13,-1.51e-13 at120). Initial(8,0) shows a recurring outer orbit
with seven measured post-burn intervals. This does not verify the book's assertion
of infinitely many cycles. The p.148 listing is recorded independently at
step.005, initial(.05,.01), mu5, horizon50.

## Sampling and estimator limitations

48 cases: three systems, sample counts256/512, intervals.05/.2, IID Gaussian
noise0/.02, dimensions2/3. Each observes coordinate0, delay2, Theiler20,
radius.15, horizon8, explicit slices[0:3] and[1:6]. Pendulum starts(1,0), Lotka
starts(2,1.5), both burn0; Van der Pol mu1 starts(.1,.2), burns60.
Solver restarts at each requested observation boundary; no interpolation is used.

Across these configurations, fitted Kantz slopes per physical time range:
pendulum-.00223 to4.35739, Lotka.01829 to2.63530, Van der Pol.00296 to1.55104.
These positive finite-window slopes on periodic controls demonstrate estimator
sensitivity to embedding, noise, sampling and window choice. They are not
Lyapunov certificates or automatic chaos classifications. Full curves, reference
counts, raw observations, parameters and fit results are retained for inspection.
