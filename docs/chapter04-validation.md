# Chapter 4 validation

Reference environment: CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2. Dependencies
remain pinned in requirements-examples.txt; core remains NumPy only. There are no
random draws (seed=null). Exact reproduction commands are in chapter04.md.
All 114 tests pass: the previous 103 plus 11 Chapter 4 tests.

## Acceptance criteria established before the examples

- Centered-difference gradients/Jacobians: absolute tolerance 1e−6. Hamiltonian
  matrix identity and bounded-fixture Lax/canonical derivative checks: 1e−12.
- Toda and r=0 quartic invariant drift over 20 time units: below 1e−6, measured as
  max |I(t)−I(0)|/max(1,|I(0)|), adaptive atol=rtol=1e−10, maximum step .02.
- Mathieu analytical constant-coefficient monodromy: absolute error below 1e−7;
  determinant error below 1e−7. Include δ>0 oscillation, δ<0 growth and δ=0
  defective unit multipliers. The coefficient period is π, not 2π.
- Refined harmonic-oscillator crossing times: absolute error below 1e−7;
  coordinate residual below 1e−10. Include endpoint inclusion, duplicate
  prevention, grazing contacts, empty results, crossing limits and failed solves.
- Harmonic trajectory errors exhibit at least ratios 1.8/3.8 on halving the step
  for Euler/Verlet respectively. Long Verlet energy error is bounded below .001
  on the supplied unit-frequency, step .05, horizon 100 fixture.

Tests also cover inaccessible shells, nonfinite inputs, Toda overflow, malformed
gradients, invalid controls, escape, iteration exhaustion and incomplete Mathieu
integration. The quartic correction is checked using its explicit polynomial
gradient and the independent separated-coordinate energies. A finite state
(.4,.7,.3,−.2) gives derivative .0977 for the printed expression, disproving
conservation. The open Toda fixture's maximum Lax commutator residual is 8.33e−17.

## Observed numerical results

The book Hénon–Heiles trajectory has H(0)=.16466666666666668. Adaptive integration
to time 124 completes with 37,206 RHS evaluations and scaled energy drift
3.63e−10. This benchmark is below the saddle energy 1/6; no automatic dynamical
classification is attached to its portrait.

Periodic/open Toda use q=(−.3,.1,.2), p=(.2,−.1,−.1), horizon 100. Maximum scaled
invariant deviations are 1.42e−9 and 3.67e−12 respectively. The open Lax
eigenvalues drift by at most 1.89e−12. The open positions separate while spectral
invariants remain constant; positional confinement is not a shared property of
these two models.

Mathieu δ=1, ε=.5 gives multipliers approximately −.4658624203 and −2.1465564865,
with determinant error 1.18e−10. The 41×31 scan over δ∈[0,5], ε∈[0,2] reports
733 oscillatory cells, 535 exponentially unstable cells and three unresolved
boundary cells. All integrations complete; maximum determinant error is 7.03e−10.
Scan labels describe linear Floquet behavior at sampled parameter values, not
continuous boundaries inferred between grid points.

## Sections and finite-step sensitivity

All sections use q1=0, p1<0, horizon 2,000, crossing cap 2,048 and Verlet steps
.02/.01. The two E=1/12 Hénon–Heiles initial states are
(sqrt(1/8),0,0,sqrt(1/24)) and (0,0,sqrt(1/6),0). Extra shells at E=.125,.16
use q2=.1,p2=0 and positive initial p1. Book quartic initial state is
(q1,q2,p1,p2)=(1,1,1.5,1).

| System / fixture | Crossings, h=.02/.01 | Scaled energy error, h=.02/.01 |
|---|---|---|
| Hénon–Heiles E=1/12 I | 323 / 323 | 7.09e−6 / 1.77e−6 |
| Hénon–Heiles E=1/12 II | 312 / 312 | 9.21e−6 / 2.30e−6 |
| Hénon–Heiles E=.125 | 322 / 322 | 1.44e−5 / 3.59e−6 |
| Hénon–Heiles E=.16 | 322 / 322 | 1.79e−5 / 4.47e−6 |
| Quartic r=0 | 421 / 421 | 3.15e−4 / 7.87e−5 |
| Quartic r=.15 | 413 / 413 | 3.07e−4 / 7.68e−5 |
| Quartic r=1 | See failure discussion | 2.63e13 / .294 |

Quartic r=0's extra-integral errors are 3.33e−4 / 8.33e−5. These observed
factor-of-four reductions are consistent with Verlet's order. Short-horizon
adaptive energy errors range from 2.17e−11 to 1.26e−8 across the seven fixtures.
Short endpoint differences are recorded separately from invariant errors; phase
errors need not stay small merely because an invariant does.

The r=1 long run is a deliberate numerical failure demonstration. At h=.02 it
escapes the explicit state bound 1e6 after severe numerical energy growth. At
h=.01 it reaches time 2,000 but has about 29% scaled energy error and hits the
section-count limit. Both runs report failed refinement brackets. Their section
portraits are unreliable approximations to the intended constant-energy flow,
and must not be interpreted as evidence of its attractor or physical escape.
The potential has nonconfining channels; excursions introduce shorter local
timescales that a fixed step can fail to resolve. Smaller steps or a validated
adaptive trajectory would be needed for trustworthy long-horizon geometry.

All successful crossing residuals are ≤1e−10. This only certifies crossing
refinement relative to each stored left state. It neither repairs the trajectory
nor certifies an accurate energy shell. Every failure and solver counter remains
in results.json. The section-energy figure makes the long-run failure visible.

## Artifacts and reproducibility

The example writes the numerical report, section/trajectory CSVs, Toda invariant
and eigenvalue CSVs, the Mathieu scan, and headless plots under ignored outputs.
Trajectory CSVs for sections are decimated by 20; energy maxima and crossings
use all stored states. No random or elapsed wall-time fields can vary between
repeated runs. Two complete final runs produced 45 byte-identical artifacts,
including 11 PNGs, all visually inspected. Maximum retained crossing residual was
9.998e−11. The r=1 runs reported 31/93 refinement failures at h=.02/.01;
all other section fixtures reported none. Source PDF and generated artifacts
are not committed.
