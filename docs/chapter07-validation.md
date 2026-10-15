# Chapter 7 validation

Reference environment: CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2, using the
unchanged frozen requirements-examples.txt. Seed 20261015 controls map measurement
noise; continuous examples use no random draws. The complete suite has 126 passing
tests: the previous 114 plus 12 Chapter 7 tests. Reproduction commands are in
chapter07.md; generated outputs remain ignored.

## Acceptance criteria fixed before recording results

- Analytic state and combined-system Jacobians versus centered differences:
  absolute tolerance 1e−6. The Rössler response Jacobian equals the corresponding
  full-Jacobian response block.
- Logistic fixed-point/period-two residuals and assigned Hénon poles: 1e−12.
  Local logistic feedback errors scale quadratically; the Hénon controlled
  remainder is independently checked against −error_x².
- Controls respect their supplied perturbation bounds, strict neighborhoods and
  nominal resets. Simultaneous map updates and exact synchronization manifolds
  are checked directly. Missing unstable directions and singular denominators are
  explicit outcomes, not successful control activations.
- Symmetric/antisymmetric coupled-Jacobian decomposition: 1e−12, including a
  real matrix with complex eigenvalues. Fixed-point logistic transverse factors
  agree with 2−r−2c to 1e−12, and a zero factor preserves a −inf log rate.
- Conditional diagonal-flow rates and continuous transverse shifts: 1e−7. An
  independent full-system calculation retains the neutral drive direction;
  response-only QR has no such drive column. Conditional trace averages and
  final driver evolution are checked separately.
- Feedforward target consistency: 1e−12. Exact target and independent damped
  linear-oscillator solutions: maximum state error below 1e−7 over 20 time units.
- Three-dimensional stroboscopic timestamps, phase factors, retained counts and
  failed integrations are checked without changing existing two-state tests.

Invalid parameters, malformed states, uncontrollable systems, singular bases,
nonfinite derivatives, escaped trajectories, integration exhaustion, seeded-noise
reproducibility and unavailable estimates are also tested. No nonlinear capture
time, synchronization threshold or claimed periodic outcome is an acceptance
target.

## Experiment conventions

Maps run at most 20,000 updates. Logistic uses r=3.8, initial .28, periods 1/2,
neighborhood .001 and applied-perturbation bounds 0/.01. Hénon uses additive
parameter a=1.4,b=.3, the positive fixed point, poles .2/.3, neighborhood .005,
bound .01, escape radius 100, and initial states (−.760343,−1.40007), (.6,.7),
and fixed_point+(1e−4,−1e−4). The Hénon driver/response example uses (.5,−.8)
and (.001,.001), L1 neighborhood .01 and perturbation bound .01. Each profile is
repeated with measurement-noise standard deviation 0/1e−6. CSVs retain true
states, measured states, proposed and applied controls, parameters and reasons.

Capture summaries use true error <1e−6. A first entry does not imply permanent
capture: separate entry/loss counts and the last-100-update test are recorded.
Machine-precision agreement in noiseless runs is not a noise-robustness result.

Continuous solvers use adaptive Fehlberg, atol=rtol=1e−9 and 1e−10, initial step
.01, maximum step .02, per-integration trial limit 1,000,000 and numerical escape
bound 1e6. QR burn-in is 20 physical time units, measured duration 100/200, interval
.1. Spectrum columns remain in their propagated order. Histories retain physical
times and running estimates. Solver evaluation counts include augmented RHS
calls; those calls evaluate both the physical equations and tangent equations.

Modulated Lorenz uses (sigma,r,b)=(10,80,.4), initial (.8,.8,.8), phase zero and
the printed omega=6.28318. Every k=0,2,11,36.8 run retains 512 exact-endpoint
stroboscopic samples after 100 and 200 periods. Residuals are maximum Euclidean
differences at lags 1/2/4 over all retained samples, not an automatic period test.

The synthetic feedforward target q=.5 cos(t) uses a=b=1,c=.1 and initial q offsets
0,−1,−.1,.1,1 over 100 time units. Rössler uses (.2,.2,9), combined initial
(.8,.4,.8,1.1,2.3), trajectory horizon 220 and a two-dimensional conditional
spectrum. Coupled Lorenz uses (16,40,4), initial (.8,.8,.8,1,.4,.3), horizon 220,
and c=0,.45,.70,.75,1. Its transverse QR integration runs along a separate
synchronized reference trajectory initialized at (.8,.8,.8).

Van der Pol benchmarks use mu=1, initial (1,0,.5,.5), coupling 0/.2 and horizon
100. The logistic benchmark uses r=2.8, initial synchronized fixed point plus
(1e−4,−1e−4), coupling 0/.05/.1/.15 and 1,000 updates. Positive coupling can
destabilize the transverse direction of a discrete map; a continuous-time
negative exponent shift is not substituted into this calculation.

## Interpretation limits

The printed Hénon gains (−.1,1.3) yield poles approximately
−.83389627±.55192120i, both of modulus one. The implemented assigned poles .2,.3
give asymptotic linear stability locally, not guaranteed capture from every
initial condition under bounded feedback.

Negative conditional estimates describe local transverse perturbations along the
sampled trajectory, not the entire drive's spectrum or a global basin. Independent
nonlinear Lorenz integrations at different tolerances, or with different tangent
systems in adaptive error control, can diverge in phase. Therefore finite-time
spectral differences need not match a constant shift to roundoff, even though the
same-reference variational identity is exact. Complete-state error, conditional
growth, recurrence residuals and forcing phase remain distinct diagnostics.
In the k=2 stroboscopic projection, the automatically scaled z-axis magnifies
scatter of order 1e−11. That visible thickness is numerical scatter, not resolved
geometry of a thick orbit.

## Observed map results

For the explicit true-error threshold 1e−6, noiseless first entries occur at
updates 650/382 for logistic period 1/2, and 6,535/35 for the two nonlocal Hénon
initial states. Their final-100-update errors are at floating-point precision.
The near-target Hénon fixture verifies local acquisition separately from the
waiting time for a nonlocal trajectory to reach a controller neighborhood.

The noiseless Hénon driver/response enters the threshold at update 2,141 and its
final errors reach zero in float64. With measurement noise 1e−6, first entry is at
2,204 but the final-100 maximum true error is 2.75e−5. Thus first entry alone
would give a misleading account of sustained agreement; CSVs and JSON retain
entries, losses, proposals and true/observed errors separately. All controls use
their explicit bounds rather than optimizing gains or claiming noise robustness.

## Observed continuous-system sensitivity

For modulated Lorenz at tolerance 1e−9 and 100 burn periods, k=2 gives maximum
lag-1/2/4 residuals approximately 17.8834 / 4.37e−11 / 5.15e−11. This supports
near period-two full-state recurrence for this experiment, rather than imposing
the book's period-one label. At k=11, increasing burn-in from 100 to 200 periods
reduces the lag-1 maximum from .00650 to 2.60e−6, demonstrating a long transient.
At k=36.8, the 1e−9/200-period run has lag-2 residual 1.18e−9, whereas the
1e−10/200-period run has lag-1/2/4 residuals about 65.7/64.6/64.6. The selected
solver tolerances lead to markedly different finite trajectories; this does not
establish chaos, prove multistability, or diagnose the book's label as an error.
All 512 samples and forcing phases remain available for each configuration.

The Rössler response error at time 220 is 1.65e−7 at both solver tolerances.
Conditional spectra at tolerance 1e−9 are approximately
(−.03039085,−8.88848616) for measurement duration 100 and
(−.06719742,−8.86417051) for duration 200. The 1e−10/200 result is
(−.06719388,−8.86418700). Both measured directions contract in these fixtures,
but the slower conditional rate is visibly horizon dependent. Error spikes and
its long decay are retained in the full trajectory rather than hidden by an
initial transient cut.

At tolerance 1e−9 and measurement duration 200, coupled Lorenz's leading
transverse rates for c=0,.45,.70,.75,1 are approximately
1.38125,.48442,−.01670,−.14323,−.62240 per physical time. At c=.70 the
100-time-unit estimate is +.00750, changing sign when the horizon is extended;
the nonlinear trajectory's error at time 220 is 2.32e−6. At c=.75 its error is
5.92e−14, while c=0/.45 retain errors around 46.5/46.1 for these initial states.
These findings demonstrate horizon dependence near a local transverse boundary,
not a universally identified synchronization threshold. Independent adaptive
reference paths explain why finite rates are not forced to differ by exactly 2c.

Exact-target anharmonic integration has maximum error 4.42e−12 over time 100
at both tolerances; the four offset trajectories approach the same prescribed
solution in these runs. This observation does not extend the tested basin to all
initial conditions. Van der Pol's final state error is .7454 without coupling
and at floating-point precision with c=.2. The logistic fixed-point transverse
rates for c=0,.05,.1,.15 are −.22314,−.10536, approximately zero, and +.09531 per
update. Its c=.15 trajectory separates rather than synchronizing, illustrating
that increasing coupling need not stabilize a discrete map.

All recorded simulation/spectrum statuses are completed. Across Rössler and
Lorenz conditional/transverse calculations, the largest discrepancy between the
exponent sum and integrated conditional trace average is 4.01e−8. This checks
variational accounting, not the accuracy of each finite-time exponent as an
asymptotic limit.

## Reproducibility

Two complete final runs produced 94 byte-identical artifacts: 86 CSVs, seven PNGs
and results.json. All seven plots were visually inspected. Parameters, seed,
versions, solver work, full running curves, control reasons and recurrence
residuals are retained. No automatic period, chaos or application classification
is introduced; no source PDF, environment or generated output is committed.
