# Chapter 6 validation

Runtime: CPython3.12.14, NumPy2.3.5, Matplotlib3.11.2. Seed20261014 for IID Gaussian
observation noise; dependencies remain pinned. All95 tests pass: the prior86 plus
9 Chapter6 tests. Reproduction commands are in chapter06.md. All examples run;
five plots were visually inspected.
The two complete example runs produced21 byte-identical JSON/CSV/PNG artifacts.
Generated outputs, environments and the source PDF remain excluded from Git.

## Approved analytical checks

- State and lifted Jacobian centered differences: absolute tolerance1e-6.
- Unforced pendulum/Van der Pol reductions and forcing periodicity checked directly.
- Exact stroboscopic timestamp/count/burn-in fixtures include nonzero initial time
  and both forcing phases; incomplete/nonfinite integration returns a prefix/status.
- Linear forced benchmark q''+2q'+2q=cos(3t): particular solution is
  Re(exp(3it)/(-7+6i)), including its analytic derivative for velocity. One-period
  endpoint and monodromy agreement require absolute error<1e-7.
- Shooting at phase time.4 from(1,1), period2*pi/3, one/two-period multiples:
  required residual<1e-9; example one-period residual3.0795e-12.
- Floquet fixtures cover sinks, positive/negative unstable multipliers, saddles,
  and unit modulus. Shooting tests cover singular solves and iteration exhaustion.
- Anharmonic exponent-sum error<1e-5; Van der Pol uses integrated Jacobian trace.
- Periodogram even/odd lengths, deterministic sinusoid, physical frequency,
  rectangular/Hann Parseval agreement to12 decimal places, DC/Nyquist power.
- Autocorrelation [1,2,3] gives [1,0,-1.5] with counts[3,2,1]; zero variance rejected.

## Driven fixtures and sections

All initial forcing phases are zero. Section solver: adaptive Fehlberg,
atol=rtol=1e-9, initial step.02, maximum step.1, escape radius1e6.

| System | Parameters | Initial state |
|---|---|---|
| Anharmonic | a1,b-10,c100,k1=0,k2=1.2,omega3.5 | (1,0) |
| Driven pendulum | a.2,b1,k1.1,omega.8 | (.8,.6) |
| Parametric pendulum | a.15,k.94,omega1.56 | (.1,.2) |
| Driven Van der Pol | a5,k5,omega2.466 | (.8,.6) |
| Two-drive pendulum | a.1,k1=k2=.3,omega1=.618034,omega2=1.618034 | (1,0) |

Single-drive section runs retain612 consecutive endpoints starting at100 periods:
samples0..511 give burn100, samples100..611 give burn200. They overlap and are
not independent experiments. Unwrapped coordinate mean differences between these
windows are about-.00263,15.1254,21.4735,-.00593 respectively; angle rotation and
finite-window drift must not be mistaken for an invariant statistic. The
two-drive section keeps512 points at the first drive's period, with second phase.
All section/trajectory runs return completed; no asymptotic attraction is asserted.

## Conditional and lifted spectra

Burn100 time units, identity initial basis, QR interval.1, maximum step.02.
Rates are natural logarithms/time in tangent-column order.

| Model | Duration100, tol1e-9 | Duration200, tol1e-9 | Duration100, tol1e-10 |
|---|---|---|---|
| Anharmonic | (.29009961,-1.29009961) | (.34086692,-1.34086692) | (.25057132,-1.25057132) |
| Driven Van der Pol | (.10804319,-7.03022052) | (.10126892,-6.88004267) | (.10716822,-7.03511866) |

Anharmonic sum-plus-a error is about1.4e-9 at default tolerance and1.5e-10 at
tighter tolerance. Van der Pol sum-minus-average-trace errors are about5.3e-9
and5.4e-10. The book's approximate.37 is a comparison, not a numerical acceptance
target. Finite-horizon rates vary under numerical perturbations of chaotic paths.

Anharmonic lifted spectrum over100 measurement units is
(.39446415,-1.39446415,0), with phase explicitly retained. Its state path differs
numerically from the conditional run because the augmented adaptive solves use
different error-controlled states. Their finite-time largest estimates need not
coincide. The extra phase direction is exactly neutral in this triangular setup.

## Observation diagnostics

8,192 unwrapped coordinate observations at dt.02/.05, burn100 time units, no noise,
for anharmonic and driven pendulum. Autocorrelation uses max_lag1000 and retains
available-pair counts. Periodograms demean and compare rectangular/Hann windows.
A separate off-bin sinusoid at frequency.713, N1024, dt.02 demonstrates leakage.
Driven pendulum rotation can introduce trends; demeaning alone is not detrending.
Different observation intervals also imply different total observation horizons.

32 Kantz configurations retain full clean/noisy observations and forcing phases:
N256/512, dt.02/.1, noise0/.02, dimensions2/3, delay2, Theiler20, radius.1*std(clean),
horizon8 and explicit fit slices[0:3]/[1:6]. All have usable neighborhoods. Fitted
rates range1.2389..21.1209 for anharmonic and.4310..7.8873 for driven pendulum.
They are window/noise/reconstruction-sensitive estimates, not substitutes for the
known-equation spectra or automatic chaos classifications.
