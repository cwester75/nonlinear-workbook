# Book review and corrections

## Chapter 8 corrections and departures

All printed pp.275-298 / PDF pp.287-310 were visually reviewed before implementation.
Pages284,294,297 were rechecked at full resolution.

| Printed page | Type | Finding | Decision / evidence |
|---|---|---|---|
| 275-277 | Interpretation boundary | Hausdorff, box capacity and finite numerical scaling curves are different constructions. | Keep existence/definition theory explanatory and reuse explicitly sampled box/pair diagnostics; do not equate fitted slopes with Hausdorff dimension. |
| 281-283 | Construction convention | Printed recursion terminates by drawing two segments and uses integer screen coordinates. | Depth0 is one unit segment; complex similarities eliminate screen rounding and provide exactly2^depth segments. |
| 284 | Verified coordinate fixture | Three .5I maps use translations(1,1),(1,50),(50,50), defining a right-triangle version. | Preserve as printed=True; normalized equilateral fixture is an explicit coordinate choice, not a correction. |
| 287-289 | Geometry distinction | Text describes a closed snowflake while the Java example starts with a segment. | Provide separate open curve and closed outward snowflake; test the area formula only on the standard equilateral snowflake. |
| 290-295 | Finite-budget convention | Legacy images color nonescaping pixels after a fixed iteration count; the Julia listing plots a filled-set approximation. | Separate escaped/count/nonfinite arrays; label finite nonescape and filled Julia explicitly. |
| 296,298 | Truncation/sampling | Infinite regularity discussion accompanies a three-term sampled applet. | Return explicit finite frequencies, geometric tail bound and aliasing examples; no current theorem-status claim. |
| 297 | Characteristic normalization error | Cosine transform retains coefficient(lambda-1)/(2lambda) after combining both symmetric jump signs, giving value1/2 at k0, inconsistent with the normalized jump distribution and following functional equation. | Use normalized sum_j p_j*cos(k*b^j); test characteristic(0)=1. |
| 297 | Explicit finite-law departure | Book assumes b>lambda>1 for its asymptotic discussion. Selected bounded fixture uses b=lambda=2. | Permit both>1 for the finite law, cap levels, renormalize retained probabilities and report omitted mass. No stable-process limit is inferred. |

All exponential constructions enforce allocation bounds before generation. Floating
frequency/jump powers avoid integer overflow. Escape images and finite point clouds
remain numerical approximations; grid resolution and iteration budgets are recorded.

## Chapter 6 corrections and departures

All printed pp.205-242 / PDF pp.217-254 were visually reviewed before implementation.
Suspected derivative errors on pp.213 and230 were rechecked at full resolution.

| Printed page | Type | Finding | Decision / evidence |
|---|---|---|---|
| 211-212,228,232 | Sampling convention | Listings select points within finite timing windows, potentially recording multiple points around a drive period. | Integrate to exact endpoint times, one sample per requested period; retain absolute phases. |
| 213 | Verified equation, not an error | Lift uses u3=Omega*t; derivative of k*cos(u3) with respect to u3 is -k*sin(u3). Planning incorrectly suspected a missing Omega factor. | No frequency correction; analytic phase Jacobians are checked against finite differences. |
| 218,224,235 | Forcing-phase departure | Some listings reset time to zero after burn-in, changing forcing phase unless the burn is a whole number of periods. | Preserve absolute time; all observation metadata include forcing phase. |
| 219 | Listing/definition mismatch | Code truncates every lag to T-length, divides by diff+1 after diff terms, and omits lag-zero normalization. | Reuse available-pair demeaned autocovariance, return counts, normalize by variance. |
| 221 | Spectral convention | Continuous/infinite-time definitions do not prescribe a finite-sample window/FFT convention. | Explicit one-sided density, dt/window-energy normalization, DC/Nyquist handling and Parseval checks. |
| 230,232 | Taylor coefficient error | Listing uses -v*(1-k*cos(phase))*cos(q); differentiating -(1+k*cos(phase))*sin(q) requires -(1+k*cos(phase))*v*cos(q). The +Omega*k*sin(phase)*sin(q) term is correct. | Correct RHS/Jacobian and adaptive Fehlberg replace the legacy Taylor step. The planning suspicion about forcing-derivative sign was incorrect. |
| 239-242 | Multi-drive interpretation | Two supplied decimal frequencies do not justify assuming one drive period closes the two-state map. | Retain both phases; report sampling at drive1 period as phase-aware observations, not a closed projected return map. |

Conditional spectra keep phases fixed under perturbation; lifted spectra include
phase coordinates and neutral directions. Periodic shooting uses unwrapped state
equality, not an implicit angular quotient. Positive finite-time divergence,
autocorrelation decay and broadband spectra do not independently prove chaos.

## Chapter 5 corrections and departures

Printed pp.177-204 / PDF pp.189-216 were visually reviewed in full before implementation.

| Printed page | Type | Finding | Decision / evidence |
|---|---|---|---|
| 179 | Linear-stability boundary | Purely imaginary eigenvalues alone do not imply all linear trajectories are ellipses if nontrivial Jordan blocks occur. | Classify zero-real-part directions as center/nonhyperbolic; do not infer nonlinear stability. |
| 183,185,188 | Verified convention, not an error | Full-resolution reinspection confirms the prose and listings agree: sigma16,r40,b4. The planning review incorrectly suspected a parameter-order inconsistency. | Use named parameters; no source correction is warranted here. |
| 191-193 | Numerical departure | Unnormalized variational evolution grows or contracts exponentially; finite initial tangent norm also enters finite-time rates. | QR normalization with identity basis and explicit burn-in; return full spectrum and running rates. |
| 197-200 | Interpretation/numerical departure | Exterior-area growth is a two-dimensional exponent, asymptotically a pair sum for a generic initial plane, not the second individual exponent. | Independently construct additive second compound; periodic Euclidean normalization; compare with the largest pair of QR estimates. |
| 201-202 | Theorem versus diagnostic | Complex eigenvalue crossing does not verify all nonlinear Hopf hypotheses. | Label bracketed results candidates only; reject ambiguous pairs and additional center eigenvalues. |
| 203 | Parameter inconsistency | Earlier line states r0,b2, but derivative conditions -2b+2=0 and final conclusion require b1. | Require r0,b1; verify derivative cancellation and numerical invariant drift. |

Symbolic REDUCE/SymbolicC++ machinery is replaced by explicit equations, analytic
Jacobians and algebraic equilibrium roots, with NumPy eigenvalues. Theorems are
documented, not represented as numerical proofs. Solver and observation parameters,
noise, horizons and fitting slices remain explicit. No automatic chaos classification.

## Chapter 3 corrections and departures

All printed pp.135-152 / PDF pp.147-164 were visually reviewed before implementation.

| Printed page | Type | Finding | Implementation / evidence |
|---|---|---|---|
| 137 | Linearization boundary | Imaginary/zero eigenvalues need not determine nonlinear phase portraits. | Report nonhyperbolic linearization separately; test linear centers and degenerate matrices. |
| 138-140 | Reference departure | Printed orbit and finite Lie listing use their own parameterization/initialization. | Independently derive the time-centered positive separatrix q=3/(2+sqrt(13)*cosh(t/sqrt(2))); verify differential equation and zero energy. Do not claim the listing's initial state lies exactly on it. |
| 141 | Classification error | Eigenvalues +1,-1 are called an unstable node. | Classify odd-pi pendulum equilibria as saddles; even-pi equilibria are nonhyperbolic linear centers. |
| 142 | Integration convention | Pendulum listing drifts then kicks; Chapter10 symplectic Euler kicks then drifts. | Compare explicit kick-drift Euler and kick-drift-kick Verlet; document ordering, not bitwise legacy reproduction. |
| 147 | Equation/claim mismatch | Printed v'=-x-mu*sin(v) has Jacobian [[0,1],[-1,-mu]] at the origin, stable for mu>0, contrary to the unstable-origin claim. | Implement printed sign unchanged and document stable origin. No numerical proof of infinitely many cycles is asserted. |
| 148 | Equation/listing mismatch | Listing updates velocity with exp(-h*mu*(x_new^2-1)) then -h*x_new; this is a Van der Pol split update, not sine damping. It is called symplectic despite dissipation. | Separate van_der_pol_listing_step from sine_damping; retain exact update convention and label it dissipative. |
| 150 | Domain boundary | The positive-population discussion omits the boundary equilibrium(0,0). | Return both equilibria; logarithmic invariant explicitly requires positive populations. |

All planar classifications use continuous-time eigenvalue real parts, not map
multiplier magnitudes. Section crossings use explicitly documented sample-linear
interpolation, not solver event roots. The shared sampling harness retains periodic
controls to expose finite-window diagnostic sensitivity rather than certify chaos.

## Chapter 10 corrections and interpretation choices

All printed pp.313-336 visually reviewed 2026-10-13.

- p.320: the simple product splitting has local remainder O(h^2), and symmetric
  splitting O(h^3): global orders one and two. The phrase "up to third order"
  should not be read as global third-order accuracy.
- p.333: first-order product splitting is printed with O(t^3), inconsistent with
  p.320 and noncommuting flows. Use O(t^2) local error.
- pp.322-323: supplied coefficients are the fifth-order Fehlberg update. The
  embedded fourth-order weights and adaptive error controller are added explicitly;
  stage times are included for nonautonomous equations.
- p.327: the displayed Lie listing retains only two Taylor terms. It is a finite
  second-order approximation, not an exact solution as the surrounding prose says.
  taylor_step requires supplied derivatives; no symbolic REDUCE dependency.
- pp.324-328: ghost onset depends on initialization, mesh and arithmetic. Tests
  verify recurrence and eventual divergence, not the reported precise onset time.
  The parasitic mode is a property of the scheme, not proof of continuous chaos.
- pp.329-334: separable kick-drift and Verlet are implemented; general implicit
  Hamiltonians and higher-order BCH compositions remain explanatory material.
- pp.335-336: the scaled recurrence is an area-preserving discrete map. Its
  irregular motion does not establish chaos of the continuous quartic oscillator.

Shared harness samples at explicitly reached solver endpoints, adds seeded
observation noise afterward, and reports diagnostics in observation-time units.

## Chapter 12 corrections and implementation departures

Source: printed pp.419-494 / PDF pp.431-506, visually reviewed 2026-10-13.

| Printed page | Type | Finding | Decision / evidence |
|---|---|---|---|
| 435 | Listing error | SetBit's clearing branch ANDs with the bit mask instead of its complement. | Independently implemented binary vectors avoid packed storage; bit clearing requires AND with complement. |
| 438 | Numerical error | Prints scalar maximum at5.64891 for cos(x)-sin(2x). | Derivative equation gives sin(x)=(1-sqrt(33))/8; maximizing x=5.648318436046016. |
| 438, 442, 445 | Listing/domain mismatch | Discussion uses [0,2*pi]; first listing uses [-1,1]; packed listing restores [0,6.28318]. | Examples use explicit [0,2*pi], inclusive chromosome endpoints. |
| 447-448 | Interpretation boundary | Fitness-proportional probabilities assume suitable objective values; expected improvement is not monotonic improvement of each realization. | Reject signed/all-zero roulette weights; tournament selection supports signed scores. Record population means and best-ever values separately. |
| 449 | Listing error | Objective contains an unreachable second return statement. | Use Gaussian expression exp(-(x-1)^2-y^2/2). |
| 451-454 | Randomness departure | Repeated wall-clock reseeding inside operators can repeat random streams. | One seeded generator per run; independently spawned island streams. |
| 460-465 | Exact benchmark | Book solution has value149, weight19, below capacity20. | Exhaustive enumeration confirms optimal value; feasibility does not require filling capacity. |
| 476-477 | Precision convention | TSP input is a rounded distance table. | Reuse Chapter11 full-precision coordinates and exact benchmark; do not assert equality with rounded legacy output. |
| 479-480 | Algorithm/listing difference | Pseudocode proposes neighbors; code proposes uniform global candidates and prints final state. | Caller controls proposal and cooling; retain both best and final states. |
| 481-494 | Runtime departure | PVM distributed infrastructure is tied to the historical implementation. | Persistent serial ring-island model replaces messaging; no parallel speedup claim. |

Other explicit choices: binary_ga uses generational roulette or tournament selection
with optional one-elite retention, rather than pretending to reproduce every
different greedy listing. TSP retains greedy parent replacement. Crossover segments
are half-open; binary decoding is MSB first. Schema matching is executable, while
the conditional schema-growth argument is explanatory and does not guarantee
global optimality. See chapter12.md for coverage and chapter12-validation.md for
reference results and limitations.

## Chapter 2

Source: The Nonlinear Workbook, printed pp. 107â€“134 (PDF pp. 119â€“146).
Local reference: C:/Users/cw_se/NonSensitive/PDFs/the-nonlinear-workbook.pdf.
The source PDF is not distributed in this repository. These findings come from
visual inspection; implementations are independently written.

| Printed page | Finding | Implementation decision |
|---|---|---|
| 112 | Derives AV=C, then later writes AC=V. | Preserve AV=C when implementing local Jacobian fitting. Implemented with least squares and QR. |
| 113 | Successor distance is labeled d[0] instead of d1[0]. | Distinguish initial and evolved distances. |
| 117 | Neighborhood includes the last sample, but later reads its successor; empty neighborhoods divide by zero. | Restrict neighbors to valid future horizons; reject empty neighborhoods. |
| 120â€“121 | Kantz listing generates x but references u; embedding construction is unsafe for dimensions above one. | Consistent array name and backward delay vectors with valid bounds. |
| 119â€“121 | Listing logs a sum of neighbor distances rather than their mean, and divides by full T despite missing references. | Average within each neighborhood; average logs only over eligible references. |
| 125, 128 | Standard deviation formula and code differ: code subtracts mean squared from unsquared observations. | Compute sqrt(mean((x-mean)**2)). |
| 125 | Scale divisibility uses T-1 while partition definition uses T. | Explicitly drop incomplete trailing blocks for each scale. |
| 132 | Normalized complexity is described as bounded by one. | Treat normalization as asymptotic; do not clip finite samples. |

Interpretation boundaries: Pearson zero correlation does not imply independence.
H=0.5 does not establish independence. A Hurst scaling estimate does not by itself
establish fractional Brownian motion or a stable distribution. The chapter's
noise-color labels are not adopted as general classifications. Positive estimated
divergence in noisy observations is not by itself proof of deterministic chaos.

The Kantz implementation adds temporal neighbor exclusion and explicit delay;
these are implementation safeguards beyond the chapter's simple demonstration.
The known-map derivative estimate is a reference benchmark, not a substitute for
data-only estimation. Wolf replacement and local-Jacobian spectra are now implemented; see the
completion review below for conventions and coverage.

## Completion review (2026-10-10)

All printed pp.107-134 were visually re-inspected. The previous deferred-method
statement describes the initial implementation; Wolf and local-Jacobian methods
are now implemented. The coverage table is in chapter02.md.

Confirmed printed inconsistencies remain those listed above. Additional explicit
implementation departures and interpretation choices:

- pp.111-112: backward rather than forward delay vectors; equivalent index
  convention. Solve row-displacement least squares Y B=Z and use A=B.T, avoiding
  inversion of covariance matrices. QR replaces explicit Gram-Schmidt.
- p.118: the displayed replacement-count average assumes unit evolution intervals.
  For variable segment lengths, divide total log growth by total elapsed time.
  Evolution thresholds, maximum per-segment steps, directed angular limits, and
  temporal exclusion are explicit. If replacement fails, retain the pair as the
  text prescribes. Record boundary and zero-separation termination.
- pp.113-117: direct estimators exclude zero distances and neighbors without
  successors. Neighborhood mode averages logs per eligible reference. These
  safeguards are not claims that finite neighborhoods are unbiased.
- pp.130-134: LZ normalization uses the observed alphabet. A constant sequence's
  one-symbol alphabet has no logarithmic normalization; return zero by convention.
  The chapter assumes a fixed binary alphabet, which yields a small nonzero
  normalized constant-sequence value. Phrase counts remain compatible.
- Numerically full-rank local fits on curved, low-dimensional reconstructions can
  be poorly conditioned and generate spurious transverse exponents. No automatic
  conditioning cutoff or optimal embedding dimension is claimed.

Tests distinguish algebraic errors from finite-sample estimator behavior; parameter
sweeps are diagnostics, not proofs of chaotic dynamics.

## Chapter 1 corrections and interpretation choices

Visually reviewed all printed pp.1-106 / PDF pp.13-118 on 2026-10-11.
Findings below distinguish algebra/listing errors from deliberate numerical choices.

| Printed page | Type | Finding | Implementation / evidence |
|---|---|---|---|
| 14 | Listing error | Independent bungalow if statements end with an else attached only to the final if; earlier branch results can be overwritten with zero. Boundary .5 is also omitted. | Use one exclusive if/elif chain with explicit boundary ownership; a=1/3 reproduces tent map exactly. |
| 25 | Equation normalization error | Logistic invariant density is printed as 1/(2*pi*sqrt(x*(1-x))), although its integral is asserted to be one. | Use 1/(pi*sqrt(x*(1-x))); its CDF is 2/pi*arcsin(sqrt(x)). Tests verify bin masses and normalization. |
| 26 | Listing bounds error | Histogram index can equal n_bins at x=1. | Include final right endpoint in last bin; retain explicit counts. |
| 29 | Interpretation inconsistency | Calls bungalow Lyapunov function convex while also giving an interior maximum. | Do not adopt the convexity claim; retain parameter-specific numerical diagnostics. |
| 36 | Finite-sample convention | Autocorrelation listing uses a common truncated range and divides by diff+1 despite summing diff terms. | Use N-lag valid pairs and return pair counts; exact small-series fixture verifies sums. |
| 61 | Listing return error | Newton returns x0, the previous iterate, after computing x1. | Return the current iterate, stop on residual, bound iterations, and report singular derivatives. |
| 87-88 | Finite-sample convention | Counts ordered nonself pairs but divides by T^2. The limiting convention does not yield an exact finite-sample probability. | Count eligible unordered pairs and divide by their exact number, with temporal exclusion. Exact pair fixtures support the choice. |
| 90 | Grid convention | Capacity listing rescales axes separately to bounding-box grid dimensions and uses rounded indices. | Use supplied physical scales and fixed origin with floor cells; record this departure and origin sensitivity. |
| 98 | Listing error | Both escape conditions examine x11; x22 is never checked. It continues after escape. | Check both coordinates and stop at first escape/nonfinite state; retain update count and finite horizon. |
| 103 | Listing stopping error | Vector Newton stops when either coordinate difference is small because the continuation condition uses AND. | Stop only on the vector residual norm; return bounded iteration status. |
| 104-106 | Equation/listing inconsistency | y update uses newly computed x, while the printed variational equations differentiate a simultaneous update using old x. | Implement the sequential map and differentiate its composition, including chain-rule terms. Analytic Jacobian is checked against finite differences. |

Other departures: QR replaces long unnormalized tangent-vector multiplication,
avoiding exponential overflow. Scalar derivatives are provided directly rather
than requiring the book's symbolic C++ libraries. Fraction/int replace legacy
Verylong/Rational. NumPy's IID RNG supplies Monte Carlo uncertainty; the book's
polynomial map is retained as a separate demonstration. Forward Fourier
normalization is made consistent across the direct DFT and custom FFT.

Tent/Bernoulli float64 trajectories can collapse to zero despite chaotic real-number
maps. Exact rational orbits can be periodic; neither fact contradicts statements
about almost all initial conditions. The demonstrations do not prove Collatz,
universality, chaos, hyperchaos, or an asymptotic dimension.

The Feigenbaum demonstration uses multiplier crossings of the logistic family;
the book listing on pp.62-63 instead locates superstable critical-point cycles of
x^2+mu. Both illustrate the same limiting ratio, but finite parameter sequences
are distinct and are not compared as identical values.

## Chapter 11 corrections and implementation departures

Source: printed pp.337-418 / PDF pp.349-430, visually reviewed 2026-10-12.
The approved implementation follows corrected mathematics rather than adding
legacy execution modes. Relevant fixtures were rechecked at full page resolution.

| Printed page | Type | Finding | Decision / evidence |
|---|---|---|---|
| 338, 340 | Activation convention / listing inconsistency | The sign listing returns 0 for negative inputs, unlike the mathematical bipolar sign; zero conventions also differ between activations and Hopfield. | Explicit 0/1 threshold and -1/+1 bipolar functions; zero maps to positive for these functions. Hopfield separately holds its old state at zero field. |
| 348-349 | Convergence scope error | The single-cell energy proof is presented after synchronous updates; unconditional convergence to a fixed point does not follow for simultaneous updates. | Symmetric zero-diagonal asynchronous updates have nonincreasing energy. Synchronous recurrence is monitored for cycles; an exact two-neuron counterexample is tested. |
| 360 | Energy normalization difference | The program returns -sum(W_ij*s_i*s_j), twice the -1/2 quadratic form printed on p.348. | Use the mathematical normalization, yielding -8 for each five-neuron stored pattern. Dynamics are unchanged by this positive rescaling at zero thresholds. |
| 361 | Listing identifier error | The field-initialization loop declares po but tests/increments p. | NumPy allocation and explicit array validation replace unsafe allocation/initialization scaffolding. |
| 364-365 | Fourier convention | The printed square-image transform has a 1/N factor, whereas two forward-normalized 1D transforms yield 1/N^2. | Use forward FFT2 normalization 1/(rows*columns), consistent with Chapter 1. Translation invariance is unaffected by constant normalization. |
| 364-365 | Representation boundary | Fourier magnitudes discard phases; invariance does not make the descriptor unique. | Test circular shifts and distinct reflected images with matching magnitudes. Do not force continuous features into bipolar states. |
| 366 | Similarity interpretation | The normalized dot-product score divides by the number of ones in the first vector; it is asymmetric and undefined when that count is zero. | Label it as a book score, retain its definition, and reject the zero denominator. It is not presented as a metric. |
| 374, 378, 380 | Schedule inconsistency / explicit departure | Learning-rate factors differ between the worked exposition and listing (1.05 versus 1.1 appears in the text). | Reproduce the first epoch with eta=.6. Subsequent examples use fully recorded geometric per-update endpoint schedules. |
| 383-386 | Heuristic boundary | Reported city tour is feasible but not Euclidean-optimal for the printed coordinates. | Enumerate all 5040 city-0-fixed tours: optimum 64.8559644, book tour 68.0193643. Report the heuristic gap without substituting the optimum. |
| 384-386 | Neighborhood departure | Printed TSP kernel uses observation-coordinate separation rather than cyclic neuron-index distance. | Implement the approved ring-index Gaussian neighborhood with explicit width and learning schedules; this is a corrected SOM demonstration, not legacy code reproduction. |
| 398 | Update scheduling difference | Listing computes a set of responses before applying its weight corrections, unlike the online presentation rule. | Perceptron updates are online, train the bias, and verify the full dataset after each epoch. |
| 411-416 | Gradient/loss convention | The text discusses alternative layer-update scheduling; the listing's epoch error accumulates per-pattern errors obtained at changing weights. | Compute all gradients from the same current parameters and monitor a fresh full-dataset mean half-squared loss after each epoch. |

The MLP loss averages over samples, so the worked sum-loss tolerance .001 for four
XOR inputs corresponds to mean tolerance .00025. Online updates use one-sample
loss; batch updates use the mean gradient. Reported epoch counts therefore describe
these explicit conventions and are not asserted to match the legacy listing.

Spurious Hopfield states, inverted memories, synchronous cycles, unused Kohonen
prototypes, and training exhaustion are distinct outcomes. None is silently
converted into a successful recall or an optimal solution. The chapter's
universal-approximation theorem is explanatory capacity theory, not an implemented
proof, a guarantee of convergence, or a guarantee of generalization.
