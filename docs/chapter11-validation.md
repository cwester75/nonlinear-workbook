# Chapter 11 validation

Executed 2026-10-12 in CPython 3.12.14, NumPy 2.3.5, and Matplotlib
3.11.2 on Windows 64-bit. Existing requirements-validation.txt and
requirements-examples.txt pins are unchanged. No new runtime dependency is added.

## Stage gates and acceptance

- Memory/similarity stage: 38 tests passed, including all original 31 Chapter 1/2 tests.
- Kohonen/TSP stage: 43 tests passed, including both previous stages.
- Final: all 49 tests pass (31 existing + 18 Chapter 11).

Numerical acceptance criteria were specified in tests before benchmark runs:
exact bipolar matrices/states and Boolean outputs require equality; first Kohonen
updates use 1e-14 absolute tolerance; Fourier invariance uses 1e-14; gradient checks
use centered differences at step 1e-6 with atol=1e-6 and rtol=1e-5. XOR requires
fresh mean half-squared loss <=.00025 and each prediction within .05 of its target.
Exact TSP lengths are checked to ten decimal places against the small reference.
These budgets establish algorithm checks, not generalization guarantees.

| Check | Measured result | Reference / interpretation |
|---|---|---|
| Five-neuron memories | Energy -8 for both patterns | Mathematical -1/2 quadratic form |
| Recall from all +1 | First stored pattern | Printed pp.355-356 |
| Sequential asynchronous enumeration | 32 of 32 converge; 16 stored, 16 inverted | Inversion reported separately from successful stored recall |
| Synchronous enumeration | 16 fixed outcomes; 16 cycles | Cycles must not be assumed to converge |
| Direct/long-term Hamming radii | 0 for both stored patterns under both fixed schedules | Exhaustive N=5 result; particular corruptions still recover |
| 40-neuron digits | Stored 1/2/4 fixed; selected three-bit corruptions recover | Only stated corruptions tested, not a general recall radius |
| Printed corrupted digit | Recalls digit 4 | Source p.361, synchronous hold-at-zero updates |
| Circular-shift feature error | 5.55e-17 | Fourier-magnitude invariance |
| Reflected-image feature error | 5.55e-17 | Distinct image can have the same descriptor |
| Kohonen first updates | Printed epoch reproduced | Winner 1 first, winner 0 second; eta=.6 |
| All Boolean truth tables | 14 converge; XOR/XNOR reach cap | Nonseparable cases are expected failures |
| Gradient-check maximum error | 5.96e-11 | Every input/output weight and bias checked |
| Online XOR | 542 epochs; loss 0.0002496301 | Book initialization; corrected current-weight gradients |
| Batch XOR | 2196 epochs; loss 0.0002499044 | Mean batch gradient, same rate and initial weights |
| Ring TSP heuristic | 65.0205052327 | Feasible tour; no optimum substitution |
| Exact eight-city optimum | 64.8559644015 | All 5040 city-0-fixed permutations |
| Heuristic gap | 0.1645408312 (0.253702%) | Seeded ring example's measured limitation |
| Book tour | 68.0193642527 | Feasible but not optimal under Euclidean costs |

Online XOR predictions in input order 00,01,10,11 are
[0.023732431, 0.979058305, 0.978580863, 0.023162014]; batch predictions are
[0.023712583, 0.979048722, 0.978540872, 0.023184009]. Both classify the four Boolean
inputs correctly and meet the stated continuous-output error bound.

## Controls and diagnostic boundaries

Memory uses unnormalized Hebbian outer products, symmetric zero-diagonal weights,
zero thresholds, and zero-field hold. The exhaustive example evaluates every N=5
configuration under synchronous and sequential asynchronous schedules. Randomized
fair sweeps are independently tested for seeded repeatability and nonincreasing
energy, including a nonzero threshold. Energy histories for asynchronous updates
are sampled after each individual neuron update; state histories are sweep-level.

The four-vector Kohonen count sweep uses 1-5 prototypes, 4000 random presentations,
seed 20261012, rate .5->.01, and winner-only neighborhoods. Final unused prototype
counts are [0, 0, 0, 1, 1]. Four
prototypes do not automatically recover four distinct observations: this seeded
run retains an unused prototype and two observations share a winner. Five
prototypes achieve near-zero quantization here with one unused prototype. Completed
means the specified update budget was executed, not that a global optimum was found.

TSP uses 24 neurons on the book-style initial circle centered at (10,10), radius 10.
Training has 15000 updates, seed 20261012, learning rate .8->.05, and Gaussian
ring-index width 12->.5. This departs from the listing's coordinate-distance
neighborhood. Decoded tour: [0, 5, 3, 1, 2, 4, 7, 6, 0]. Every city occurs exactly once plus
closure; winner ties use city index and tours are rotated to begin at city 0.

XOR uses two hidden units, the printed input/hidden/output weights transcribed
into separate bias vectors, sigmoid slopes 10/10, rate .05, epoch cap 10000,
seed 20261012, and mean loss tolerance .00025. Online order is shuffled per epoch;
batch mode uses all four rows simultaneously. The batch gradient is the mean,
so equal learning rates do not imply equal updates per epoch. Reported loss is
always recomputed at the final weights of the epoch.

The universal-approximation discussion is explanatory; this sigmoid-output model
restricts targets to [0,1]. No training-success guarantee or out-of-sample assessment
is claimed. Fourier magnitude only gives exact invariance for circular translations,
and its phase loss is demonstrated explicitly.

## Reproduction and artifacts

Follow [Chapter 11 guide](chapter11.md) for exact environment/install/test commands.
Run memory.py, kohonen.py, and feedforward.py from examples/chapter11. Each accepts
--output and records runtime versions, fixed seeds, controls, statuses, and histories.

Two final runs produced byte-identical JSON reports, CSV files, and PNG plots for
all 22 artifacts. SHA256 values are recorded in
[output hashes](chapter11-output-hashes.json). Outputs remain in ignored
outputs/chapter11/. All seven final figures were visually inspected for labels,
legends, correct data representation, and unclipped layout.

Additional tests cover empty/mismatched/nonfinite input, invalid schedules, ties,
zero denominators, zero-field hold, cycle/iteration exhaustion, unused prototypes,
activation saturation, gradient overflow, trained biases, same-current-weight
updates, and exact lower-dimensional distance fixtures. Source PDF and generated
environments are excluded from repository artifacts.
