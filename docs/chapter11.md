# Chapter 11: Neural Networks

Working NumPy implementations of printed pp.337-418 (PDF pp.349-430), visually
reviewed on 2026-10-12. Numerical fixtures were rechecked against the printed pages
before implementation. Source PDF and rendered book pages remain outside Git.

The chapter treats memory as attractor dynamics, learning as prototype organization,
and classification as adjustment of decision boundaries. These introductory examples
are accompanied by exact small-system checks and explicit failure demonstrations.

## Coverage and examples

| Book section / printed pages | Implementation and working demonstration | Validation |
|---|---|---|
| 11.1 / 337-341 | threshold, bipolar_activation, logistic, tanh_activation; printed weighted-input fixture | Equality rules, extreme-input saturation, net=-1.33 |
| 11.2.1 / 342-344 | Hamming distance; memory patterns | Exact four-bit distance 3 |
| 11.2.2 / 345-347 | Synchronous Hopfield iteration and Hebbian weights | Five-neuron matrix and two-cycle detection |
| 11.2.3 / 348-349 | hopfield_energy; individual asynchronous energy histories | Nonincrease across all 32 five-neuron states |
| 11.2.4 / 350 | Enumerated five-neuron basins and Hamming radii | Exhaustive outcomes for each initial state, fixed schedule |
| 11.2.5 / 351 | classify_recall distinguishes stored/inverted/unmatched states | Inverted states are not labelled successful recall; cycles retained |
| 11.2.6 / 352-353 | hebbian_weights | Outer-product matrix, zero diagonal; weighted variants and spectral discussion remain explanatory |
| 11.2.7 / 354-356 | Five-neuron worked memory | Stored energies -8 and recall from all +1 |
| 11.2.8 / 357-362 | 40-neuron digits 1,2,4; printed corrupted input | Stored patterns are fixed points; specified three-bit corruptions recover |
| 11.2.9 / 363 | Sequential and randomized asynchronous sweeps | Fair full sweeps, deterministic seeds, energy checks |
| 11.2.10 / 364-365 | fourier_magnitude for digit images and circular shifts | Translation invariance; reflection ambiguity |
| 11.3 / 366-369 | Hamming, Euclidean, directed/symmetric Hausdorff; book binary scores | Exact fixtures, symmetry where applicable, undefined-score rejection |
| 11.4.1-2 / 370-373 | Winner-only, chain, and ring Kohonen updates | Index topology, tie rule, neighborhood tests |
| 11.4.3 / 374-381 | Four-vector worked epoch; prototype counts 1-5 | Printed first updates and final assignments/unused prototypes |
| 11.4.4 / 382-386 | Eight-city ring heuristic and exact enumeration | Valid closed tours and optimum 64.8559644 |
| 11.5.1-4 / 387-394 | Perceptron prediction/learning; geometric separator | Correct full-dataset classification and trained bias |
| 11.5.2-3 / 390-392 | All 16 two-variable Boolean truth tables | 14 linearly separable cases converge; XOR/XNOR exhaust their caps |
| 11.5.5-6 / 395-400 | Perceptron architecture and AND learning | Exact AND truth table and bounded online training |
| 11.5.7 / 401-405 | Explicit two-hidden-unit XOR construction | All four outputs exactly correct |
| 11.6.1-2 / 406-407 | Online/batch distinction; universal-approximation context | Capacity is explained, not claimed as an optimization guarantee |
| 11.6.3 / 408-418 | One-hidden-layer sigmoid MLP, fresh-loss histories, trained biases | Every weight/bias gradient checked; online and batch XOR converge |

The original C++ allocation, GUI scaffolding, and unsafe loops are replaced by
Python interfaces. Every chapter example family is accounted for; theoretical
results, general weighted-Hebbian variants, and global optimization guarantees
are not represented as newly proved numerical results.

## Public APIs and conventions

Import from `nonlinear_workbook.chapter11`. All data are finite real arrays.
Samples are rows; features are columns. Core algorithms use NumPy only. Fixtures
are in chapter11.fixtures and carry printed page references. Chapter 1 and Chapter 2
interfaces, generators, and dependency pins are preserved.

**Activations and similarity.** threshold returns 0/1 with 1 at zero;
bipolar_activation returns -1/+1 with +1 at zero. Hopfield has its own hold-at-zero
rule. logistic and tanh_activation accept a positive slope; logistic avoids unsafe
exponentials. hamming_distance counts unequal entries without normalization;
euclidean_distance measures vector distance. binary_similarity accepts only 0/1
vectors and kinds hamming, euclidean, or dot. The book dot score divides overlap
by the number of ones in the first vector, is asymmetric, and rejects a zero
first-vector denominator. These scores are not interchangeable metrics.

Hausdorff functions accept nonempty point matrices with matching dimensions;
directed_hausdorff(a,b) need not equal its reverse. fourier_magnitude uses FFT2
with forward normalization 1/(rows*columns). Only circular translations preserve
this descriptor exactly; phase loss also creates ambiguities between different
images. Floating Fourier features are compared directly, not silently quantized
into bipolar Hopfield states.

**Hopfield.** hebbian_weights sums bipolar outer products and zeros the diagonal.
hopfield_recall(weights, initial, mode=..., thresholds=..., max_sweeps=..., seed=...)
requires symmetric weights and bipolar states. Nonzero thresholds enter the field
as W@s-theta and energy as -0.5*s@W@s+theta@s. Numerical symmetry differences up
to 1e-12 are symmetrized; nonzero diagonal entries are rejected.

Modes are synchronous, asynchronous (index-order sweeps), and randomized (seeded
permutations, every neuron updated once per sweep). An iteration is one simultaneous
step or one complete asynchronous sweep. HopfieldResult returns state, boundary
states, energies, iterations, status, cycle_start/length, and parameters. Energies
include the initial value and every individual asynchronous update; synchronous
energies are recorded once per simultaneous step. Zero fields retain state.
Statuses are converged, cycle, and max_sweeps. classify_recall reports exact stored,
inverted, or unmatched comparisons; convergence status must be checked separately.

The five-neuron enumeration evaluates all 32 initial configurations under each
fixed schedule. Its reported direct and long-term Hamming radii require every
configuration within the radius to reach the specified pattern. Both radii are
zero for this small fixture, even though particular corruptions can recover.
These exact finite-state results do not establish recall radii for the 40-neuron
network, which is tested only on stated inputs.

**Kohonen.** kohonen_update returns new prototype rows and the pre-update winner.
Ties choose the first index. Winner-only learning changes just that prototype.
Chain/ring neighborhoods use index distances and exp(-d_index^2/(2*width^2));
a ring uses the shortest cyclic distance. Learning rates lie in (0,1].

kohonen_train accepts n_neurons, update budget, two-endpoint learning-rate/width
schedules, topology, initial rows, seed, order, checkpoint interval, and optional
stabilization tolerance. Schedules are geometric over individual updates. Random
order samples with replacement; sequential order cycles through the supplied rows.
When initial is omitted, prototypes are seeded uniformly inside the data bounding
box. A one-update schedule uses its first endpoint.

KohonenResult includes prototypes, assignments, distances, unused_prototypes,
checkpoint_updates, prototype_history, quantization_errors, updates, status, and
parameters. Quantization error is mean nearest Euclidean distance. Unused means
not nearest to any final observation, not necessarily never updated during training.
Fixed-budget execution returns completed; an optional displacement tolerance
returns converged or max_updates. Stabilization is not global optimality.

**TSP.** tour_length requires each city exactly once plus the closing index.
decode_ring_tour sorts cities by winning ring index with city-index ties, then
rotates to start at city 0; feasibility is retained even when winners tie.
exact_tsp fixes city 0 and enumerates all remaining permutations, limited to 2-10
cities. It returns tour, length, and evaluated_tours. The eight-city fixture uses
5040 permutations. Distances are Euclidean in the printed coordinate units.

The heuristic uses 24 ring neurons, 15000 updates, seed 20261012, a starting circle
centered at (10,10) with radius 10, learning rate .8->.05, and width 12->.5. Its
measured gap is reported. It does not silently replace its tour with the optimum.

**Perceptrons.** perceptron_predict uses w@x+b >= 0 for class 1. perceptron_train
accepts 0/1 targets, rate, epoch cap, optional initial weights/bias, seed, and shuffle.
The bias is updated along with weights. Each epoch is followed by a fresh full-data
verification. PerceptronResult returns weights, bias, predictions, error_counts,
epochs, updates, status, and parameters; statuses are converged/max_epochs.
xor_network is the explicit book threshold construction and takes two binary inputs.

**MLP.** MLPParameters stores hidden_weights (hidden,input), hidden_bias (hidden),
output_weights (output,hidden), and output_bias (output). mlp_initialize creates
seeded small random parameters. mlp_forward always returns (samples,outputs).
mlp_loss_and_gradients returns loss and matching MLPGradients, using current weights
throughout. Loss is mean over samples of half the sum of squared output errors.
Sigmoid slopes are included in both layers' derivatives; both biases are trained.

mlp_train exposes hidden width, initial parameters, learning rate, slopes, mode,
maximum epochs, tolerance, seed, and shuffling. Online mode updates after each
sample; batch mode uses one mean gradient per epoch. Every history entry is a
fresh full-dataset loss, beginning with the initial loss. MLPResult returns weights,
predictions, losses, epochs, updates, status, and parameters. The seed stream for
initialization is separate from the online permutation stream. Modes are online
and batch; statuses are converged/max_epochs. Invalid numerical states or gradient
overflow raise ValueError rather than return misleading finite diagnostics.

The XOR fixture uses the printed initial weights, two hidden units, slopes 10,
learning rate .05, seed 20261012, maximum 10000 epochs, and mean loss tolerance
.00025. Acceptance additionally requires every output within .05 of its target.
Targets for this sigmoid-output MLP are restricted to [0,1]; this is not the general
linear-output construction in Cybenko's theorem. Capacity does not guarantee that
training finds suitable weights or that a model generalizes beyond its examples.

## Run and reproduce

Reference environment: CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2, with all
optional dependencies frozen in requirements-examples.txt. From the repository
root in PowerShell:

```powershell
python -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-examples.txt
.\.venv-examples\Scripts\python -m pip install --no-build-isolation --no-deps -e .
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/chapter11/memory.py
.\.venv-examples\Scripts\python examples/chapter11/kohonen.py
.\.venv-examples\Scripts\python examples/chapter11/feedforward.py
```

Setting `$env:PYTHONPATH='src'` is an alternative to editable installation.
Each script accepts `--output PATH`. Defaults are outputs/chapter11/memory,
outputs/chapter11/kohonen, and outputs/chapter11/feedforward. Reports contain
versions, seeds, controls, status, and histories. Outputs include numerical CSVs
and seven PNG figures. Generated directories remain ignored.

See [validation](chapter11-validation.md), [output hashes](chapter11-output-hashes.json),
and [book corrections](book-corrections.md). This implementation adds no external
datasets, forecast models, production classifier, or deep-learning framework.
