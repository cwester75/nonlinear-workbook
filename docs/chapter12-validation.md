# Chapter 12 validation

Validated 2026-10-13: CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2;
existing complete frozen environment in requirements-examples.txt, unchanged.
Core remains NumPy only. Standard-library unittest: **61 tests pass**, including
all 49 existing Chapter 1/2/11 tests, 9 Chapter 12 tests and 3 numerical regressions.

Review fixes: TSP caches population scores, counting each actual tour evaluation
once (population_size + generations*(population_size + 2*floor(population_size/2))).
An independent roll-call counter checks zero/positive generations and odd/even
populations. Binary decoding preserves endpoints and avoids overflow for opposite
sign bounds such as [-1e308,1e308]. Chapter1 Monte Carlo rejects vector and complex
integrand outputs, requiring one real scalar per sample for both methods.

Exact fixtures use integer assertions, or floating arithmetic tolerance about
1e-7 (unittest's default seven places). Heuristic runs are observations with
reported gaps; they are not acceptance assertions selected after the run.
No tolerance is imposed on seed-dependent convergence. Tests assert exact
small-problem outcomes only for fixed deterministic regression fixtures.

```powershell
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-examples\Scripts\python.exe examples/chapter12/search.py
.\.venv-examples\Scripts\python.exe examples/chapter12/search.py --output outputs/chapter12-repeat
```

Both runs produced byte-identical report.json, sensitivity.csv and search.png.
The PNG was visually inspected. Outputs and environments are ignored.

## Observed examples

Seeds 42,43,44,45,46. Binary populations 32, generations 100, crossover .7,
mutation .03 per bit, tournament selection and one elite; Gaussian uses roulette.
All parameters, runtime versions and full histories are stored in report.json.

| Example | Reference | Observed |
|---|---|---|
| Scalar objective cos(x)-sin(2x), [0,2*pi], 10 bits | Continuous 1.760172593046087; exhaustive encoded maximum 1.760160899251661 | Four seeds reach encoded maximum; seed42 obtains 1.760137528264751 (gap 0.000023371 from encoded target) |
| Gaussian exp(-(x-1)^2-y^2/2), [-2,2]^2, 12 bits per coordinate | Continuous maximum 1; grid cannot encode (1,0) exactly | 0.9999993440288633 |
| Printed 12-item knapsack, capacity20 | Exhaustive 4096 subsets: value149, weight19 | All five reach149 |
| Eight-city Euclidean TSP, population20/generations100 | Chapter11 exact 5040 tours: 64.85596440153056 | Both PMX and composition reach target for all five seeds (roundoff <3e-14) |
| Annealing sin(x)*x^2*exp(-x/15), [0,100] | Derivative sign brackets, 60 bisections, endpoints: -121.79850504401014 at29.84547537503274 | Best energies range -121.762741 to -121.524296; positive gaps .035764 to .274209 |

The TSP uses the Chapter11 coordinates at full precision. Chapter12 p.476 provides
a rounded distance matrix, so displayed legacy lengths are approximate comparisons.
Annealing uses independent uniform proposals, initial50, 100 geometric temperatures
100 to .01, 20 proposals per temperature. The derivative reference uses a .01
bracketing grid; it is a numerical reference, not an interval-arithmetic certificate.

## Sensitivity and limits

At the same binary population/generation budget, bits6/10/14 and mutation0/.03/.15
are swept across all seeds. With no mutation, 14-bit runs range1.685749 to1.760134;
with .03 they reach1.760172545. Six-bit runs plateau at1.757112523 because of
quantization. More bits increase precision and search-space size simultaneously.
Mutation can recover lost alleles but high mutation may disrupt useful structure.
No optimal setting is inferred from this small sweep.

Tests independently check the p.424 decoder, schema matching, printed p.469 PMX
offspring, repeated permutation validity, exact knapsack, square tours, evaluation
accounting, monotonic elitist histories, Metropolis probabilities, controlled
contracting proposals and input failures. Island tests exercise persistent migration;
they do not validate PVM message passing or parallel speedup.
