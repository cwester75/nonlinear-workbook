# Chapter 12: Genetic Algorithms

Source: printed pp.419-494 / PDF pp.431-506. All 76 pages were visually reviewed
on 2026-10-13. Algorithms are independently implemented in
`src/nonlinear_workbook/chapter12.py`; no source PDF is redistributed.

| Section / printed pages | Coverage | Validation |
|---|---|---|
| 12.1-12.2, 419-424 | Population search, roulette/tournament selection, binary decoding, crossover, mutation | Exact 8-bit decoding, deterministic runs, evaluation accounting |
| 12.3, 425-427 | Schema matching, order, defining length; theorem and building-block hypothesis are explanatory | Four matches for *101*, order/length fixture |
| 12.4-12.5, 428-437 | NumPy binary vectors and standard Python integer bit operations replace packed C++ storage | Crossover/mutation fixtures and invalid binary inputs; assembly/storage examples are historical material |
| 12.6, 438-446 | Explicit bounded scalar fitness search | Continuous and exhaustive 10-bit references for cos(x)-sin(2x); cos(x) has endpoint maximum one |
| 12.7, 447-457 | Two-block chromosome for exp(-(x-1)^2-y^2/2) | Known maximum one at (1,0); quantization documented |
| 12.8.1, 458-459 | Explicit infeasible scores; penalty objectives can be passed as callbacks | No automatic penalty choice; Lagrange multipliers remain explanatory |
| 12.8.2, 460-465 | Binary knapsack and bounded exhaustive reference | All 4096 subsets; value 149, weight 19 |
| 12.8.3, 466-477 | PMX, permutation composition, greedy replacement, swap mutation, cyclic tour distance | Printed PMX children, randomized validity, square tour, eight-city exhaustive reference |
| 12.9, 478-480 | Metropolis acceptance, caller-supplied proposals and schedule, best/final distinction | Exact acceptance probabilities, controlled proposals, derivative-bracket scalar reference |
| 12.10, 481-494 | Persistent serial islands, independent random streams, ring migration | Population shapes, migration/accounting, elitist progress; PVM infrastructure is replaced, no speedup claimed |

## API conventions

Import `from nonlinear_workbook import chapter12 as c`.
`decode_binary` uses MSB-first bits and inclusive bounds. `binary_ga` maximizes a
finite objective evaluated on a copy of the chromosome. Choose selection,
crossover probability, per-bit mutation probability and elitism explicitly.
Raw roulette weights must be nonnegative with a positive total. Tournament
selection supports signed fitness, including a negative infeasible knapsack score.
There is no automatic fitness shift or inferred optimal parameter setting.

`SearchResult` returns best-ever candidate/value, population mean, best and
unique-chromosome fraction histories, final population, evaluation count, and
budget termination. Histories include the initial population. Unique fraction
counts representations: cyclic/reversed equivalent tours can count separately.
`tsp_ga` minimizes tour length; its histories follow that direction. It supports
nonnegative directed distance matrices as well as symmetric ones.

`pmx` uses a half-open segment [start,stop) and arbitrary unique integer labels.
`permutation_composition` requires labels 0 through n-1. Tours include their closing
edge. `exact_knapsack` refuses instances beyond its explicit enumeration limit.

`simulated_annealing` minimizes; proposal(state_copy,rng) returns a finite state
with the same shape. Schedules are positive and nonincreasing; repetitions are
explicit. Results retain every accepted-state value, acceptance flags, temperature,
best-so-far values, best state and final state. No automatic cooling schedule.

`island_ga` evolves persistent populations for a specified number of rounds.
At each boundary every island's best competes with the next island's worst.
Migration evaluation costs are included. Execution is serial; it demonstrates
the island algorithm, not a parallel runtime benchmark.

## Reproduction

Use the existing frozen `requirements-examples.txt` environment:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python examples/chapter12/search.py
```

The example writes JSON histories/parameters, a sensitivity CSV and a headless
PNG beneath ignored `outputs/chapter12`. See [validation](chapter12-validation.md)
and [corrections](book-corrections.md). Five fixed seeds show finite-run variation.
Bit resolution and mutation sweeps illustrate lost diversity and discretization;
they do not establish a generally optimal configuration. The finite success of
small knapsack/TSP examples is not a global-optimality guarantee for larger cases.
