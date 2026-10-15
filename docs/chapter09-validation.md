# Chapter 9 validation

CPython3.12.14, NumPy2.3.5, Matplotlib3.11.2; unchanged pinned dependencies.
Seed20261015. All136 tests pass: previous126 plus10 Chapter9 tests. Tests use exact
integer equality throughout; there are no fitted numerical tolerances or
automatically assigned behavioral classes.

## Exact acceptance checks

- All256 elementary rules and all eight local configurations are checked against
  their independent integer-bit definition. Rule56's printed truth table and
  Rule90 XOR are exact. Independent scalar loops validate finite evolution.
- Every initial configuration of periodic Rule90 rings of sizes1..8 is checked
  for spin correspondence and the product identity after one update. Every
  configuration of sizes2,4,8 collapses to zero by N/2 updates. The five-site book
  history is checked against an independent scalar implementation.
- Independent scalar two-dimensional loops check both neighborhoods and both
  boundaries for shapes1×1,1×2,2×2,3×5,5×3. Totalistic and outer-totalistic
  lookup results, code round trips, synchronous updates and input immutability
  are checked exactly.
- All18 Life center/count cases are tested, including overpopulation counts4..8.
  Block stability, blinker period2, and padded glider translation(+1,+1) after
  four updates are exact. Life's outer-totalistic code is224.
- Cycle detection checks preperiod, period, repeated endpoint, zero-step searches
  and exhausted bounds. Allocation rejection occurs before invoking callbacks.
- Puzzle tests check the printed swap, invalid clicks, immutable moves, no wrap,
  seeded200-move shuffles and inverse replay. An independent Manhattan-adjacency
  traversal exhausts all12 reachable2×2 states and verifies each transition.
- GUI controller tests cover legal/blank clicks, move counts, reset and the
  deterministic next-seed reshuffle. Invalid rules, states, tables, dimensions,
  counters, puzzle tiles and coordinates are rejected.

## Book recurrence correction

The printed initial spins(−1,−1,+1,−1,−1) correspond to binary00100. Exact
generations0..7 are:

```text
0  00100
1  01010
2  10001
3  11011
4  01010
5  10001
6  11011
7  01010
```

Thus the p.302 claim S(6)=S(1) is false for its stated map and initial condition;
S(4)=S(1)=S(7) is correct. Cycle detection returns preperiod1, period3, first
repeated endpoint4. Printed pp.302–303 were rechecked at full resolution after
the initial test exposed this mismatch. The implementation does not alter the
rule to force agreement with the printed recurrence.

## Demonstrations and finite-grid outcomes

Elementary examples use rules0,30,56,90,110,255, width257,128 updates, fixed-zero
exterior cells, and centered single-cell/seeded Bernoulli-half initial states.
Rule90 ring comparisons use sizes2,4,8,16,32,5,7,9,128 exported updates and a
256-update cycle-search budget. Initial random bits are recorded explicitly;
an all-zero draw is replaced by a centered cell so the collapse demonstration
starts from a nonzero state.

Periodic power-of-two rings collapse by generations1,2,4,8,16 respectively in
these fixtures. Fixed-zero boundaries on sizes4,8,16,32 instead give periods
3,14,30,62 for the selected states. Periodic odd sizes5,7,9 give periods3,7,7.
These are state- and boundary-specific finite outcomes, not infinite-lattice
classifications or periods guaranteed for every initial configuration.

Life examples run128 updates on64×64 grids, plus the printed8×8 block fixture
at rows/columns5–6. Blocks retain4 live cells and period1; blinkers retain3 live
cells and period2. A glider starts at rows/columns60–62 to expose the boundary:
periodic wrapping preserves its5 cells, while fixed-zero boundaries change it
into a four-cell still life at the edge.
Seeded random populations finish with425 cells under periodic boundaries and241
under fixed-zero boundaries; neither cycle search finds repetition within128
steps. Exhaustion means only no repeat was found within the supplied budget.

The totalistic fixtures use sum parity, including the center: lookup codes42
for von Neumann and682 for Moore. Both use a centered single cell,64×64 grids,
128 updates and both boundaries. Full histories make boundary-induced differences
visible even when the final periodic image resembles the initial state.

The seeded4×4 puzzle after200 legal moves is:

```text
 9  8 10  5
 6  4  1  2
 7  3  0 11
13 14 15 12
```

All200 inverse moves recover the goal exactly. Legal-walk shuffling guarantees
reachability from the goal, not uniform sampling of reachable permutations.
The optional interface passed a real TkAgg smoke check exercising its click,
reset and reshuffle handlers and canvas rendering; its visual layout was also
rendered and inspected. Normal validation uses Agg and opens no windows.

Two complete final headless runs produce54 byte-identical JSON/CSV/PNG files.
All six scientific/demo plots were visually inspected, together with the puzzle
interface rendering. Source PDF, environments and outputs remain outside Git.
