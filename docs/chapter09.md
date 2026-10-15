# Chapter 9: Cellular Automata

Printed pp.299–312 / PDF pp.311–324 were visually reviewed before implementation.
Full-resolution checks include pp.302–303,305,307,311. NumPy remains the only core
dependency; Matplotlib remains optional and uses the existing frozen environment.

| Section / printed pages | Executable coverage | Validation |
|---|---|---|
| 9.1, 299–301 | Elementary binary rules, explicit neighborhoods and synchronous evolution | All 256 rules × eight local configurations; code round trips and scalar reference |
| 9.2, 302–303 | Rule90/spin correspondence, bounded cycles and finite rings | Exhaust all configurations through ring size8; spin product, collapse and corrected book recurrence |
| 9.3, 304–308 | von Neumann/Moore counts, totalistic/outer-totalistic rules and Life | Scalar rectangular-grid references, code224, all local Life cases, block/blinker/glider |
| 9.4, 309–312 | Sliding puzzle engine, seeded legal shuffling, replay and optional interface | Printed move, inverse replay, all reachable2×2 states, callback and live TkAgg smoke checks |

Historical classifications, infinite-lattice fractal dimensions, reversibility
theory and continuum-model connections remain explanatory. We do not infer chaos,
thermodynamic violations, computational universality or trading usefulness from
patterns or finite statistics. Rule90 is linear over GF(2), even though its spin
representation is multiplicative. Finite deterministic grids eventually repeat;
the cycle of an infinite lattice does not follow from a finite-grid experiment.

## Automaton APIs and conventions

Import `from nonlinear_workbook import chapter09 as c`. Discrete states must be
nonempty integer/bool arrays containing only0/1. Float arrays are rejected even
when their values happen to be integral. Output states are uint8. All steps are
synchronous and return new arrays without modifying input states.

- `decode_rule(code,entries=...)` returns ascending bit indices; `encode_rule`
  performs the inverse using Python integers. Lookups have at most1024 entries.
  Elementary rules have8 entries, index `4*left+2*center+right`; reverse the table
  to obtain the book's111,...,000 display order.
- `elementary_step(state,rule=...,boundary=...)` accepts rules0..255. Boundary
  must be explicitly `periodic` or `fixed_zero`. Fixed-zero means exterior ghost
  cells are zero; edge cells themselves still evolve.
- `binary_to_spin` uses S=2a−1; `spin_to_binary` validates spins−1/+1. These
  conversion helpers accept one- or two-dimensional arrays. Rule90 remains the
  same elementary engine, not a duplicated implementation.
- `neighbor_counts(state,neighborhood=...,boundary=...)` accepts rectangular
  grids and `von_neumann` (four offsets) or `moore` (eight). Counts use int64.
  Periodic offsets are counted with multiplicity: a1×1 live Moore grid has count8.
  We do not deduplicate neighbors or silently change the finite topology.
- `totalistic_step` takes a one-dimensional binary table indexed by the sum of
  the center and neighbors; its length is6 or10. `outer_totalistic_step` takes
  shape(2,5) or(2,9), indexed by center and neighbor count separately.
  `encode_outer_rule` and `decode_outer_rule` use bit position2*n+center.
  `life_table()` supplies B3/S23, whose code is224; `life_step` reuses this engine.

`evolve(step,initial,updates=...,max_elements=...)` returns generation zero plus
exactly updates generations. `detect_cycle(step,initial,max_steps=...,
max_elements=...)` returns CycleResult with history, preperiod, period,
examined_steps and status. Exact byte equality is used after canonical conversion
to uint8, and the repeated endpoint is retained. A search ending without repetition
reports `limit_exhausted` and None for preperiod/period. Zero updates/steps are valid.
Cycle conclusions assume a deterministic step function; state repetition alone
does not establish a cycle for a stochastic or time-dependent callback.
Callbacks receive a copy and must preserve shape and binary values.

Both history-producing routines check `(steps+1)*initial.size` using Python
integers before allocating the history; default max_elements is1,000,000. The
limit counts stored state elements, not dictionary/array-object overhead or all
temporary bytes. Cycle keys require additional bounded storage. Excessive requests
raise ValueError before invoking the step function.

## Sliding puzzle and optional interface

Puzzle boards are integer rectangular arrays with at least2 rows/columns and
tiles0..size−1 exactly once. Zero is the blank. `puzzle_goal(rows=4,columns=4)`
returns row-major tiles1..size−1 followed by zero, with a preallocation limit.
`legal_moves` returns adjacent tile coordinates. `puzzle_move(board,coordinate)`
returns a PuzzleMove containing a new board, moved flag, and blank positions.
In-bounds nonadjacent, diagonal and blank clicks leave the board unchanged;
malformed/out-of-bounds coordinates raise ValueError. No moves wrap.

`shuffle_puzzle(board,moves=...,seed=...)` returns the final board, forward tile
coordinates and inverse coordinates already in replay order. It performs exactly
the requested legal moves, avoiding immediate reversal by default. It preserves
reachability from its supplied starting board, without claiming uniform sampling,
optimal scramble distance or solvability of an arbitrary supplied board.
The default element limit bounds the board plus both coordinate histories.

The optional four-by-four interface is separate from core mechanics. Its
GUI-independent PuzzleController is tested directly. Reset restores the current
shuffle and clears the move counter; Reshuffle starts from the goal using the next
seed (`initial_seed+shuffle_index`). Only successful moves increment the counter.
Matplotlib/Tk is loaded only by the explicitly launched interface, never by core
imports or normal tests.

## Reproduction

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-examples.txt
$env:PYTHONPATH='src'
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/chapter09/validation.py
.\.venv-examples\Scripts\python examples/chapter09/validation.py --output outputs/chapter09-repeat
# Optional local interface; requires a graphical backend such as TkAgg:
.\.venv-examples\Scripts\python examples/chapter09/puzzle.py --seed 20261015
# Short real-backend check, then closes the window:
.\.venv-examples\Scripts\python examples/chapter09/puzzle.py --smoke
```

Examples record seed20261015, parameters, lookup tables/codes, versions, population
curves and cycle-search outcomes. Flattened grid CSV columns are in row-major
order; array shape is recorded in JSON. The puzzle replay CSV retains every board
including its initial scrambled state and final goal. See chapter09-validation.md
for exact acceptance checks, observed results and corrections.
