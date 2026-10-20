# v0.2.0: validated coverage of all 13 chapters

Release date: 2026-10-19. This additive minor release includes Chapter 13 and its
review fixes, following v0.1.0. The [coverage matrix](coverage.md) accounts for all
13 chapters; guides distinguish executable examples from explanatory mathematics.
The numerical source baseline is 119ffec (Chapter 13 fixes). Release preparation
changes package version/documentation only, preserving existing public APIs and
frozen dependencies. v0.1.0 is retained unchanged.

## Validation contract

All validation is synthetic or uses explicitly transcribed book fixtures. No
outlierhunter archive, market data, held-out dataset or external observations
are read. No features are admitted to detection decisions, and no thresholds,
sizing, exits, portfolio or execution behavior are changed. Synthetic feature
variation does not establish incremental utility on future real observations.

Reference environment: CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2,
setuptools 81.0.0, pip 25.0.1; the complete release environment is frozen through
requirements-release.txt and requirements-examples.txt. NumPy remains the only
core numerical dependency. Local pip check passes. Hosted CI is not claimed to
have run; the existing workflow includes all tests and Chapter 13 smoke checks.
No remote push or package-index publication is part of this local release.

## Evidence

- The complete 172-test suite passes. It includes Chapter 13's fixed-width
  integer and finite-extreme Gaussian/logistic regressions, exact rational/integer
  checks, analytical rates, conservation identities and independent reference
  implementations from the preceding chapters.
- All 19 noninteractive example entrypoints across Chapters 1-13 and shared
  validation are run for this release. The optional puzzle interface is checked
  with its bounded TkAgg smoke command. Full runs retain explicit unavailable,
  escaped or exhausted numerical outcomes rather than hiding them.
- Numerical artifacts are compared against prior deterministic fixtures. Chapter
  13 is repeated, including smoke checks. Its nine outputs reproduce exactly.
- The observation diagnostic sweep retains 1,280 configurations: periodic,
  Gaussian IID, AR(1), and transient-contaminated periodic controls, five seeds,
  explicit sample counts/intervals, burn-in, noise, embedding and fit windows.
  Raw observations, curves, eligible counts and failures remain available.
- A wheel and immutable VCS installation are checked separately. The consumer's
  installed direct_url.json must match the full release commit; its Python/NumPy
  versions must match the pin. Package metadata must report 0.2.0.

Full artifacts and logs are ignored under outputs/release-v0.2.0; source PDFs,
environments and generated plots are not committed. Exact prior chapter targets,
tolerances and limitations remain in each chapter's validation document.

## Reproduction

```powershell
py -3.12 -m venv .venv-examples
.venv-examples/Scripts/python.exe -m pip install -r requirements-release.txt
.venv-examples/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
$env:PYTHONPATH='src'
.venv-examples/Scripts/python.exe -m unittest discover -s tests -v
$examples = @(
 'chapter01/stage_a.py', 'chapter01/stage_b.py',
 'chapter02.py', 'chapter02/research_validation.py',
 'chapter03/validation.py', 'chapter04/validation.py',
 'chapter05/validation.py', 'chapter06/validation.py',
 'chapter07/validation.py', 'chapter08/validation.py',
 'chapter09/validation.py', 'chapter10/validation.py',
 'chapter11/memory.py', 'chapter11/kohonen.py', 'chapter11/feedforward.py',
 'chapter12/search.py', 'chapter13/validation.py',
 'release_smoke.py', 'diagnostic_sensitivity.py'
)
foreach ($example in $examples) {
 .venv-examples/Scripts/python.exe "examples/$example"
 if ($LASTEXITCODE -ne 0) { throw "Example failed: $example" }
}
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --output outputs/chapter13-repeat
.venv-examples/Scripts/python.exe examples/chapter13/validation.py --smoke
.venv-examples/Scripts/python.exe examples/chapter09/puzzle.py --backend TkAgg --smoke
.venv-examples/Scripts/python.exe -m pip check
```

Use `git rev-parse 'v0.2.0^{commit}'` to obtain the immutable release commit.
A dependency should pin that full commit rather than a moving branch. For local
VCS installation use `git+file:///C:/Users/cw_se/quant/nonlinear-workbook@<commit>`.
The file URL requires the adjacent checkout; no remote availability is implied.

## Shadow consumer

Only experiments/nonlinear_workbook_shadow in outlierhunter is updated. Its
requirements.lock pins the full workbook release commit, and workbook-pin.json
records release version/tag and the exact Python/NumPy runtime. The adapter code
and runtime decision layers remain unchanged. The consumer CLI accepts only the
explicit --synthetic-smoke path; causal future-perturbation, constant-input and
insufficient-window checks run against the immutable installation. Mismatched
commits are rejected. The preceding release pin remains recoverable in Git.

## Interpretation boundaries

Finite dimension fits and divergence estimates remain sensitive to sample size,
noise, sampling, transient removal and fit windows. Reconstructed transverse
exponents can be spurious. Finite-horizon boundedness is not asymptotic attraction;
long quartic failures remain reported. Rule grades are not calibrated probabilities,
and fuzzy force-inference examples do not prove stabilizing control. No numerical
portrait or individual statistic is treated as proof of chaos or trading usefulness.


## Hosted CI follow-up (2026-10-20)

The first release push passed Linux validation, but Windows failed during runtime
setup: actions/setup-python had no Windows 3.12.14 build. No numerical tests ran
in that job. The CI repair preserves Python 3.12.14, using the publisher's
20260929 python-build-standalone Windows archive and SHA256 verification. Linux
continues to use setup-python. The matrix explicitly validates both current head
and immutable v0.2.0 on each OS; the release tag and numerical code are unchanged.
The consumer receives its own isolated synthetic-only workflow, with a portable
HTTPS VCS URL for the same full release commit. No archive-backed tests are added.
