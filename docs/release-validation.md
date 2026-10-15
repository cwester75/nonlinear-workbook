# Validation-first release: v0.1.0

Chapter13 is deferred while validation and optional outlierhunter integration
receive priority. The user selected synthetic controls and an adapter first:
the governed24-instrument holdout archive was not opened or evaluated. The
[coverage matrix](coverage.md) records12 implemented chapters and the remaining
Chapter13 scope without representing numerical experiments as theorem proofs.

## Independent checks and affected examples

The audit starts from bf28697cdf24e455922c95ef59dcde63cf22914b. Fresh inspection
confirms that `_rhs` rejects nonfinite trial states before model validation;
integrate catches FloatingPointError/OverflowError while preserving ValueError
for invalid parameters/shapes. RHS work counts actual attempted evaluations.
Central difference checks both supplied states before invoking its recurrence,
including n=2. A fresh later-trial failure oracle verifies that previously
accepted times/states remain aligned with work counts. An independently formed
GF(2) shift matrix validates Rule90 for ring sizes1,2,3,5,8,16 over20 updates.

All144 tests pass:138 previous tests, four causal-adapter tests and two independent
release-audit tests. Existing Chapter9 exhaustive/local/scalar-reference tests and
puzzle replay tests remain unchanged. Chapter9's optional GUI had already passed
TkAgg smoke testing; automated CI and release checks remain headless.

The full affected Chapter3–7 examples were rerun into outputs/release-audit,
alongside Chapter9 and Chapter10 examples. Comparisons use SHA256 hashes against
the earlier saved artifacts. Smoke checks cover short integrations of the planar,
Hamiltonian, dissipative, driven and response equations, plus Mathieu, conditional
QR, numerical overflow and initial escape. None is a replacement for the full
chapter validations.

## Reproducible environment and CI

Reference platform is64-bit Windows, CPython3.12.14. requirements-release.txt
includes the complete existing examples lock (NumPy2.3.5, Matplotlib3.11.2,
setuptools81.0.0 and plotting transitive dependencies), plus pip25.0.1.
Core numerical requirements remain unchanged. `pip check` passes locally.

```powershell
py -3.12 -m venv .venv-examples
.\.venv-examples\Scripts\python -m pip install -r requirements-release.txt
.\.venv-examples\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.\.venv-examples\Scripts\python -m unittest discover -s tests -v
.\.venv-examples\Scripts\python examples/release_smoke.py
.\.venv-examples\Scripts\python examples/chapter09/validation.py
.\.venv-examples\Scripts\python examples/chapter10/validation.py
.\.venv-examples\Scripts\python examples/diagnostic_sensitivity.py
```

The GitHub Actions workflow runs the suite and smoke/example commands on
ubuntu24.04 and windows2022 with exact Python3.12.14. Checkout/setup-python are
pinned to full upstream commit hashes, resolved from their official repositories;
permissions are read-only and credentials are not persisted. Local equivalent
commands are validated. Hosted CI cannot be claimed passed until the commit is
pushed and that workflow executes. The local annotated tag v0.1.0 identifies the
reviewed source; no remote publication is implied.

## Synthetic diagnostic sensitivity

The fixed design contains1,280 configurations: periodic sine, IID Gaussian,
AR(1) with coefficient .8, and sine with a decaying transient; seeds11,23,37,41,53;
windows256/512; intervals .05/.2; transient removal0/100 samples; noise0/.02;
dimensions2/3; delay2; Theiler20; horizon8; and explicit fits[0:3]/[1:6].
The neighborhood radius is .2 times the clean signal's standard deviation,
fixed before adding observation noise. R/S scales are8,16,32,64. Development seeds
are11,23,37;41,53 are reserved synthetic seeds. No tuning, model selection or
market held-out evaluation is performed. Clean/noisy observations, parameter
records, full R/S/Kantz curves and eligible counts are preserved in ignored output.
Summary provenance records both HEAD and whether the working tree was dirty.

Observed ranges across this sensitivity design:

| Control | R/S H | Normalized LZ76 | Fitted divergence / time |
|---|---|---|---|
| Periodic | .059–.583 | .188–.469 | −.021–1.816 |
| IID Gaussian | .521–.666 | 1.063–1.156 | .022–24.595 |
| AR(1), short memory | .757–.957 | .688–1.031 | .253–19.742 |
| Transient periodic | .059–.665 | .088–.469 | −.819–4.756 |

These ranges mix explicitly recorded parameter settings; they are neither
confidence intervals nor acceptance regions. Finite R/S bias, short-memory
correlation, symbolization, sampling and early divergence jumps can dominate the
estimates. Positive slopes on IID/periodic controls do not establish deterministic
chaos. LZ values above1 are retained. All three diagnostics are accompanied by
mean, standard deviation, median absolute deviation, mean absolute increment and
lag-one centered cosine correlation. Descriptive within-control feature
correlations are reported; they do not demonstrate incremental predictive value.

## outlierhunter handoff

The optional consumer is isolated in an experiments folder and does not change
signals, sizing, exits, portfolio or execution. Its pin names a full workbook VCS
commit and requires CPython3.12.14/NumPy2.3.5. It verifies installed direct_url.json
against that pin and refuses editable or mismatched installs. See
../integrations/outlierhunter/README.md for the contract.

WindowConfig requires explicit units, sample interval, neighborhood, embedding,
exclusion and fit windows. The adapter uses an exclusive prefix endpoint and
only the trailing window. It does not validate external timestamps: callers must
ensure uniform sampling, or explicitly use observation/bar units rather than
claiming elapsed seconds. No price/return transformation, padding or imputation
is implicit. Missing/invalid windows and unavailable estimates retain explicit
statuses; causal future-perturbation tests pass. Window checksums, source IDs,
parameters, versions and verified commit provide provenance.

A later held-out incremental-value experiment must separately designate data,
freeze baselines/feature definitions and causal alignment, restrict learned
transforms to training data, and specify evaluation endpoints before inspection.
No held-out gain, feature admission or detector behavior is claimed by this release.
