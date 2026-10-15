# Optional outlierhunter shadow adapter

This adapter is an experiment, not a signal, detector, forecast, sizing input or
execution consumer. It must remain outside outlierhunter's runtime decision
layers. No existing governed holdout archive is read. The current authorization
is synthetic controls and adapter mechanics only.

Copy adapter.py into an isolated experiments/nonlinear_workbook_shadow folder in
outlierhunter. A release installation supplies workbook-pin.json with full
workbook_commit, python and numpy values, and a requirements lock naming that
immutable VCS commit. The adapter rejects an editable install, a mismatched VCS
commit, or a different numerical environment. An installed package's
direct_url.json supplies the independently checked commit.

Run `python adapter.py --synthetic-smoke` in that pinned experiment environment.
No real dataset is accepted by the CLI. Python callers may use observe_prefix
only within their authorized research scope. The input endpoint is exclusive;
the adapter measures at most the specified trailing window and never reads
future values into a statistic. Units, sample interval, neighborhood radius,
R/S scales and divergence fit slice are explicit WindowConfig fields. Current
reference windows are256/512 observations; default R/S scales require at least128.
No padding or imputation occurs. Missing/invalid windows and unavailable
individual estimates have explicit statuses/reasons and None values.

Baseline mean, standard deviation, median absolute deviation, mean absolute
increment and lag-one correlation accompany Hurst R/S, median-symbol LZ76 and
Kantz divergence. LZ values above1 are retained. Divergence is a fit to finite
neighborhood-separation growth, in natural logs per stated time unit; it is not
automatically a Lyapunov exponent or a chaos classification. Window hashes,
source IDs, parameters, dependency versions and the verified commit support
provenance. Median symbolization uses only the same causal window; ties map to0.

Later held-out comparisons must freeze feature definitions, simple baselines,
window alignment, missingness handling, training-only transforms and an
evaluation endpoint before consuming the designated data. Synthetic separation
and descriptive correlation do not demonstrate incremental detection utility.
No held-out comparison or feature admission is claimed here.
