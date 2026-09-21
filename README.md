# Replication Package for Weighting Geometry in Aggregate-Instrument Causal Analysis of Renewable Energy Policy and Emissions

**Author:** Saleh Dadjouy, University of Northern Colorado<br>
**Manuscript status:** Under review at Statistics and Public Policy (submitted July 2026)<br>
**Repository:** https://github.com/SalehDadjouy/energy-policy-emissions-code

This package implements the paper's empirical and simulation comparisons of
two-stage least squares (TSLS) and the robust aggregate-instrument estimator
(Robust). It includes analysis-ready data, reference results, a program for
the state-exposure figure, and verification procedures. Autoregressive
integrated moving average (ARIMA) inference uses stationary likelihood
estimation and analytical covariance with the observed denominator.

## Data and Reproduction Boundary

The inputs are `data/panel_lag2.csv`, `data/exposure_full.csv`, and
`data/exposure_restricted.csv`. Sources and construction are documented in
[`data/README.md`](data/README.md). This package starts from those analytical
files, not the underlying raw agency records.

The finite-window weight analysis also takes the fixed 20,000-draw
objective file, `data/finite_window_population_objective.npz`, as an input.
The workflow reproduces the paired estimator evaluation using that bank.
The package also includes the bank's reference weights and convergence
summaries. Construction of the bank is outside the reproduction workflow.

## Installation and Preflight

Use Python 3.11 and the exact versions in `requirements-reproduction.txt`.
This file pins the direct and transitive dependencies and installation
tools. Create the environment outside the repository:

```bash
python3.11 -m venv ../replication_environment
source ../replication_environment/bin/activate
python -m pip install -r requirements-reproduction.txt
python -B reproduce.py --preflight
python -B reproduce.py --structural-only
```

Preflight verifies package integrity and records the Python version, numerical
libraries, operating system, thread settings, and figure font. Structural
tests do not fit models or run Monte Carlo experiments. The default command
is preflight only, so execution requires an explicit flag.

## Numerical Execution

Save the preflight environment record outside the repository before running:

```bash
python -B reproduce.py --preflight > ../preflight.json
python -c 'import json; p=json.load(open("../preflight.json")); json.dump(p["environment"],open("../environment.json","w"),indent=2)'
python -B reproduce.py --execute --output ../primary_run --environment-lock ../environment.json
```

The destination must be new and outside the package. The workflow executes
the empirical analysis, principal simulation, denominator diagnostics,
state-panel comparison, confounding grid, finite-window weight evaluation,
target-anchor analysis, shock-orientation analysis, and exposure figures.
The exact commands, artifact counts, keys, and schemas are in `WORKFLOW.json`.
Reference outputs provide the comparison values for validation.

Completed stages have checksummed receipts. To resume after interruption, use
the same command with `--resume`. Completed, unchanged stages are reused.
An incomplete stage is preserved for inspection before execution resumes.

## Independent Reproduction

For an independent reproduction, check out the same release or commit in a
separate location and install a fresh environment with the pinned configuration.
Run the same workflow with a new output and cache directory. Use the original
environment record; an environment mismatch must be investigated before
execution. Then compare the two runs:

```bash
python -B reproduce.py --compare ../primary_run ../clean_clone_run
```

Execution integrity, same-environment reproduction, and archived numerical
compatibility are separate checks. Independent reproduction requires exact
bytes for the declared scientific CSV files, objective input, and rendered
figures. Path-dependent manifests and logs retain their own integrity records.
Fixed figure timestamps prevent a timestamp alone from changing PDF bytes.
Comparisons with archived results use the tolerances specified in
`WORKFLOW.json`. The report identifies differences separately from the
comparison between the two independently generated runs.

An execution exit code of zero establishes execution completion only.
The comparison command reports each acceptance check and exits unsuccessfully
if independent reproduction or either archived comparison has not passed.

Automatic checks on GitHub verify package integrity and run structural tests.
Numerical reproduction uses the explicit execution commands above.

## Inference and Provenance

ARIMA inference fits stationary ARIMA(2,0,0) by likelihood and uses analytical
covariance with the observed denominator. The comparison procedures use
heteroskedasticity-and-autocorrelation-consistent (HAC) standard errors and
Anderson-Rubin-style test inversion. In the shock-orientation analysis,
severity calibration and interval calculation are separate steps. If the
initial likelihood fit does not converge, the interval calculation retries
the same likelihood using the specified transformed-parameter gradient.

`qualified_inference/provenance.json` preserves source hashes and the status
recorded when the routines were assembled. Completed verification for the
published package is summarized in the
[v1.2.0 release](https://github.com/SalehDadjouy/energy-policy-emissions-code/releases/tag/v1.2.0).
`PACKAGE_LOCK.json` identifies the complete package tree.
`reference_outputs/SHA256SUMS` identifies the archived reference files.
[`RESEARCH_OUTPUTS.md`](RESEARCH_OUTPUTS.md) maps the manuscript exhibits.
The package does not rebuild the manuscript's typesetting.

## Citation and Contact

Until a journal citation or archival DOI is available:

Dadjouy, Saleh. 2026. "Weighting Geometry in Aggregate-Instrument Causal Analysis
of Renewable Energy Policy and Emissions." Manuscript under review,
Statistics and Public Policy.

Use [`CITATION.cff`](CITATION.cff) and identify the release or commit used.
Code is distributed under the [MIT License](LICENSE); data conditions are in
[`data/RIGHTS.md`](data/RIGHTS.md). Contact: saleh.dadjouy@unco.edu.
