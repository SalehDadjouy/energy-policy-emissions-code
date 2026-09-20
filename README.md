# Replication Package for Weighting Geometry in Aggregate-Instrument Causal Analysis of Renewable Energy Policy and Emissions

**Author:** Saleh Dadjouy, University of Northern Colorado<br>
**Manuscript status:** Under review at Statistics and Public Policy (submitted July 2026)<br>
**Repository:** https://github.com/SalehDadjouy/energy-policy-emissions-code

This package implements the empirical TSLS and Robust aggregate-instrument
analyses and the simulation comparisons described in the paper. It includes
analysis-ready data, reference results, the state-exposure figure program,
and explicit verification procedures. The version 1.2.0 implementation uses the qualified
likelihood-based ARIMA reporting implementation; the study's data-generating
designs and point-estimation procedures retain their existing definitions.

## Data and Reproduction Boundary

The inputs are `data/panel_lag2.csv`, `data/exposure_full.csv`, and
`data/exposure_restricted.csv`. Sources and construction are documented in
[`data/README.md`](data/README.md). This package starts from those analytical
files, not the underlying raw agency records.

The finite-window weight analysis also takes the verified 20,000-draw
objective file, `data/finite_window_population_objective.npz`, as an input.
It regenerates the paired evaluation, not the objective bank. The bank's
weight and convergence summaries remain archived references. Their presence
does not constitute a new reproduction of the bank-construction experiment.

## Installation and Preflight

Use Python 3.11 and the exact versions in `requirements-reproduction.txt`.
This file includes the direct and transitive dependencies and installation
tools used for qualification. For qualification,
create the environment outside the repository:

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
No reference output is substituted for a generated result.

Completed stages have checksummed receipts. To resume after interruption, use
the same command with `--resume`. Completed, unchanged stages are reused.
An incomplete stage is preserved and requires investigation; the workflow
does not silently delete it or start that stage again.

## Independent Reproduction

Before publication, clone the exact release-candidate commit into a new
location and install a fresh environment with the same pinned configuration.
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
Archived comparisons retain their specified tolerances. Differences are
reported and block release acceptance; they do not cancel an otherwise valid
independent reproduction.

An execution exit code of zero establishes execution completion only.
The comparison command reports each acceptance check and exits unsuccessfully
if independent reproduction or either archived comparison has not passed.

The clean-clone run is the independent reproduction, not an additional third
experiment. After publication, compare the published tree with the verified
tree and perform integrity and structural checks. Automatic CI runs these
checks only; it does not launch another simulation.

## Inference and Provenance

ARIMA reporting fits stationary ARIMA(2,0,0) by likelihood and uses analytical
covariance with the observed denominator. The existing critical-value rules,
HAC and orthogonality-inversion procedures remain in place. Shock-orientation
severity calibration retains its existing helper; reporting uses the
qualified conditional gradient retry only for nonconvergence.

`qualified_inference/provenance.json` records the transferred routines and
their preparation status. It is not a live release-acceptance certificate.
Acceptance is established by the separate execution and reproduction reports.
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
