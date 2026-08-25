# Replication Package for Weighting Geometry in Aggregate-Instrument Causal Analysis of Renewable Energy Policy and Emissions

**Author:** Saleh Dadjouy, University of Northern Colorado<br>
**Manuscript status:** Unpublished working paper, August 2026<br>
**Repository:** <https://github.com/SalehDadjouy/energy-policy-emissions-code>

This repository reproduces the computational evidence reported in the paper
from the included analysis-ready data. It contains the empirical TSLS and
Robust aggregate-instrument analyses, the principal
Monte Carlo simulation, the reported sensitivity analyses, denominator
diagnostics, the state-exposure figure, and validation checks for the values
displayed in the manuscript.

## Reproducibility Scope

The documented workflow reproduces the reported analysis from
`data/panel_lag2.csv`, `data/exposure_full.csv`, and
`data/exposure_restricted.csv`. It covers:

- empirical point estimates, first-stage diagnostics, and interval procedures;
- primary- and longer-window simulation results, with state exposure estimated
  from the learning window in every replication and design;
- paired estimator comparisons and denominator diagnostics;
- full- and restricted-panel simulation comparisons;
- finite-window weight-learning, target-anchor, aggregate-confounding, and
  omitted-shock-orientation sensitivities;
- the state-exposure figure; and
- automated checks against the reference results associated with this version.

The repository does not reconstruct `panel_lag2.csv` from the original agency
files. Those source records, the treatment-construction stages, and the
resulting analysis-ready panel are described in [`data/README.md`](data/README.md)
and in the paper. This distinction separates reproduction of the reported
analysis from reconstruction of the underlying data.

## Repository Contents

| Path | Purpose |
|---|---|
| [`data/`](data/) | Analysis-ready panel, full and restricted exposure profiles, variable definitions, source provenance, and data-rights statement |
| [`empirical.py`](empirical.py) | Empirical TSLS and Robust estimates, first-stage diagnostics, and interval procedures |
| [`run_simulation.py`](run_simulation.py) | Paired Monte Carlo simulation for the primary and longer synthetic windows |
| [`sim/`](sim/) | Simulation data-generating processes, estimators, weighting, and inference functions |
| [`analysis/`](analysis/) | Prespecified state-panel, confounding-grid, weight-learning, target-anchor, and orientation analyses |
| [`audit_denominator_tails.py`](audit_denominator_tails.py) | Denominator-tail diagnostics for shock-active simulation designs |
| [`make_exposure_figure.py`](make_exposure_figure.py) | State-exposure figure and plotted coefficients |
| [`validate_results.py`](validate_results.py) | Data-integrity and paper-value checks |
| [`reference_outputs/`](reference_outputs/) | Versioned empirical results, simulation records, diagnostics, figure, and checksums |
| [`RESEARCH_OUTPUTS.md`](RESEARCH_OUTPUTS.md) | Exact mapping from manuscript exhibits to programs and generated files |

## Reproduction Instructions

Python 3.11 is recommended. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduce.py
```

The final command runs the empirical analysis, the one-thousand-replication
principal simulation, every reported sensitivity analysis, the denominator
diagnostics, the exposure-figure build, and the validation checks. The
principal simulation estimates state exposure from the learning years before
constructing either estimator's weights. A successful full run ends with:

```text
PASS: generated data, empirical results, simulations, and displayed paper values agree.
```

For a structural check of the complete workflow with fewer Monte Carlo draws:

```bash
python reproduce.py --quick
```

The quick workflow verifies execution and file production. Its Monte Carlo
results are not the values reported in the paper.

To reproduce only the empirical analysis, principal simulation, diagnostics,
figure, and validation checks:

```bash
python reproduce.py --core-only
```

The archived sensitivity summaries remain available under
`reference_outputs/sensitivity/` and are validated by the default full
workflow.

## Computational Requirements

- Python 3.11
- Packages and exact versions in [`requirements.txt`](requirements.txt)
- Approximately 1 GB of free space for the repository, environment, and
  generated outputs
- No proprietary software, private credentials, or absolute filesystem paths

The sensitivity analyses are computationally intensive because they add
state-panel, parameter-grid, target, weight-learning, and orientation
replications to the principal simulation. On the documented macOS system, the
principal one-thousand-replication simulation finished in approximately seven
minutes after dependency installation. The quick workflow finished in
approximately two minutes. Installation time and computational runtime vary
by machine and network.

GitHub Actions runs syntax checks, validates the archived reference results,
and executes the quick workflow on Ubuntu with Python 3.11.

## Data Availability

The repository includes the three analysis-ready CSV files required by the
documented workflow. The underlying data originate from publicly available
records of the U.S. Environmental Protection Agency, U.S. Energy Information
Administration, U.S. Department of the Treasury, and U.S. Bureau of Economic
Analysis. [`data/README.md`](data/README.md) documents the files, variables,
transformations, original sources, and reuse conditions.

## Outputs and Verification

`python reproduce.py` writes regenerated files under `outputs/`. The paper-to-
output mapping appears in [`RESEARCH_OUTPUTS.md`](RESEARCH_OUTPUTS.md).

To validate an existing output directory directly:

```bash
python validate_results.py
```

The validation program checks the input files, empirical results, principal
simulation summaries, sensitivity summaries, and manuscript values. SHA-256
checksums in [`reference_outputs/SHA256SUMS`](reference_outputs/SHA256SUMS)
identify the reference files associated with this repository version.

## Citation and Versioning

The machine-readable citation record is [`CITATION.cff`](CITATION.cff). Until a
journal citation or archival DOI is available, the paper should be cited as an
unpublished working paper:

> Dadjouy, Saleh. 2026. “Weighting Geometry in Aggregate-Instrument Causal
> Analysis of Renewable Energy Policy and Emissions.” Unpublished working
> paper, University of Northern Colorado.

When citing the computational materials, identify the GitHub release or commit
used. A future archival release will add a persistent DOI without changing the
reproduction commands.

## License and Contact

The programs are released under the [MIT License](LICENSE). The included
analytical data retain the citation and reuse conditions of their underlying
public sources; see [`data/README.md`](data/README.md) and
[`data/RIGHTS.md`](data/RIGHTS.md).

Questions about the research materials may be directed to Saleh Dadjouy at
saleh.dadjouy@unco.edu.
