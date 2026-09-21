# Manuscript Exhibits and Computational Outputs

This map connects the manuscript's figures and tables to the package's
reference outputs. The exhibit labels were checked against the manuscript
on September 20, 2026. Independently generated results appear under the run
directory selected by the user.

| Exhibit | Content | Reference file under `reference_outputs/` |
|---|---|---|
| Figure 1 | State exposure coefficients | `figures/exposure_coefficients.csv`; `figures/paper_exposure_distribution.pdf` |
| Table 1 | Empirical emissions estimates and instrument-treatment diagnostics | `empirical/results_summary.csv` |
| Table 2 | Observed and simulated aggregate-equation diagnostics | `empirical/results_summary.csv`; `simulation/raw.csv`; `simulation/summary.csv` |
| Table 3 | Point-estimation performance by panel and simulation design | `simulation/summary.csv`; `sensitivity/state_panel/summary.csv` |
| Table 4 | Interval coverage by panel, procedure, and design | `simulation/summary.csv`; `sensitivity/state_panel/summary.csv` |
| Table 5 | Bias and root mean squared error differences across confounding settings | `sensitivity/confounding_grid/paired_differences.csv` |
| Table 6 | Interval coverage across confounding settings | `sensitivity/confounding_grid/summary.csv` |
| Table 7 | Finite-window weight learning and Robust performance | `sensitivity/finite_window_weight_learning/weight_summary.csv`; `estimate_summary.csv`; `paired_effects.csv`; `population_convergence.csv` in the same directory |
| Table S1 | Performance differences across simulation targets | `sensitivity/target_anchor/estimator_differences.csv` |
| Table S2 | Performance differences under estimator-oriented shock loadings | `sensitivity/shock_orientation/summary.csv`; `paired_differences.csv`; `geometry_summary.csv` in the same directory |
| Table S3 | Longer-window point and interval performance | `simulation/summary.csv` |

The manuscript's typesetting is not rebuilt by this package. The analysis
programs are `empirical.py`, `run_simulation.py`,
`make_exposure_figure.py`, and the four programs under `analysis/`.
`audit_denominator_tails.py` supplies the denominator diagnostics.
The finite-window analysis evaluates the included fixed objective bank.

`WORKFLOW.json` specifies the commands and generated artifacts. Table 7 uses
the bank's archived convergence summaries alongside the paired evaluation
produced by the workflow. Construction of the bank is outside the workflow.
The comparison command separately evaluates agreement between independent
runs and agreement with the archived reference results. Checksums establish
file integrity.
