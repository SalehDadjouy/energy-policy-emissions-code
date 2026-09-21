# Manuscript Exhibits and Computational Outputs

This map follows the current SPP manuscript's table labels, checked against
its compiled auxiliary file on September 20, 2026. No manuscript was changed
to prepare this candidate. Paths below are archived reference files, not a
claim that a numerical qualification has been completed. The corresponding
generated files appear under the chosen run directory.

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
programs remain `empirical.py`, `run_simulation.py`,
`make_exposure_figure.py`, and the four programs under `analysis/`.
`audit_denominator_tails.py` supplies the denominator diagnostics.
The finite-window analysis must evaluate the included objective file instead
of rebuilding it during the proposed qualified workflow.

WORKFLOW.json declares the commands and generated artifacts. Table 7 uses
the bank's archived convergence summaries alongside the regenerated paired
evaluation; bank construction is an input boundary, not a newly executed
stage. Independent numerical reproduction and archived compatibility are
separate acceptance checks; neither is established by validating reference
files against their own checksums.
