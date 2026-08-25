# Manuscript Exhibits and Computational Outputs

This document maps each computational exhibit in *Weighting Geometry in
Aggregate-Instrument Causal Analysis of Renewable Energy Policy and Emissions*
to the program and generated file that supports it. Table and figure numbers
refer to the August 2026 working-paper version.

## Exhibit Map

| Manuscript exhibit | Reported content | Program | Generated file |
|---|---|---|---|
| Figure 1. State Exposure Coefficients for Wind-and-Solar Retail Sales Intensity | Full-panel exposure coefficients and ordered state ranking | `python make_exposure_figure.py` | `outputs/figures/paper_exposure_distribution.pdf`; plotted values in `outputs/figures/exposure_coefficients.csv` |
| Table 1. Empirical First-Stage Slopes, Standard Errors, and HAC F Statistics | Instrument-treatment slopes and first-stage diagnostics by estimator and panel | `python empirical.py` | `outputs/empirical/results_summary.csv` |
| Table 2. Empirical Point Estimates with ARIMA, HAC, and Orthogonality-Inversion Inference | Point estimates, standard errors, p-values, and confidence sets | `python empirical.py` | `outputs/empirical/results_summary.csv` |
| Table 3. Empirical and Simulated First-Stage, Standard-Error, and Residual Diagnostics in the Primary Window | Restricted-panel empirical diagnostics and simulated GFE + Aggregate Shock distributions | `python empirical.py`; `python run_simulation.py` | `outputs/empirical/results_summary.csv`; `outputs/simulation/raw.csv`; `outputs/simulation/summary.csv` |
| Table 4. Finite-Window Consequences of Weight Learning for the Robust Estimator | Reference, full-window-exposure, and learning-window-exposure weight and interval comparisons | `python analysis/finite_window_weight_learning.py --stage all` | `outputs/sensitivity/finite_window_weight_learning/weight_summary.csv`; `estimate_summary.csv`; `paired_effects.csv`; `population_convergence.csv` |
| Table 5. Estimator Differences Across Simulation Target Anchors in the Primary Window | Paired estimator differences at three empirically defined targets | `python analysis/target_anchor_sensitivity.py --execute-full` | `outputs/sensitivity/target_anchor/estimator_differences.csv`; supporting summaries in the same directory |
| Table 6. Primary-Window Bias and Root Mean Squared Error in the Full and Restricted Panels | Bias and RMSE across four designs and two state panels | `python run_simulation.py`; `python analysis/state_panel_and_confounding_sensitivity.py --mode sample` | `outputs/simulation/summary.csv`; `outputs/sensitivity/state_panel/summary.csv` |
| Table 7. Bias and Root Mean Squared Error Differences Across Aggregate-Confounding Settings | Paired point-performance differences over the aggregate-confounding grid | `python analysis/state_panel_and_confounding_sensitivity.py --mode grid` | `outputs/sensitivity/confounding_grid/paired_differences.csv` |
| Table 8. Estimator-Oriented Aggregate-Shock Diagnostic in the Combined Design | Point and interval results under TSLS-oriented and Robust-oriented shock loadings | `python analysis/shock_orientation_sensitivity.py` | `outputs/sensitivity/shock_orientation/summary.csv`; `paired_differences.csv`; `geometry_summary.csv` |
| Table 9. Primary-Window Interval Coverage in the Full and Restricted Panels | ARIMA, HAC, and orthogonality-inversion coverage by design and state panel | `python run_simulation.py`; `python analysis/state_panel_and_confounding_sensitivity.py --mode sample` | `outputs/simulation/summary.csv`; `outputs/sensitivity/state_panel/summary.csv` |
| Table 10. Interval Coverage Across Aggregate-Confounding Settings | ARIMA coverage over the aggregate-confounding grid | `python analysis/state_panel_and_confounding_sensitivity.py --mode grid` | `outputs/sensitivity/confounding_grid/summary.csv` |
| Table 11. Longer-Window Bias and Root Mean Squared Error by Simulation Design | Fixed-state, larger-time-dimension point performance | `python run_simulation.py` | `outputs/simulation/summary.csv` |
| Table 12. Longer-Window Interval Coverage by Inference Procedure and Simulation Design | Fixed-state, larger-time-dimension ARIMA, HAC, and orthogonality-inversion coverage | `python run_simulation.py` | `outputs/simulation/summary.csv` |

## Supporting Computational Evidence

| Paper content | Program | Generated file |
|---|---|---|
| Empirical aggregation weights | `python empirical.py` | `outputs/empirical/weights_full.csv`; `outputs/empirical/weights_restricted.csv` |
| Simulation component, calibration, and learning-window exposure-estimation record | `python run_simulation.py` | `outputs/simulation/component_summary.csv`; `outputs/simulation/manifest.json` |
| Full replication-level simulation records | `python run_simulation.py` | `outputs/simulation/raw.csv` |
| Denominator-tail discussion | `python audit_denominator_tails.py` | `outputs/simulation/denominator_tail_summary.csv`; `denominator_tail_conditional_summary.csv`; `denominator_tail_top_records.csv` |
| State-panel paired estimator differences | `python analysis/state_panel_and_confounding_sensitivity.py --mode sample` | `outputs/sensitivity/state_panel/paired_differences.csv` |
| Aggregate-confounding component diagnostics | `python analysis/state_panel_and_confounding_sensitivity.py --mode grid` | `outputs/sensitivity/confounding_grid/component_summary.csv` |
| Target-anchor contrasts and weight stability | `python analysis/target_anchor_sensitivity.py --execute-full` | `outputs/sensitivity/target_anchor/target_contrasts.csv`; `weight_target_contrast_summary.csv` |

## Complete Workflow

Run all empirical, simulation, sensitivity, diagnostic, figure, and validation
stages with:

```bash
python reproduce.py
```

`validate_results.py` checks the values displayed in the manuscript and
compares regenerated files with the reference results. When the sensitivity
analyses have been reproduced, include their output directory:

```bash
python validate_results.py --sensitivity-outputs outputs/sensitivity
```

The archived reference files appear under `reference_outputs/`. Their SHA-256
checksums are recorded in `reference_outputs/SHA256SUMS`.
