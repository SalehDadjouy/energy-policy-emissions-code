# Analysis Data and Provenance

## Reproduction Boundary

The repository includes the analysis-ready files used by the empirical and
simulation programs. The documented top-level workflow begins from these
files. It does not download the original agency records or reconstruct the
analysis panel from those records.

This boundary matters for interpreting the repository's reproducibility claim:
the empirical results, simulations, diagnostics, and figure can be reproduced
without external data, while independent reconstruction of the treatment and
panel requires the source records and construction procedures described in the
paper.

## Included Files

### `panel_lag2.csv`

This file is the analysis-ready state-year panel. It contains 1,479 rows for
the 50 states and the District of Columbia over 1994-2022. The reported
empirical analysis uses the 2008-2022 effective window.

| Variable | Definition |
|---|---|
| `state` | Two-letter state or District of Columbia abbreviation |
| `year` | Final year of the two-year transformed observation |
| `Y_it_lag2` | Two-year log change in state electric-power-sector combustion emissions, measured in carbon dioxide equivalent units |
| `W_it_lag2` | Two-year change in wind-and-solar electricity attributed to state retail sales, divided by base-year total retail electricity sales |
| `Z_t_lag2` | Two-year change in the national renewable-subsidy series, divided by base-year national gross domestic product |

The first two transformed years for a state are missing when the required
two-year lag is unavailable. The empirical programs select the effective
window and complete observations used in the paper.

### `exposure_full.csv` and `exposure_restricted.csv`

These files contain the full-panel exposure profile and the 49-state profile
used after California and Vermont are excluded under the diagnostic
restriction. `validate_results.py` re-estimates both exposure profiles from
`panel_lag2.csv` and requires agreement to numerical precision.

| Variable | Definition |
|---|---|
| `unit` | Two-letter state abbreviation |
| `w_tsls` | Centered exposure weight, `D_tsls` minus its 49-state mean, stored on the unnormalized TSLS scale |
| `w_rob` | Robust aggregation weight learned from the restricted empirical panel |
| `D_tsls` | State exposure coefficient used by the TSLS construction and simulation calibration |
| `D_rob` | Duplicate of `D_tsls`, retained for compatibility with the estimator-specific input schema; Robust uses the same exposure constraint |

## Original Data Sources

The analysis-ready panel combines the following public records. The paper
provides the complete treatment-construction method and bibliographic
citations.

| Component | Source | Public location |
|---|---|---|
| State electric-power greenhouse-gas emissions | U.S. Environmental Protection Agency, *State Greenhouse Gas Emissions and Removals by Economic Sector* | <https://www.epa.gov/ghgemissions/state-ghg-emissions-and-removals> |
| Emissions methodology | U.S. Environmental Protection Agency, *Methodology Report: Inventory of U.S. Greenhouse Gas Emissions and Sinks: State 1990-2022* | <https://www.epa.gov/ghgemissions/methodology-report-inventory-us-greenhouse-gas-emissions-and-sinks-state-1990-2022> |
| Grid operations, generation, and interchange | U.S. Energy Information Administration, Form EIA-930 | <https://www.eia.gov/electricity/gridmonitor/about.php> |
| Retail sales and balancing-authority service territories | U.S. Energy Information Administration, Form EIA-861 detailed data | <https://www.eia.gov/electricity/data/eia861/> |
| State generation, retail sales, and interstate-flow accounting inputs | U.S. Energy Information Administration, State Energy Data System | <https://www.eia.gov/state/seds/seds-technical-notes.php> |
| Section 1603 renewable-energy payments | U.S. Department of the Treasury, Section 1603 program records | <https://home.treasury.gov/policy-issues/financial-markets-financial-institutions-and-fiscal-service/1603-program-payments-for-specified-energy-property-in-lieu-of-tax-credits> |
| Renewable-energy tax expenditures | U.S. Department of the Treasury, federal tax-expenditure estimates | <https://home.treasury.gov/system/files/131/Tax-Expenditures-FY2024-update.pdf> |
| National gross domestic product | U.S. Bureau of Economic Analysis, *Gross Domestic Product* | <https://www.bea.gov/data/gdp/gross-domestic-product> |

## Construction Summary

The outcome is formed as

```text
log(emissions in year t) - log(emissions in year t - 2).
```

The treatment attributes wind-and-solar electricity to the retail demand it
serves. For 2019-2022, the construction uses EIA-930 generation and interchange
records with a proportional-sharing flow-tracing procedure. Earlier years use
a State Energy Data System proxy calibrated to the flow-traced overlap period.
The treatment transformation divides the two-year change in attributed
wind-and-solar retail sales by base-year total retail electricity sales.

The instrument combines Section 1603 payments and federal renewable-energy tax
expenditures. Its two-year change is divided by base-year national gross
domestic product. No additional normalization is applied in the empirical
analysis.

## Access and Reuse

The author obtained the underlying records from the public agency sources
listed above and includes the derived analytical files for research
reproduction. Users should cite the original agencies and comply with any
source-specific terms applicable at the time of reuse.

The repository's MIT license applies to the programs. It does not supersede
rights or attribution requirements attached to the underlying source records.
The derived CSV files are provided for verification and replication of the
reported analysis. [`RIGHTS.md`](RIGHTS.md) states the applicable attribution
and reuse boundary.
