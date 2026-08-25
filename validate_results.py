#!/usr/bin/env python3
"""Validate generated outputs against the values reported in the paper."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


# Tolerances cover cross-version drift observed up to 8.2e-11, far below manuscript reporting precision.
REFERENCE_RTOL = 1e-8
REFERENCE_ATOL = 1e-10


ROOT = Path(__file__).resolve().parent
DATA_HASHES = {
    "panel_lag2.csv": "4a5081df39f0c72c226dd6524007564d5761ca26655222339ae91da3003ecd0b",
    "exposure_restricted.csv": "47fd48cf6b6cb8c809afcb99b6acd78494473896ffda63c328684fdaecddb959",
    "exposure_full.csv": "b1c8eb131d5ee8210d30b7198b0b2d83bf912265511dab3d05cc12e12183f026",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def displayed(value: float, digits: int = 3) -> float:
    return float(f"{float(value):.{digits}f}")


def assert_displayed(actual: float, expected: float, *, label: str, digits: int = 3) -> None:
    if math.isclose(expected, 0.0, abs_tol=0.0):
        ok = abs(float(actual)) < 0.5 * 10 ** (-digits)
    else:
        ok = displayed(actual, digits) == float(expected)
    if not ok:
        fail(f"{label}: expected displayed value {expected:.{digits}f}, got {actual:.12g}")


def validate_data() -> None:
    for name, expected in DATA_HASHES.items():
        path = ROOT / "data" / name
        if sha256(path) != expected:
            fail(f"Input hash mismatch: {path}")

    panel = pd.read_csv(ROOT / "data" / "panel_lag2.csv").dropna()
    panel = panel[panel["year"].between(2008, 2022)].copy()
    years = np.arange(2008, 2023)
    z = panel.drop_duplicates("year").set_index("year").reindex(years)["Z_t_lag2"].to_numpy(float)
    zc = z - z.mean()
    exposure_rows = []
    for state, group in panel.groupby("state", sort=True):
        w = group.set_index("year").reindex(years)["W_it_lag2"].to_numpy(float)
        exposure_rows.append((state, float(((w - w.mean()) @ zc) / (zc @ zc))))
    calculated = pd.DataFrame(exposure_rows, columns=["unit", "D_calculated"])
    restricted = pd.read_csv(ROOT / "data" / "exposure_restricted.csv")
    merged = restricted.merge(calculated, on="unit", validate="one_to_one")
    if set(calculated["unit"]) - {"CA", "VT"} != set(restricted["unit"]):
        fail("Restricted exposure file does not contain exactly the 49 non-CA/VT jurisdictions.")
    if not np.allclose(merged["D_tsls"], merged["D_calculated"], rtol=0.0, atol=1e-12):
        fail("Restricted exposure coefficients do not reproduce from the analytical panel.")

    full = pd.read_csv(ROOT / "data" / "exposure_full.csv")
    full_merged = full.merge(calculated, on="unit", validate="one_to_one")
    if set(full["unit"]) != set(calculated["unit"]):
        fail("Full exposure file does not contain all 50 states and the District of Columbia.")
    if not np.allclose(full_merged["D_tsls"], full_merged["D_calculated"], rtol=0.0, atol=1e-12):
        fail("Full exposure coefficients do not reproduce from the analytical panel.")


def validate_reference_hashes() -> None:
    checksum_file = ROOT / "reference_outputs" / "SHA256SUMS"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if sha256(path) != expected:
            fail(f"Frozen reference hash mismatch: {relative}")


def row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    keep = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        keep &= frame[column].eq(value).to_numpy()
    subset = frame.loc[keep]
    if len(subset) != 1:
        fail(f"Expected one row for {conditions}, found {len(subset)}")
    return subset.iloc[0]


def validate_empirical(path: Path) -> None:
    frame = pd.read_csv(path)
    expected = {
        ("full", "tsls"): (-0.696, 4.001, 3.679, 0.937, 0.519, 0.273, 11.784),
        ("full", "siv"): (-2.216, 4.194, 2.622, 1.018, 0.547, 0.308, 10.908),
        ("restricted", "tsls"): (-1.136, 4.945, 4.185, 0.893, 0.539, 0.289, 9.566),
        ("restricted", "siv"): (-2.613, 4.094, 2.815, 0.891, 0.602, 0.348, 6.553),
    }
    for (sample, estimator), values in expected.items():
        r = row(frame, label=sample)
        tau, arima_se, hac_se, pi, ols_se, pi_hac_se, f_hac = values
        assert_displayed(r[f"{estimator}_tau"], tau, label=f"{sample} {estimator} tau")
        assert_displayed(r[f"{estimator}_se_arima"], arima_se, label=f"{sample} {estimator} ARIMA SE")
        assert_displayed(r[f"{estimator}_se_hac"], hac_se, label=f"{sample} {estimator} HAC SE")
        assert_displayed(r[f"{estimator}_pi"], pi, label=f"{sample} {estimator} first stage")
        assert_displayed(r[f"{estimator}_pi_se_ols"], ols_se, label=f"{sample} {estimator} OLS first-stage SE")
        assert_displayed(r[f"{estimator}_pi_se_hac"], pi_hac_se, label=f"{sample} {estimator} HAC first-stage SE")
        assert_displayed(r[f"{estimator}_f_hac"], f_hac, label=f"{sample} {estimator} HAC F")
        for lag in (0, 1):
            lo = float(r[f"{estimator}_ar_l{lag}_lo"])
            hi = float(r[f"{estimator}_ar_l{lag}_hi"])
            if not lo <= 0.0 <= hi:
                fail(f"{sample} {estimator} lag-{lag} orthogonality set does not include zero.")


def validate_simulation(summary_path: Path, paired_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    arima = summary[summary["method"].eq("ARIMA-Z")]
    primary = {
        "Basic": {"TSLS": (0.001, 0.024, 0.900), "SIV": (0.001, 0.026, 0.902)},
        "GFE": {"TSLS": (0.020, 0.566, 0.849), "SIV": (0.000, 0.090, 0.884)},
        "Aggregate Shock": {"TSLS": (0.357, 2.703, 0.530), "SIV": (0.012, 0.066, 0.776)},
        "GFE + Agg. Shock": {"TSLS": (0.880, 9.586, 0.636), "SIV": (0.061, 0.247, 0.809)},
    }
    longer = {
        "Basic": {"TSLS": (0.000, 0.007, 0.929), "SIV": (0.000, 0.019, 0.918)},
        "GFE": {"TSLS": (-0.002, 0.113, 0.914), "SIV": (0.000, 0.012, 0.926)},
        "Aggregate Shock": {"TSLS": (0.432, 0.456, 0.179), "SIV": (0.002, 0.024, 0.830)},
        "GFE + Agg. Shock": {"TSLS": (0.428, 0.467, 0.309), "SIV": (0.007, 0.020, 0.820)},
    }
    for config, expected in (("finite_length", primary), ("longer_length", longer)):
        for branch, estimators in expected.items():
            for estimator, (bias, rmse, rate) in estimators.items():
                r = row(arima, config=config, branch=branch, estimator=estimator)
                assert_displayed(r["mean_bias"], bias, label=f"{config} {branch} {estimator} bias")
                assert_displayed(r["rmse"], rmse, label=f"{config} {branch} {estimator} RMSE")
                assert_displayed(r["coverage"], rate, label=f"{config} {branch} {estimator} rate")

    coverage_expected = {
        "Basic": {"ARIMA-Z": (0.900, 0.902), "HAC": (0.824, 0.835), "AR": (0.834, 0.840)},
        "GFE": {"ARIMA-Z": (0.849, 0.884), "HAC": (0.802, 0.785), "AR": (0.811, 0.812)},
        "Aggregate Shock": {"ARIMA-Z": (0.530, 0.776), "HAC": (0.416, 0.655), "AR": (0.467, 0.677)},
        "GFE + Agg. Shock": {"ARIMA-Z": (0.636, 0.809), "HAC": (0.495, 0.681), "AR": (0.564, 0.714)},
    }
    for branch, methods in coverage_expected.items():
        for method, (tsls_rate, siv_rate) in methods.items():
            for estimator, expected_rate in (("TSLS", tsls_rate), ("SIV", siv_rate)):
                r = row(summary, method=method, config="finite_length", branch=branch, estimator=estimator)
                assert_displayed(r["coverage"], expected_rate, label=f"{branch} {method} {estimator} rate")

    paired = pd.read_csv(paired_path)
    paired_expected = {
        "Basic": (-0.000, 0.000, 0.001, 0.000, 0.002, 0.002),
        "GFE": (-0.020, 0.018, -0.323, 0.013, -0.476, 0.035),
        "Aggregate Shock": (-0.344, 0.085, -0.697, 0.082, -2.637, 0.246),
        "GFE + Agg. Shock": (-0.819, 0.302, -1.016, 0.301, -9.339, 0.173),
    }
    columns = (
        "delta_signed_bias_siv_minus_tsls",
        "mcse_delta_signed_bias",
        "delta_abs_bias_siv_minus_tsls",
        "mcse_delta_abs_bias",
        "delta_rmse_siv_minus_tsls",
        "delta_coverage_siv_minus_tsls",
    )
    for branch, values in paired_expected.items():
        r = row(paired, config="finite_length", branch=branch)
        for column, expected_value in zip(columns, values):
            assert_displayed(r[column], expected_value, label=f"{branch} {column}")


def compare_reference(generated: Path, reference: Path, *, keys: list[str]) -> None:
    actual = pd.read_csv(generated).sort_values(keys).reset_index(drop=True)
    expected = pd.read_csv(reference).sort_values(keys).reset_index(drop=True)
    common = [column for column in expected.columns if column in actual.columns]
    if len(actual) != len(expected):
        fail(f"Row-count mismatch: {generated} ({len(actual)}) vs {reference} ({len(expected)})")
    for column in common:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(
                actual[column],
                expected[column],
                rtol=REFERENCE_RTOL,
                atol=REFERENCE_ATOL,
                equal_nan=True,
            ):
                fail(f"Numerical mismatch in {generated}, column {column}")
        elif not actual[column].fillna("").equals(expected[column].fillna("")):
            fail(f"Text mismatch in {generated}, column {column}")


def validate_sensitivity_outputs(outputs: Path) -> None:
    """Compare regenerated article sensitivities with their archived summaries."""

    reference = ROOT / "reference_outputs" / "sensitivity"
    comparisons = {
        "state_panel": {
            "summary.csv": ["sample", "method", "config", "branch", "estimator"],
            "paired_differences.csv": ["sample", "method", "config", "branch"],
            "component_summary.csv": ["sample", "config", "branch"],
        },
        "confounding_grid": {
            "summary.csv": [
                "h_z_corr",
                "theta_y_scale_multiplier",
                "method",
                "config",
                "branch",
                "estimator",
            ],
            "paired_differences.csv": [
                "h_z_corr",
                "theta_y_scale_multiplier",
                "method",
                "config",
                "branch",
            ],
            "component_summary.csv": ["h_z_corr", "theta_y_scale_multiplier", "config", "branch"],
        },
        "finite_window_weight_learning": {
            "population_weights.csv": ["branch", "state_index"],
            "weight_summary.csv": ["branch", "variant"],
            "estimate_summary.csv": ["branch", "variant", "method"],
            "paired_effects.csv": ["branch", "candidate", "reference", "method"],
            "population_convergence.csv": ["branch", "draws", "comparison"],
        },
        "target_anchor": {
            "summary.csv": ["anchor_mode", "target_role", "config", "branch", "method", "estimator"],
            "estimator_differences.csv": ["anchor_mode", "target_role", "config", "branch", "method"],
            "target_contrasts.csv": ["anchor_mode", "endpoint_role", "midpoint_role", "branch", "method", "estimator"],
            "weight_target_contrast_summary.csv": ["anchor_mode", "target_role", "branch", "estimator"],
        },
        "shock_orientation": {
            "summary.csv": ["method", "config", "branch", "arm", "severity", "estimator"],
            "paired_differences.csv": ["method", "config", "branch", "arm", "severity"],
            "geometry_summary.csv": ["config", "branch", "arm", "severity"],
        },
    }
    for directory, files in comparisons.items():
        for name, keys in files.items():
            compare_reference(
                outputs / directory / name,
                reference / directory / name,
                keys=keys,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated results against the paper and frozen outputs.")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--sensitivity-outputs",
        type=Path,
        help="Optional directory containing regenerated sensitivity analyses.",
    )
    args = parser.parse_args()

    validate_data()
    validate_reference_hashes()
    empirical = args.outputs / "empirical" / "results_summary.csv"
    simulation = args.outputs / "simulation"
    validate_empirical(empirical)
    validate_simulation(simulation / "summary.csv", simulation / "paired_differences.csv")
    if not args.skip_reference:
        compare_reference(
            empirical,
            ROOT / "reference_outputs" / "empirical" / "results_summary.csv",
            keys=["label"],
        )
        compare_reference(
            simulation / "summary.csv",
            ROOT / "reference_outputs" / "simulation" / "summary.csv",
            keys=["method", "config", "branch", "estimator"],
        )
        compare_reference(
            simulation / "paired_differences.csv",
            ROOT / "reference_outputs" / "simulation" / "paired_differences.csv",
            keys=["config", "branch"],
        )
        for name in ("weights_full.csv", "weights_restricted.csv"):
            compare_reference(
                args.outputs / "empirical" / "lag2" / name,
                ROOT / "reference_outputs" / "empirical" / name,
                keys=["unit"],
            )
    if args.sensitivity_outputs is not None:
        validate_sensitivity_outputs(args.sensitivity_outputs)
    print("PASS: generated data, empirical results, simulations, and displayed paper values agree.")


if __name__ == "__main__":
    main()
