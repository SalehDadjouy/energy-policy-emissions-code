#!/usr/bin/env python3
"""Evaluate sensitivity to three empirically anchored simulation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import empirical as empirical_analysis  # noqa: E402
import paper_config as active  # noqa: E402
import run_simulation as production  # noqa: E402
from simulation_reporting import (  # noqa: E402
    BRANCHES,
    _crit,
)
from sim.estimators import (  # noqa: E402
    aggregate_series,
    ols_slope_with_intercept,
    unit_level_slope_vector,
)
from sim.helpers import shock_bias_component  # noqa: E402
from sim.inference import ar_style_test, hac_delta_se, raw_z_arp_design_based_se  # noqa: E402
from sim.model_class import BRANCH_SWITCHES, ModelClassComponents, build_branch_panel, estimate_branch_panel  # noqa: E402


PROTOCOL_PATH = ROOT / "analysis" / "protocols" / "target_anchor_sensitivity_v1.json"
ANCHOR_MODES = ("fixed_template", "reanchored_template")
METHODS = ("ARIMA-Z", "HAC", "AR")
POINT_METHODS = ("ARIMA-Z", "HAC")
KEY_COLUMNS = ["rep", "config", "T0", "T1", "branch", "estimator", "method"]
REFERENCE_NUMERIC_COLUMNS = [
    "weight_corr",
    "shock",
    "iv_moment_tau",
    "iv_moment_abs_tau",
    "iv_cov_zw",
    "iv_moment_bias_ratio",
    "bias_component_total",
    "bias_component_gfe",
    "bias_component_shock",
    "bias_component_eps",
    "tau",
    "bias",
    "se",
    "pi",
    "critical",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_map(protocol: dict[str, Any]) -> dict[str, float]:
    return {
        str(item["role"]): float(item["value"])
        for item in protocol["empirical_targets"]
    }


def midpoint_role(protocol: dict[str, Any]) -> str:
    roles = [
        str(item["role"])
        for item in protocol["empirical_targets"]
        if "midpoint" in str(item["role"])
    ]
    if roles != ["restricted_midpoint"]:
        raise AssertionError(f"Expected one restricted midpoint role, observed {roles}.")
    return roles[0]


def recompute_empirical_targets() -> dict[str, float]:
    data = empirical_analysis.load_panel(ROOT / "data" / "panel_lag2.csv")
    data = data[
        (data["time"] >= empirical_analysis.YEAR_START)
        & (data["time"] <= empirical_analysis.YEAR_END)
        & (~data["unit"].isin(empirical_analysis.RESTRICTED))
    ].copy()
    Y, W, Z, _units, _years = empirical_analysis.pivot(data)
    exposure = empirical_analysis.exposure_profile(W, Z)
    weights_tsls = empirical_analysis.tsls_weights(exposure)
    weights_robust = empirical_analysis.robust_siv_weights(Y, W, Z, exposure)
    post = slice(empirical_analysis.T0, None)

    def estimate(weights: np.ndarray) -> float:
        return float(
            empirical_analysis.slope_ratio(
                empirical_analysis.aggregate(Y[:, post], weights),
                empirical_analysis.aggregate(W[:, post], weights),
                Z[empirical_analysis.T0 :],
            )["tau"]
        )

    tsls = estimate(weights_tsls)
    robust = estimate(weights_robust)
    return {
        "restricted_tsls": tsls,
        "restricted_midpoint": 0.5 * (tsls + robust),
        "restricted_robust": robust,
    }


def verify_reproduction_context(protocol: dict[str, Any], tolerance: float = 5e-13) -> dict[str, Any]:
    observed_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "statsmodels": statsmodels.__version__,
    }
    reference = protocol["production_reference"]
    reference_dir = ROOT / str(reference["output_directory"])
    required = tuple(reference["reference_files"])
    missing = [filename for filename in required if not (reference_dir / filename).is_file()]
    if missing:
        raise AssertionError(f"Principal simulation references are missing: {missing}.")

    expected_targets = target_map(protocol)
    observed_targets = recompute_empirical_targets()
    for role, expected in expected_targets.items():
        observed = observed_targets[role]
        if abs(observed - expected) > tolerance:
            raise AssertionError(f"Empirical target mismatch for {role}: {observed} != {expected}.")
    midpoint = 0.5 * (
        expected_targets["restricted_tsls"] + expected_targets["restricted_robust"]
    )
    if abs(midpoint - expected_targets["restricted_midpoint"]) > tolerance:
        raise AssertionError("Frozen midpoint is not the arithmetic midpoint of the endpoint targets.")
    if abs(float(reference["baseline_target"]) - expected_targets["restricted_midpoint"]) > tolerance:
        raise AssertionError("Frozen production target does not match the empirical midpoint.")
    if abs(float(active.TAU) - expected_targets["restricted_midpoint"]) > tolerance:
        raise AssertionError("Active simulation target does not match the empirical midpoint.")

    return {
        "environment": observed_environment,
        "empirical_targets": observed_targets,
        "reference_sha256": {
            filename: sha256_file(reference_dir / filename)
            for filename in required
        },
    }


def empirical_svd_base(tau_anchor: float) -> dict[str, Any]:
    panel = production._load_restricted_empirical_panel()
    exposure = np.asarray(panel["exposure"], dtype=float)
    z = np.asarray(panel["Z"], dtype=float)
    W = np.asarray(panel["W"], dtype=float)
    Y = np.asarray(panel["Y"], dtype=float)
    W_resid = production._two_way_residual(W - np.outer(exposure, z))
    Y_resid = production._two_way_residual(Y - float(tau_anchor) * W)
    lw_loadings, lw_scores, lowrank_w = production._svd_parts(W_resid, int(active.PARAMS.rank))
    ly_loadings, ly_scores, lowrank_y = production._svd_parts(Y_resid, int(active.PARAMS.rank))
    return {
        **panel,
        "lw_loadings": lw_loadings,
        "lw_scores": lw_scores,
        "ly_loadings": ly_loadings,
        "ly_scores": ly_scores,
        "lowrank_w_emp": lowrank_w,
        "lowrank_y_emp": lowrank_y,
        "stats": {
            "empirical_t": float(z.size),
            "rank": float(active.PARAMS.rank),
            "tau_anchor": float(tau_anchor),
            "sd_lowrank_w_emp": float(np.std(lowrank_w, ddof=0)),
            "sd_lowrank_y_emp": float(np.std(lowrank_y, ddof=0)),
            "sd_w_residual": float(np.std(W_resid, ddof=0)),
            "sd_y_residual": float(np.std(Y_resid, ddof=0)),
            "share_w_residual_rank2": float(np.sum(lowrank_w * lowrank_w) / np.sum(W_resid * W_resid)),
            "share_y_residual_rank2": float(np.sum(lowrank_y * lowrank_y) / np.sum(Y_resid * Y_resid)),
        },
    }


def build_geometries(protocol: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    reference = protocol["production_reference"]
    targets = target_map(protocol)
    midpoint = midpoint_role(protocol)
    baseline_base = empirical_svd_base(float(reference["baseline_target"]))
    geometries: dict[tuple[str, str], dict[str, Any]] = {}
    for anchor_mode in ANCHOR_MODES:
        for role, tau in targets.items():
            empirical = baseline_base if anchor_mode == "fixed_template" else empirical_svd_base(tau)
            geometries[(anchor_mode, role)] = production._fixed_geometry(
                config=str(reference["configuration"]),
                T0=int(reference["T0"]),
                T1=int(reference["T1"]),
                seed=int(protocol["paired_execution"]["seed"]),
                empirical=empirical,
                gfe_scale_mode=str(reference["gfe_scale_mode"]),
                gfe_scale_multiplier=1.0,
                gfe_w_multiplier=float(reference["gfe_w_multiplier"]),
                gfe_y_multiplier=float(reference["gfe_y_multiplier"]),
                shock_w_multiplier=float(reference["shock_w_multiplier"]),
                shock_y_multiplier=float(reference["shock_y_multiplier"]),
            )

    baseline = geometries[("fixed_template", midpoint)]
    max_differences: dict[str, float] = {}
    for key, geometry in geometries.items():
        label = "|".join(key)
        max_differences[f"{label}|lowrank_w"] = float(
            np.max(np.abs(np.asarray(geometry["lowrank_w"]) - np.asarray(baseline["lowrank_w"])))
        )
        max_differences[f"{label}|theta_w"] = float(
            np.max(np.abs(np.asarray(geometry["theta_w"]) - np.asarray(baseline["theta_w"])))
        )
        max_differences[f"{label}|theta_y"] = float(
            np.max(np.abs(np.asarray(geometry["theta_y"]) - np.asarray(baseline["theta_y"])))
        )
        if key[0] == "fixed_template":
            max_differences[f"{label}|lowrank_y_fixed"] = float(
                np.max(np.abs(np.asarray(geometry["lowrank_y"]) - np.asarray(baseline["lowrank_y"])))
            )
    midpoint_reanchored = geometries[("reanchored_template", midpoint)]
    max_differences["midpoint_anchor_mode_lowrank_y"] = float(
        np.max(
            np.abs(
                np.asarray(midpoint_reanchored["lowrank_y"])
                - np.asarray(baseline["lowrank_y"])
            )
        )
    )
    if max(max_differences.values(), default=0.0) > 5e-12:
        failed = {name: value for name, value in max_differences.items() if value > 5e-12}
        raise AssertionError(f"Frozen geometry identity check failed: {failed}")
    return geometries


def draw_components(
    protocol: dict[str, Any],
    *,
    rep: int,
    geometry: dict[str, Any],
    exposure: np.ndarray,
) -> ModelClassComponents:
    reference = protocol["production_reference"]
    return production._draw_components(
        config=str(reference["configuration"]),
        rep=int(rep),
        seed=int(protocol["paired_execution"]["seed"]),
        T0=int(reference["T0"]),
        T1=int(reference["T1"]),
        exposure=np.asarray(exposure, dtype=float),
        geometry=geometry,
    )


def component_biases(
    components: ModelClassComponents,
    panel: Any,
    weights: np.ndarray,
    *,
    tau_true: float,
    T0: int,
    T1: int,
) -> dict[str, float]:
    post = slice(int(T0), int(T0) + int(T1))
    z = np.asarray(components.Z, dtype=float)[post]
    gfe_on, shock_on = BRANCH_SWITCHES[panel.branch]

    def slope(component: np.ndarray) -> float:
        return ols_slope_with_intercept(aggregate_series(component[:, post], weights), z)

    denom = ols_slope_with_intercept(aggregate_series(panel.W[:, post], weights), z)
    if abs(denom) <= 1e-15:
        raise ValueError("Post-learning weighted denominator is too close to zero.")
    gfe_y = components.lowrank_y if gfe_on else np.zeros_like(components.lowrank_y)
    shock_y = components.shock_y if shock_on else np.zeros_like(components.shock_y)
    total_y = panel.Y - float(tau_true) * panel.W
    return {
        "bias_component_total": float(slope(total_y) / denom),
        "bias_component_gfe": float(slope(gfe_y) / denom),
        "bias_component_shock": float(slope(shock_y) / denom),
        "bias_component_eps": float(slope(components.eps_y) / denom),
    }


def iv_moment_fields(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    tau_true: float,
) -> dict[str, float]:
    z_centered = np.asarray(z, dtype=float) - float(np.mean(z))
    w_centered = np.asarray(w, dtype=float) - float(np.mean(w))
    residual = np.asarray(y, dtype=float) - float(tau_true) * np.asarray(w, dtype=float)
    moment = float(np.mean(z_centered * residual))
    covariance_zw = float(np.mean(z_centered * w_centered))
    return {
        "iv_moment_tau": moment,
        "iv_moment_abs_tau": abs(moment),
        "iv_cov_zw": covariance_zw,
        "iv_moment_bias_ratio": moment / covariance_zw if abs(covariance_zw) > 1e-15 else float("nan"),
    }


def common_component_difference(component_sets: dict[tuple[str, str], ModelClassComponents]) -> float:
    reference = component_sets[("fixed_template", "restricted_midpoint")]
    fields = ("Z", "D", "alpha_w", "alpha_y", "lowrank_w", "shock_w", "shock_y", "eps_w", "eps_y")
    maximum = 0.0
    for components in component_sets.values():
        for field in fields:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            np.asarray(getattr(components, field), dtype=float)
                            - np.asarray(getattr(reference, field), dtype=float)
                        )
                    )
                ),
            )
    return maximum


def run_diagnostic(
    protocol: dict[str, Any],
    *,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if reps <= 0:
        raise ValueError("reps must be positive.")
    verify_reproduction_context(protocol)
    reference = protocol["production_reference"]
    T0 = int(reference["T0"])
    T1 = int(reference["T1"])
    config = str(reference["configuration"])
    targets = target_map(protocol)
    geometries = build_geometries(protocol)
    exposure = np.asarray(production._load_restricted_empirical_panel()["exposure"], dtype=float)
    records: list[dict[str, Any]] = []
    weights_bank: dict[tuple[str, str, int, str, str], np.ndarray] = {}
    maximum_common_component_difference = 0.0

    for rep in range(int(reps)):
        component_sets = {
            key: draw_components(protocol, rep=rep, geometry=geometry, exposure=exposure)
            for key, geometry in geometries.items()
        }
        maximum_common_component_difference = max(
            maximum_common_component_difference,
            common_component_difference(component_sets),
        )
        for anchor_mode in ANCHOR_MODES:
            for role, tau_true in targets.items():
                components = component_sets[(anchor_mode, role)]
                for branch in BRANCHES:
                    panel = build_branch_panel(
                        components,
                        branch=branch,
                        tau_true=tau_true,
                        T0=T0,
                        T1=T1,
                    )
                    first_stage_slopes = unit_level_slope_vector(
                        panel.W[:, :T0], components.Z[:T0]
                    )
                    estimate = estimate_branch_panel(
                        panel,
                        components.Z,
                        first_stage_slopes=first_stage_slopes,
                    )
                    post = slice(T0, T0 + T1)
                    shock_on = BRANCH_SWITCHES[branch][1]
                    active_shock = components.shock_y if shock_on else np.zeros_like(components.shock_y)
                    for estimator, weights in (
                        ("TSLS", estimate.weights_tsls),
                        ("SIV", estimate.weights_robust),
                    ):
                        weights = np.asarray(weights, dtype=float)
                        weights_bank[(anchor_mode, role, rep, branch, estimator)] = weights.copy()
                        y = aggregate_series(panel.Y[:, post], weights)
                        w = aggregate_series(panel.W[:, post], weights)
                        z = np.asarray(components.Z, dtype=float)[post]
                        moment = iv_moment_fields(y, w, z, tau_true=tau_true)
                        component = component_biases(
                            components,
                            panel,
                            weights,
                            tau_true=tau_true,
                            T0=T0,
                            T1=T1,
                        )
                        shock = shock_bias_component(
                            shock_y=active_shock,
                            W=panel.W,
                            Z=components.Z,
                            weights=weights,
                            T0=T0,
                            T1=T1,
                        )
                        base = {
                            "anchor_mode": anchor_mode,
                            "target_role": role,
                            "tau_true": float(tau_true),
                            "rep": int(rep),
                            "config": config,
                            "T0": T0,
                            "T1": T1,
                            "branch": branch,
                            "estimator": estimator,
                            "weight_corr": float(estimate.weight_correlation),
                            "weight_mean": float(np.mean(weights)),
                            "weight_exposure_constraint": float((weights @ first_stage_slopes) / float(weights.size)),
                            "weight_max_abs": float(np.max(np.abs(weights))),
                            "shock": float(shock),
                            **moment,
                            **component,
                        }

                        arima = raw_z_arp_design_based_se(
                            y,
                            w,
                            z,
                            z_process=np.asarray(components.Z, dtype=float),
                            ar_order=active.ARIMA_AR_ORDER,
                            varz_mode="window",
                        )
                        arima_bias = float(arima.tau - tau_true)
                        records.append(
                            {
                                **base,
                                "method": "ARIMA-Z",
                                "tau": float(arima.tau),
                                "bias": arima_bias,
                                "abs_error": abs(arima_bias),
                                "squared_error": arima_bias * arima_bias,
                                "se": float(arima.se),
                                "covered": int(abs(arima_bias) <= _crit("ARIMA-Z", T1) * float(arima.se)),
                                "pi": float(arima.pi),
                                "critical": _crit("ARIMA-Z", T1),
                            }
                        )

                        hac = hac_delta_se(y, w, z, maxlags=active.HAC_LAGS)
                        hac_bias = float(hac.tau - tau_true)
                        records.append(
                            {
                                **base,
                                "method": "HAC",
                                "tau": float(hac.tau),
                                "bias": hac_bias,
                                "abs_error": abs(hac_bias),
                                "squared_error": hac_bias * hac_bias,
                                "se": float(hac.se),
                                "covered": int(abs(hac_bias) <= _crit("HAC", T1) * float(hac.se)),
                                "pi": float(hac.pi),
                                "critical": _crit("HAC", T1),
                            }
                        )

                        ar = ar_style_test(y, w, z, tau_true, maxlags=active.HAC_LAGS)
                        records.append(
                            {
                                **base,
                                "method": "AR",
                                "tau": float("nan"),
                                "bias": float("nan"),
                                "abs_error": float("nan"),
                                "squared_error": float("nan"),
                                "se": float("nan"),
                                "covered": int(abs(float(ar["t_stat"])) <= _crit("AR", T1)),
                                "pi": float("nan"),
                                "critical": _crit("AR", T1),
                            }
                        )

    raw = pd.DataFrame.from_records(records)
    weight_contrasts = build_weight_contrasts(protocol, weights_bank, reps=reps)
    runtime = {
        "maximum_common_random_component_difference": maximum_common_component_difference,
        "records": len(raw),
        "weight_contrast_records": len(weight_contrasts),
    }
    return raw, weight_contrasts, runtime


def build_weight_contrasts(
    protocol: dict[str, Any],
    weights_bank: dict[tuple[str, str, int, str, str], np.ndarray],
    *,
    reps: int,
) -> pd.DataFrame:
    midpoint = midpoint_role(protocol)
    endpoint_roles = [role for role in target_map(protocol) if role != midpoint]
    rows: list[dict[str, Any]] = []
    for anchor_mode in ANCHOR_MODES:
        for role in endpoint_roles:
            for rep in range(int(reps)):
                for branch in BRANCHES:
                    for estimator in ("TSLS", "SIV"):
                        candidate = weights_bank[(anchor_mode, role, rep, branch, estimator)]
                        baseline = weights_bank[(anchor_mode, midpoint, rep, branch, estimator)]
                        difference = candidate - baseline
                        rms = float(np.sqrt(np.mean(difference * difference)))
                        baseline_rms = float(np.sqrt(np.mean(baseline * baseline)))
                        correlation = float(np.corrcoef(candidate, baseline)[0, 1])
                        rows.append(
                            {
                                "anchor_mode": anchor_mode,
                                "target_role": role,
                                "midpoint_role": midpoint,
                                "rep": rep,
                                "branch": branch,
                                "estimator": estimator,
                                "weight_correlation_with_midpoint": correlation,
                                "weight_rms_distance": rms,
                                "weight_relative_rms_distance": rms / baseline_rms,
                                "weight_max_abs_difference": float(np.max(np.abs(difference))),
                            }
                        )
    return pd.DataFrame.from_records(rows)


def mcse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")


def rmse_difference_mcse(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.size <= 1 or reference.size != candidate.size:
        return float("nan")
    mse_candidate = float(np.mean(candidate * candidate))
    mse_reference = float(np.mean(reference * reference))
    rmse_candidate = math.sqrt(mse_candidate)
    rmse_reference = math.sqrt(mse_reference)
    if rmse_candidate <= 0.0 or rmse_reference <= 0.0:
        return float("nan")
    influence = (
        (candidate * candidate - mse_candidate) / (2.0 * rmse_candidate)
        - (reference * reference - mse_reference) / (2.0 * rmse_reference)
    )
    return mcse(influence)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "anchor_mode",
        "target_role",
        "tau_true",
        "config",
        "T0",
        "T1",
        "branch",
        "estimator",
        "method",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        errors = group["bias"].to_numpy(dtype=float)
        finite_errors = errors[np.isfinite(errors)]
        standard_errors = group["se"].to_numpy(dtype=float)
        finite_se = standard_errors[np.isfinite(standard_errors)]
        coverage = float(group["covered"].mean())
        mean_bias = float(np.mean(finite_errors)) if finite_errors.size else float("nan")
        row.update(
            {
                "n": len(group),
                "coverage": coverage,
                "coverage_mcse": math.sqrt(coverage * (1.0 - coverage) / float(len(group))),
                "mean_bias": mean_bias,
                "absolute_bias": abs(mean_bias) if math.isfinite(mean_bias) else float("nan"),
                "mean_abs_error": float(np.mean(np.abs(finite_errors))) if finite_errors.size else float("nan"),
                "rmse": float(math.sqrt(float(np.mean(finite_errors * finite_errors)))) if finite_errors.size else float("nan"),
                "mean_se": float(np.mean(finite_se)) if finite_se.size else float("nan"),
                "sd_error": float(np.std(finite_errors, ddof=1)) if finite_errors.size > 1 else float("nan"),
                "mean_weight_corr": float(group["weight_corr"].mean()),
                "mean_weight_max_abs": float(group["weight_max_abs"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def estimator_differences(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "anchor_mode",
        "target_role",
        "tau_true",
        "config",
        "T0",
        "T1",
        "branch",
        "method",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        tsls = group[group["estimator"] == "TSLS"].sort_values("rep")
        robust = group[group["estimator"] == "SIV"].sort_values("rep")
        if not np.array_equal(tsls["rep"].to_numpy(), robust["rep"].to_numpy()):
            raise AssertionError("Estimator pairing failed.")
        containment_difference = robust["covered"].to_numpy(dtype=float) - tsls["covered"].to_numpy(dtype=float)
        point_available = str(row["method"]) in POINT_METHODS
        if point_available:
            error_tsls = tsls["bias"].to_numpy(dtype=float)
            error_robust = robust["bias"].to_numpy(dtype=float)
            signed_difference = error_robust - error_tsls
            absolute_error_difference = np.abs(error_robust) - np.abs(error_tsls)
            mean_bias_tsls = float(np.mean(error_tsls))
            mean_bias_robust = float(np.mean(error_robust))
            rmse_tsls = float(math.sqrt(float(np.mean(error_tsls * error_tsls))))
            rmse_robust = float(math.sqrt(float(np.mean(error_robust * error_robust))))
        else:
            error_tsls = np.array([], dtype=float)
            error_robust = np.array([], dtype=float)
            signed_difference = np.array([], dtype=float)
            absolute_error_difference = np.array([], dtype=float)
            mean_bias_tsls = mean_bias_robust = rmse_tsls = rmse_robust = float("nan")
        row.update(
            {
                "n_pairs": len(tsls),
                "coverage_tsls": float(tsls["covered"].mean()),
                "coverage_robust": float(robust["covered"].mean()),
                "delta_coverage_robust_minus_tsls": float(np.mean(containment_difference)),
                "mcse_delta_coverage": mcse(containment_difference),
                "bias_tsls": mean_bias_tsls,
                "bias_robust": mean_bias_robust,
                "delta_signed_bias_robust_minus_tsls": float(np.mean(signed_difference)) if signed_difference.size else float("nan"),
                "mcse_delta_signed_bias": mcse(signed_difference),
                "absolute_bias_tsls": abs(mean_bias_tsls) if math.isfinite(mean_bias_tsls) else float("nan"),
                "absolute_bias_robust": abs(mean_bias_robust) if math.isfinite(mean_bias_robust) else float("nan"),
                "delta_absolute_bias_robust_minus_tsls": (
                    abs(mean_bias_robust) - abs(mean_bias_tsls)
                    if math.isfinite(mean_bias_tsls) and math.isfinite(mean_bias_robust)
                    else float("nan")
                ),
                "delta_mean_abs_error_robust_minus_tsls": float(np.mean(absolute_error_difference)) if absolute_error_difference.size else float("nan"),
                "mcse_delta_mean_abs_error": mcse(absolute_error_difference),
                "rmse_tsls": rmse_tsls,
                "rmse_robust": rmse_robust,
                "delta_rmse_robust_minus_tsls": rmse_robust - rmse_tsls if point_available else float("nan"),
                "mcse_delta_rmse": rmse_difference_mcse(error_robust, error_tsls),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def target_contrasts(protocol: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    midpoint = midpoint_role(protocol)
    endpoint_roles = [role for role in target_map(protocol) if role != midpoint]
    rows: list[dict[str, Any]] = []
    for anchor_mode in ANCHOR_MODES:
        for endpoint_role in endpoint_roles:
            for branch in BRANCHES:
                for estimator in ("TSLS", "SIV"):
                    for method in METHODS:
                        selection = raw[
                            (raw["anchor_mode"] == anchor_mode)
                            & (raw["branch"] == branch)
                            & (raw["estimator"] == estimator)
                            & (raw["method"] == method)
                        ]
                        midpoint_rows = selection[selection["target_role"] == midpoint].sort_values("rep")
                        endpoint_rows = selection[selection["target_role"] == endpoint_role].sort_values("rep")
                        if not np.array_equal(midpoint_rows["rep"].to_numpy(), endpoint_rows["rep"].to_numpy()):
                            raise AssertionError("Target pairing failed.")
                        coverage_difference = (
                            endpoint_rows["covered"].to_numpy(dtype=float)
                            - midpoint_rows["covered"].to_numpy(dtype=float)
                        )
                        if method in POINT_METHODS:
                            midpoint_error = midpoint_rows["bias"].to_numpy(dtype=float)
                            endpoint_error = endpoint_rows["bias"].to_numpy(dtype=float)
                            bias_difference = endpoint_error - midpoint_error
                            abs_error_difference = np.abs(endpoint_error) - np.abs(midpoint_error)
                            rmse_midpoint = float(math.sqrt(float(np.mean(midpoint_error * midpoint_error))))
                            rmse_endpoint = float(math.sqrt(float(np.mean(endpoint_error * endpoint_error))))
                        else:
                            midpoint_error = endpoint_error = np.array([], dtype=float)
                            bias_difference = abs_error_difference = np.array([], dtype=float)
                            rmse_midpoint = rmse_endpoint = float("nan")
                        rows.append(
                            {
                                "anchor_mode": anchor_mode,
                                "endpoint_role": endpoint_role,
                                "midpoint_role": midpoint,
                                "branch": branch,
                                "estimator": estimator,
                                "method": method,
                                "n_pairs": len(midpoint_rows),
                                "delta_mean_bias_endpoint_minus_midpoint": float(np.mean(bias_difference)) if bias_difference.size else float("nan"),
                                "mcse_delta_mean_bias": mcse(bias_difference),
                                "delta_mean_abs_error_endpoint_minus_midpoint": float(np.mean(abs_error_difference)) if abs_error_difference.size else float("nan"),
                                "mcse_delta_mean_abs_error": mcse(abs_error_difference),
                                "rmse_midpoint": rmse_midpoint,
                                "rmse_endpoint": rmse_endpoint,
                                "delta_rmse_endpoint_minus_midpoint": rmse_endpoint - rmse_midpoint if method in POINT_METHODS else float("nan"),
                                "mcse_delta_rmse": rmse_difference_mcse(endpoint_error, midpoint_error),
                                "coverage_midpoint": float(midpoint_rows["covered"].mean()),
                                "coverage_endpoint": float(endpoint_rows["covered"].mean()),
                                "delta_coverage_endpoint_minus_midpoint": float(np.mean(coverage_difference)),
                                "mcse_delta_coverage": mcse(coverage_difference),
                            }
                        )
    return pd.DataFrame.from_records(rows)


def summarize_weight_contrasts(weight_contrasts: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["anchor_mode", "target_role", "midpoint_role", "branch", "estimator"]
    metric_columns = [
        "weight_correlation_with_midpoint",
        "weight_rms_distance",
        "weight_relative_rms_distance",
        "weight_max_abs_difference",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in weight_contrasts.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["n"] = len(group)
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
            row[f"max_{metric}"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def numeric_difference_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: list[str],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for column in columns:
        observed = left[column].to_numpy(dtype=float)
        expected = right[column].to_numpy(dtype=float)
        finite = np.isfinite(observed) & np.isfinite(expected)
        if not np.array_equal(np.isnan(observed), np.isnan(expected)):
            raise AssertionError(f"NaN pattern mismatch for {column}.")
        if finite.any():
            difference = np.abs(observed[finite] - expected[finite])
            scale = np.maximum(np.abs(expected[finite]), 1.0)
            maximum_absolute = max(maximum_absolute, float(np.max(difference)))
            maximum_relative = max(maximum_relative, float(np.max(difference / scale)))
            allowed = absolute_tolerance + relative_tolerance * np.abs(expected[finite])
            if np.any(difference > allowed):
                index = int(np.argmax(difference - allowed))
                raise AssertionError(
                    f"Numeric mismatch for {column}: observed={observed[finite][index]}, "
                    f"expected={expected[finite][index]}, difference={difference[index]}, "
                    f"allowed={allowed[index]}."
                )
    return maximum_absolute, maximum_relative


def structural_checks(
    protocol: dict[str, Any],
    raw: pd.DataFrame,
    *,
    reps: int,
    runtime: dict[str, Any],
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
) -> dict[str, Any]:
    equivalence = protocol["numerical_equivalence"]
    absolute_tolerance = (
        float(equivalence["absolute_tolerance"])
        if absolute_tolerance is None
        else float(absolute_tolerance)
    )
    relative_tolerance = (
        float(equivalence["relative_tolerance"])
        if relative_tolerance is None
        else float(relative_tolerance)
    )
    reference = protocol["production_reference"]
    midpoint = midpoint_role(protocol)
    targets = target_map(protocol)
    expected_rows = int(reps) * len(ANCHOR_MODES) * len(targets) * len(BRANCHES) * 2 * len(METHODS)
    if len(raw) != expected_rows:
        raise AssertionError(f"Raw row count {len(raw)} != {expected_rows}.")
    duplicate_keys = ["anchor_mode", "target_role", *KEY_COLUMNS]
    if raw.duplicated(duplicate_keys).any():
        raise AssertionError("Duplicate raw keys detected.")
    if set(raw["anchor_mode"]) != set(ANCHOR_MODES):
        raise AssertionError("Anchor-mode set mismatch.")
    if set(raw["target_role"]) != set(targets):
        raise AssertionError("Target-role set mismatch.")
    if set(raw["branch"]) != set(BRANCHES):
        raise AssertionError("Branch set mismatch.")
    if set(raw["method"]) != set(METHODS):
        raise AssertionError("Inference-method set mismatch.")
    if not raw["covered"].isin([0, 1]).all():
        raise AssertionError("Coverage indicator is not binary.")
    if float(np.max(np.abs(raw["weight_mean"].to_numpy(dtype=float)))) > 1e-8:
        raise AssertionError("Weight centering constraint failed.")
    if float(np.max(np.abs(raw["weight_exposure_constraint"].to_numpy(dtype=float) - 1.0))) > 1e-8:
        raise AssertionError("Weight exposure constraint failed.")

    sort_columns = KEY_COLUMNS
    baseline = raw[
        (raw["anchor_mode"] == "fixed_template")
        & (raw["target_role"] == midpoint)
    ].sort_values(sort_columns).reset_index(drop=True)
    reference_rows = pd.read_csv(ROOT / str(reference["output_directory"]) / "raw.csv")
    reference_rows = reference_rows[
        (reference_rows["config"] == reference["configuration"])
        & (reference_rows["rep"] < int(reps))
    ].sort_values(sort_columns).reset_index(drop=True)
    if not baseline[sort_columns].equals(reference_rows[sort_columns]):
        raise AssertionError("Baseline simulation keys do not match the principal reference.")
    baseline_difference, baseline_relative_difference = numeric_difference_diagnostics(
        baseline,
        reference_rows,
        REFERENCE_NUMERIC_COLUMNS,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not np.array_equal(baseline["covered"].to_numpy(), reference_rows["covered"].to_numpy()):
        raise AssertionError("Baseline coverage indicators differ from the principal reference.")

    fixed_midpoint = baseline.sort_values(sort_columns).reset_index(drop=True)
    reanchored_midpoint = raw[
        (raw["anchor_mode"] == "reanchored_template")
        & (raw["target_role"] == midpoint)
    ].sort_values(sort_columns).reset_index(drop=True)
    midpoint_difference, midpoint_relative_difference = numeric_difference_diagnostics(
        fixed_midpoint,
        reanchored_midpoint,
        REFERENCE_NUMERIC_COLUMNS,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    if not np.array_equal(fixed_midpoint["covered"].to_numpy(), reanchored_midpoint["covered"].to_numpy()):
        raise AssertionError("Anchor-mode coverage differs at the midpoint.")

    tsls = raw[
        (raw["anchor_mode"] == "fixed_template")
        & (raw["estimator"] == "TSLS")
    ]
    invariance_columns = [
        "bias",
        "se",
        "pi",
        "weight_mean",
        "weight_exposure_constraint",
        "weight_max_abs",
        "shock",
        "iv_moment_tau",
        "iv_moment_abs_tau",
        "iv_cov_zw",
        "iv_moment_bias_ratio",
        "bias_component_total",
        "bias_component_gfe",
        "bias_component_shock",
        "bias_component_eps",
    ]
    tsls_maximum = 0.0
    tsls_relative_maximum = 0.0
    tsls_coverage_equal = True
    for endpoint in [role for role in targets if role != midpoint]:
        midpoint_rows = tsls[tsls["target_role"] == midpoint].sort_values(sort_columns).reset_index(drop=True)
        endpoint_rows = tsls[tsls["target_role"] == endpoint].sort_values(sort_columns).reset_index(drop=True)
        endpoint_maximum, endpoint_relative_maximum = numeric_difference_diagnostics(
            midpoint_rows,
            endpoint_rows,
            invariance_columns,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        tsls_maximum = max(tsls_maximum, endpoint_maximum)
        tsls_relative_maximum = max(tsls_relative_maximum, endpoint_relative_maximum)
        tsls_coverage_equal = tsls_coverage_equal and np.array_equal(
            midpoint_rows["covered"].to_numpy(), endpoint_rows["covered"].to_numpy()
        )
    if not tsls_coverage_equal:
        raise AssertionError("Fixed-template TSLS coverage is not invariant across targets.")
    if float(runtime["maximum_common_random_component_difference"]) > absolute_tolerance:
        raise AssertionError("Common random components differ across target cells.")

    return {
        "status": "pass",
        "reps_checked": int(reps),
        "raw_rows": len(raw),
        "expected_raw_rows": expected_rows,
        "baseline_max_abs_difference": baseline_difference,
        "baseline_max_relative_difference": baseline_relative_difference,
        "midpoint_anchor_mode_max_abs_difference": midpoint_difference,
        "midpoint_anchor_mode_max_relative_difference": midpoint_relative_difference,
        "fixed_template_tsls_max_abs_invariance_difference": tsls_maximum,
        "fixed_template_tsls_max_relative_invariance_difference": tsls_relative_maximum,
        "fixed_template_tsls_coverage_invariant": tsls_coverage_equal,
        "maximum_common_random_component_difference": float(runtime["maximum_common_random_component_difference"]),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "production_reference_unchanged": True,
    }


def write_outputs(
    protocol: dict[str, Any],
    *,
    output: Path,
    raw: pd.DataFrame,
    weight_contrasts: pd.DataFrame,
    reps: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(raw)
    differences = estimator_differences(raw)
    target_difference = target_contrasts(protocol, raw)
    weight_summary = summarize_weight_contrasts(weight_contrasts)
    checks = structural_checks(protocol, raw, reps=reps, runtime=runtime)
    frames = {
        "raw.csv": raw,
        "summary.csv": summary,
        "estimator_differences.csv": differences,
        "target_contrasts.csv": target_difference,
        "weight_target_contrasts.csv": weight_contrasts,
        "weight_target_contrast_summary.csv": weight_summary,
    }
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    (output / "invariance_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    context = verify_reproduction_context(protocol)
    manifest = {
        "analysis": protocol["protocol"],
        "execution_scope": "full" if int(reps) == int(protocol["paired_execution"]["reps"]) else "structural_smoke",
        "interpret_results": bool(int(reps) == int(protocol["paired_execution"]["reps"])),
        "reps_requested": int(reps),
        "seed": int(protocol["paired_execution"]["seed"]),
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "reproduction_context": context,
        "runtime": runtime,
        "structural_checks": checks,
        "principal_outputs_modified": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_files = [*frames, "invariance_checks.json"]
    checksum_text = "\n".join(
        f"{sha256_file(output / filename)}  {filename}" for filename in checksum_files
    )
    (output / "SHA256SUMS").write_text(checksum_text + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--execute-full",
        action="store_true",
        help="Run the protocol's prespecified 1,000-replication analysis.",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    full_reps = int(protocol["paired_execution"]["reps"])
    if args.execute_full:
        if args.reps is not None and int(args.reps) != full_reps:
            parser.error(f"--execute-full requires --reps {full_reps} or no --reps argument.")
        reps = full_reps
    else:
        reps = 3 if args.reps is None else int(args.reps)
        if reps >= full_reps:
            parser.error("Full execution requires the explicit --execute-full flag.")
    if args.out is not None:
        output = args.out
    elif args.execute_full:
        output = ROOT / str(protocol["output_directory"])
    else:
        output = ROOT / f"{protocol['output_directory']}_smoke"
    raw, weight_contrasts, runtime = run_diagnostic(protocol, reps=reps)
    manifest = write_outputs(
        protocol,
        output=output,
        raw=raw,
        weight_contrasts=weight_contrasts,
        reps=reps,
        runtime=runtime,
    )
    print(json.dumps({
        "status": "pass",
        "output": str(output),
        "execution_scope": manifest["execution_scope"],
        "reps": reps,
        "structural_checks": manifest["structural_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
